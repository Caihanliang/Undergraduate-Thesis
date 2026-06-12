"""
python train_moment.py
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from momentfm import MOMENTPipeline
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class TrafficFlowDataset(Dataset):
    """Custom Dataset for traffic flow data with 4 features"""
    
    def __init__(self, input_data, target_data):
        """
        Args:
            input_data: [samples, seq_len, n_features] - Input sequences
            target_data: [samples, pred_len, n_features] - Target sequences
        """
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return {
            'x_enc': torch.FloatTensor(self.input_data[idx]),  # [seq_len, n_features]
            'y': torch.FloatTensor(self.target_data[idx])  # [pred_len, n_features]
        }

def load_processed_data():
    """Load preprocessed data (independent train/val sets, no splitting performed here)"""
    print("Loading processed data...")
    
    # Load training set (pre-split)
    train_data = np.load('moment_data/train_dataset.npz')
    print(f"  ✓ Train data keys: {list(train_data.keys())}")
    
    # Load validation set (pre-split)
    val_data = np.load('moment_data/val_dataset.npz')
    print(f"  ✓ Val data keys: {list(val_data.keys())}")
    
    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        norm_params = json.load(f)
    
    # Load station mapping
    with open('station_mapping.json', 'r', encoding='utf-8') as f:
        station_mapping = json.load(f)
    
    return train_data, val_data, norm_params, station_mapping

def create_model(n_features, forecast_horizon=96, seq_len=8):
    """Create and initialize MOMENT model for forecasting with Fine-tuning mode"""
    print(f"\nCreating MOMENT model:")
    print(f"  Features (n_channels): {n_features}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Forecast horizon: {forecast_horizon}")
    print(f"  Mode: Fine-tuning (unfreeze encoder and embedder)")
    
    # Set mirror endpoint for faster download in China
    import os
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"  ✓ Using HuggingFace mirror: {os.environ['HF_ENDPOINT']}")
    
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", 
        model_kwargs={
            'task_name': 'forecasting',
            'forecast_horizon': forecast_horizon,
            'n_channels': n_features,  # Number of features (stations * 4 features)
            'seq_len': seq_len,  # IMPORTANT: Must match actual input sequence length
            'head_dropout': 0.1,
            'weight_decay': 0,
            # Fine-tuning mode: Unfreeze all components
            'freeze_encoder': False,    # Unfreeze transformer encoder
            'freeze_embedder': False,   # Unfreeze patch embedding
            'freeze_head': False,       # Train forecasting head
        },
    )
    
    model.init()
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count parameters by component
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    embedder_params = sum(p.numel() for p in model.patch_embedding.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"\nModel statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Training ratio: {trainable_params/total_params*100:.2f}%")
    print(f"\nParameter breakdown:")
    print(f"  Encoder (T5): {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"  Embedder: {embedder_params:,} ({embedder_params/total_params*100:.1f}%)")
    print(f"  Forecasting Head: {head_params:,} ({head_params/total_params*100:.1f}%)")
    print(f"\n✓ Fine-tuning mode enabled - All components will be trained")
    
    return model


def create_optimizer(model, encoder_lr=1e-5, head_lr=1e-4, weight_decay=1e-4):
    """
    Create optimizer with layer-wise learning rates for fine-tuning
    
    Strategy:
    - Encoder (pretrained): Slow learning rate to avoid catastrophic forgetting
    - Head (new): Faster learning rate for quick adaptation
    """
    # Separate parameters by component
    encoder_params = []
    embedder_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'patch_embedding' in name or 'embedder' in name:
            embedder_params.append(param)
        elif 'head' in name or 'forecasting' in name:
            head_params.append(param)
        else:
            # Default to encoder learning rate
            encoder_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': weight_decay,
        },
        {
            'params': embedder_params,
            'lr': encoder_lr,  # Embedder also uses slow LR
            'weight_decay': weight_decay,
        },
        {
            'params': head_params,
            'lr': head_lr,
            'weight_decay': weight_decay,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    
    print(f"\nOptimizer configuration:")
    print(f"  Encoder/Embedder LR: {encoder_lr} ({len(encoder_params) + len(embedder_params)} params)")
    print(f"  Head LR: {head_lr} ({len(head_params)} params)")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Optimizer: AdamW")
    
    return optimizer

def train_epoch(model, dataloader, optimizer, criterion, device, gradient_clipping=1.0):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        x_enc = batch['x_enc'].to(device)  # [batch, seq_len, n_features]
        y = batch['y'].to(device)  # [batch, pred_len, n_features]
        
        # MOMENT expects input as [batch, n_channels, seq_len], so we need to transpose
        x_enc = x_enc.permute(0, 2, 1)  # [batch, n_features, seq_len]
        y = y.permute(0, 2, 1)  # [batch, n_features, pred_len]
        
        # Forward pass
        output = model(x_enc=x_enc)
        predictions = output.forecast  # [batch, n_features, pred_len]
        
        # Transpose predictions back to [batch, pred_len, n_features] for loss calculation
        predictions = predictions.permute(0, 2, 1)
        
        # Calculate loss (MSE)
        loss = criterion(predictions, y.permute(0, 2, 1))
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  Warning: NaN or Inf loss detected! Skipping this batch.")
            print(f"  Predictions stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
            print(f"  Targets stats: min={y.permute(0,2,1).min():.4f}, max={y.permute(0,2,1).max():.4f}, mean={y.permute(0,2,1).mean():.4f}")
            continue
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        if gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    return avg_loss

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            x_enc = batch['x_enc'].to(device)
            y = batch['y'].to(device)
            
            # Transpose to match MOMENT's expected input format
            x_enc = x_enc.permute(0, 2, 1)  # [batch, n_features, seq_len]
            y_original = y  # Keep original for metrics calculation
            
            output = model(x_enc=x_enc)
            predictions = output.forecast  # [batch, n_features, pred_len]
            
            # Transpose predictions back to [batch, pred_len, n_features]
            predictions = predictions.permute(0, 2, 1)
            
            loss = criterion(predictions, y)
            
            total_loss += loss.item()
            n_batches += 1
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / n_batches
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate MAE
    mae = np.mean(np.abs(all_predictions - all_targets))
    
    return avg_loss, mae, all_predictions, all_targets

def plot_feature_predictions(predictions, targets, sample_idx=0, station_idx=0, 
                           save_path=None, station_mapping=None):
    """
    Plot predictions vs targets for all 4 features of a specific station
    
    Args:
        predictions: [samples, pred_len, n_features]
        targets: [samples, pred_len, n_features]
        sample_idx: Which sample to plot
        station_idx: Which station to plot
        save_path: Path to save the figure
        station_mapping: Station index to name mapping
    """
    # Set font to support Chinese characters
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # Get station info
    station_info = station_mapping.get(str(station_idx), {})
    station_name = station_info.get('station_name', f'Station {station_idx}')
    station_code = station_info.get('station_code', '')
    
    # Calculate feature indices for this station
    # Features are organized as: [feat0_station0, feat1_station0, ..., feat0_station1, ...]
    base_feat_idx = station_idx * 4
    
    # Use English feature names to avoid font issues
    feature_names = ['Passenger Car Up', 'Passenger Car Down', 
                     'Non-Passenger Car Up', 'Non-Passenger Car Down']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Predictions vs Actual - {station_name} ({station_code})\nSample {sample_idx}', 
                 fontsize=14, fontweight='bold')
    
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
    
    for i, feat_name in enumerate(feature_names):
        ax = axes[i // 2][i % 2]
        feat_idx = base_feat_idx + i
        
        pred = predictions[sample_idx, :, feat_idx]
        target = targets[sample_idx, :, feat_idx]
        
        # Denormalize for better visualization (optional)
        # You can add denormalization here if needed
        
        time_steps = range(len(pred))
        
        ax.plot(time_steps, target, label='Actual', linewidth=2.5, color=colors[i], alpha=0.8)
        ax.plot(time_steps, pred, label='Predicted', linewidth=2.5, linestyle='--', 
                color=colors[i], alpha=0.6)
        
        # Calculate metrics for this feature
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Normalized Flow', fontsize=11)
        ax.set_title(f'{feat_name}\nMAE: {mae:.3f} | RMSE: {rmse:.3f}', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add minor ticks
        ax.minorticks_on()
        ax.grid(which='minor', alpha=0.2, linestyle=':')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    
    plt.show()
    plt.close()

def main():
    # Configuration - Fine-tuning mode for better performance
    SEQ_LEN = 8    # Input sequence length (hours) - must match preprocessing
    PRED_LEN = 8   # Prediction horizon (hours) - must match preprocessing
    
    # Fine-tuning configuration
    BATCH_SIZE = 8          # Smaller batch for fine-tuning
    EPOCHS = 30             # More epochs for fine-tuning
    ENCODER_LR = 1e-5       # Slow learning rate for pretrained encoder
    HEAD_LR = 1e-4          # Faster learning rate for new forecasting head
    WEIGHT_DECAY = 1e-4     # Strong regularization
    GRADIENT_CLIPPING = 1.0 # Prevent gradient explosion
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("MOMENT Traffic Flow Forecasting - Fine-tuning Mode")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Mode: Fine-tuning (unfreeze encoder)")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Prediction horizon: {PRED_LEN}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Encoder LR: {ENCODER_LR}")
    print(f"  Head LR: {HEAD_LR}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  Gradient clipping: {GRADIENT_CLIPPING}")
    
    # Load data - 使用独立的训练集和验证集
    train_data, val_data, norm_params, station_mapping = load_processed_data()
    
    # 从NPZ文件中提取数据
    train_input = train_data['input']
    train_target = train_data['target']
    val_input = val_data['input']
    val_target = val_data['target']
    
    print(f"\nData loaded:")
    print(f"  Train input shape: {train_input.shape}")
    print(f"  Train target shape: {train_target.shape}")
    print(f"  Val input shape: {val_input.shape}")
    print(f"  Val target shape: {val_target.shape}")
    
    # Validate data is not empty
    if train_input.size == 0 or train_target.size == 0:
        raise ValueError(
            "❌ Loaded training data is empty! Please run preprocessing first:\n"
            "   python preprocess_data.py\n"
            "Check moment_data/train_dataset.npz exists and contains valid data."
        )
    
    if val_input.size == 0 or val_target.size == 0:
        raise ValueError(
            "❌ Loaded validation data is empty! Please run preprocessing first:\n"
            "   python preprocess_data.py\n"
            "Check moment_data/val_dataset.npz exists and contains valid data."
        )
    
    # Get number of features
    n_features = train_input.shape[2]
    n_stations = n_features // 4
    
    print(f"  Number of stations: {n_stations}")
    print(f"  Features per station: 4 (小客车上/下行, 非小客车上/下行)")
    print(f"  Total features: {n_features}")
    
    print(f"\nData statistics (loaded from pre-processed files):")
    print(f"  Train samples: {len(train_input)}")
    print(f"  Val samples: {len(val_input)}")
    
    # Create datasets
    train_dataset = TrafficFlowDataset(train_input, train_target)
    val_dataset = TrafficFlowDataset(val_input, val_target)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Create model - Fine-tuning mode
    model = create_model(n_features=n_features, forecast_horizon=PRED_LEN, seq_len=SEQ_LEN)
    model = model.to(DEVICE)
    
    # Create optimizer with layer-wise learning rates
    optimizer = create_optimizer(
        model, 
        encoder_lr=ENCODER_LR, 
        head_lr=HEAD_LR, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=EPOCHS // 3,  # Restart every 1/3 of epochs
        T_mult=2,
        eta_min=ENCODER_LR / 10  # Minimum learning rate
    )
    
    criterion = torch.nn.MSELoss()
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*60)
        
        # Train with gradient clipping
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, 
                                gradient_clipping=GRADIENT_CLIPPING)
        print(f"  Train Loss: {train_loss:.6f}")
        
        # Validate
        val_loss, val_mae, predictions, targets = validate(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val MAE: {val_mae:.6f}")
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_mae'].append(val_mae)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Current Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': {
                    'n_features': n_features,
                    'forecast_horizon': PRED_LEN,
                    'seq_len': SEQ_LEN
                }
            }, 'checkpoints/best_model.pth')
            print(f"\n  ✓ Saved best model (val_loss: {val_loss:.6f}, val_mae: {val_mae:.6f})")
        
        # Plot some predictions at the last epoch
        if epoch == EPOCHS - 1:
            print("\nGenerating prediction plots...")
            # Plot first 3 samples for first 2 stations
            for sample_idx in range(min(3, len(predictions))):
                for station_idx in range(min(2, n_stations)):
                    plot_feature_predictions(
                        predictions, targets,
                        sample_idx=sample_idx,
                        station_idx=station_idx,
                        save_path=f'results/prediction_sample{sample_idx}_station{station_idx}.png',
                        station_mapping=station_mapping
                    )
    
    # Save training history - convert numpy types to Python native types
    os.makedirs('results', exist_ok=True)
    
    # Convert numpy float32 to Python float for JSON serialization
    training_history_serializable = {
        'train_loss': [float(x) for x in training_history['train_loss']],
        'val_loss': [float(x) for x in training_history['val_loss']],
        'val_mae': [float(x) for x in training_history['val_mae']]
    }
    
    with open('results/training_history.json', 'w') as f:
        json.dump(training_history_serializable, f, indent=2)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to results/")
    print("="*60)

if __name__ == "__main__":
    main()