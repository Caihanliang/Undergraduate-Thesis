#!/usr/bin/env python3
"""
TimesFM 2.5 微调脚本 - 针对高速公路交通流量预测（8输入8输出）

使用 LoRA 进行参数高效微调，预测4维交通流量特征：
1. 小客车上行
2. 小客车下行
3. 非小客车上行 (汽车自然数 - 小客车)
4. 非小客车下行 (汽车自然数 - 小客车)

时序配置：8输入8输出（使用过去8小时预测未来8小时）

基于 HuggingFace Transformers + PEFT
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入配置
import config

# 设置 HuggingFace 镜像加速
os.environ['HF_ENDPOINT'] = config.HF_ENDPOINT

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(config.TRAINING_LOG, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据集类
# ============================================================================

class TrafficTimeSeriesDataset(Dataset):
    """
    交通流量时间序列随机窗口数据集（8输入8输出）
    
    采用与 Chronos-2 类似的随机切片策略，每个样本是一个完整的 context_len 窗口
    （无零填充），避免破坏 TimesFM 内部的 RevIN 归一化统计。
    
    无需外部归一化 - TimesFM 内部处理实例归一化。
    """
    
    def __init__(
        self,
        series_list: list,
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples = []
        
        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        
        # 过滤出足够长的序列
        valid_indices = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        
        if not valid_indices:
            raise ValueError(
                f"没有序列足够长以满足 context_len={context_len} + horizon_len={horizon_len}。"
                f"最短序列长度: {min(len(s) for s in series_list)}"
            )
        
        # 随机采样窗口
        for _ in range(num_samples):
            idx = rng.choice(valid_indices)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))
        
        logger.info(f"数据集创建完成: {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        series_idx, start = self.samples[idx]
        series = self.series_list[series_idx]
        end = start + self.context_len + self.horizon_len
        
        # 提取上下文和目标
        context = torch.tensor(
            series[start:start + self.context_len], dtype=torch.float32
        )
        target = torch.tensor(
            series[start + self.context_len:end], dtype=torch.float32
        )
        
        return context, target


class TrafficValidationDataset(Dataset):
    """验证数据集 - 使用每个序列的最后窗口"""
    
    def __init__(self, series_list: list, context_len: int, horizon_len: int):
        self.items = []
        min_len = context_len + horizon_len
        
        for series in series_list:
            if len(series) >= min_len:
                context = torch.tensor(
                    series[-min_len:-horizon_len], dtype=torch.float32
                )
                target = torch.tensor(
                    series[-horizon_len:], dtype=torch.float32
                )
                self.items.append((context, target))
        
        logger.info(f"验证集创建完成: {len(self.items)} 个样本")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


# ============================================================================
# 数据加载
# ============================================================================

def load_traffic_data(data_dir: str, context_len: int, horizon_len: int, 
                     num_samples: int, seed: int):
    """
    加载预处理后的交通流量数据
    
    Returns:
        train_dataset, val_dataset
    """
    logger.info(f"从 {data_dir} 加载交通流量数据...")
    
    # 加载训练序列
    train_data = np.load(os.path.join(data_dir, 'train_series.npz'), 
                        allow_pickle=True)
    train_series = train_data['data'].tolist()
    
    # 加载验证序列
    val_data = np.load(os.path.join(data_dir, 'val_series.npz'), 
                      allow_pickle=True)
    val_series = val_data['data'].tolist()
    
    # 加载元数据
    with open(os.path.join(data_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"加载了 {len(train_series)} 个训练序列")
    logger.info(f"加载了 {len(val_series)} 个验证序列")
    logger.info(f"特征名称: {metadata['feature_names']}")
    logger.info(f"配置: {metadata.get('config', 'unknown')}")
    
    # 创建数据集
    train_dataset = TrafficTimeSeriesDataset(
        train_series, context_len, horizon_len, 
        num_samples=num_samples, seed=seed
    )
    
    val_dataset = TrafficValidationDataset(
        val_series, context_len, horizon_len
    )
    
    return train_dataset, val_dataset, metadata


# ============================================================================
# 训练函数
# ============================================================================

def train(args):
    """执行微调训练"""
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # ------------------------------------------------------------------
    # 1. 加载模型
    # ------------------------------------------------------------------
    logger.info(f"加载模型: {args.model_id}")
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    horizon_len = args.horizon_len
    context_len = min(args.context_len, model.config.context_length)
    
    # TimesFM 2.5 使用 Patch 机制，patch_len=32
    # 输入长度必须是 patch_len 的倍数
    patch_len = 32
    if context_len < patch_len:
        logger.warning(f"⚠️  警告: context_len={context_len} 小于 patch_len={patch_len}")
        logger.warning(f"   TimesFM 2.5 要求输入长度 >= {patch_len}")
        logger.warning(f"   自动调整 context_len 从 {context_len} -> {patch_len}")
        context_len = patch_len
    
    # 确保 context_len 是 patch_len 的倍数
    if context_len % patch_len != 0:
        adjusted_context_len = ((context_len // patch_len) + 1) * patch_len
        logger.warning(f"⚠️  警告: context_len={context_len} 不是 patch_len={patch_len} 的倍数")
        logger.warning(f"   自动调整 context_len 从 {context_len} -> {adjusted_context_len}")
        context_len = adjusted_context_len
    
    logger.info(f"实际使用的 context_len: {context_len}, horizon_len: {horizon_len}")
    logger.info(f"配置: {context_len}输入{horizon_len}输出")
    logger.info(f"Patch 长度: {patch_len}")
    
    # ------------------------------------------------------------------
    # 2. 应用 LoRA
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ------------------------------------------------------------------
    # 3. 准备数据
    # ------------------------------------------------------------------
    train_dataset, val_dataset, metadata = load_traffic_data(
        args.data_dir, context_len, horizon_len, 
        num_samples=args.num_samples, seed=args.seed
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    logger.info(
        f"训练样本: {len(train_dataset)} ({len(train_loader)} batches) | "
        f"验证样本: {len(val_dataset)}"
    )
    
    # ------------------------------------------------------------------
    # 4. 优化器和学习率调度器
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    # ------------------------------------------------------------------
    # 5. 训练循环
    # ------------------------------------------------------------------
    best_val_loss = float('inf')
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'epochs': [],
        'config': f'{context_len}_input_{horizon_len}_output'
    }
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (context, target_vals) in enumerate(train_loader):
            context = context.to(device)
            target_vals = target_vals.to(device)
            
            # 前向传播
            outputs = model(
                past_values=context,
                future_values=target_vals,
                forecast_context_len=context_len,
            )
            loss = outputs.loss
            
            # 检查 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Batch {batch_idx}: Loss is NaN/Inf, skipping...")
                optimizer.zero_grad()
                continue
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # 打印进度
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Epoch {epoch}/{args.epochs} [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(device)
                target_vals = target_vals.to(device)
                
                outputs = model(
                    past_values=context,
                    future_values=target_vals,
                    forecast_context_len=context_len,
                )
                val_loss += outputs.loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        
        # 记录历史
        training_history['train_losses'].append(float(avg_train_loss))
        training_history['val_losses'].append(float(avg_val_loss))
        training_history['epochs'].append(epoch)
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} — "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            output_path = args.output_dir
            model.save_pretrained(output_path)
            logger.info(f"✓ 保存最佳适配器到: {output_path}")
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    
    return best_val_loss


# ============================================================================
# 评估函数
# ============================================================================

def evaluate(args):
    """评估微调后的模型"""
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("加载基础模型...")
    base_model = TimesFm2_5ModelForPrediction.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()
    
    horizon_len = args.horizon_len
    context_len = min(args.context_len, base_model.config.context_length)
    
    logger.info(f"从 {args.output_dir} 加载 LoRA 适配器...")
    ft_model = PeftModel.from_pretrained(base_model, args.output_dir)
    ft_model.eval()
    
    # 加载测试数据
    test_data = np.load(os.path.join(args.data_dir, 'test_series.npz'), 
                       allow_pickle=True)
    test_series = test_data['data'].tolist()
    
    with open(os.path.join(args.data_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    logger.info(f"测试序列数: {len(test_series)}")
    logger.info(f"配置: {context_len}输入{horizon_len}输出")
    
    # 评估
    base_maes = []
    ft_maes = []
    
    min_len = context_len + horizon_len
    
    for i, series in enumerate(test_series[:20]):  # 评估前20个序列
        if len(series) < min_len:
            continue
        
        # 使用最后窗口进行测试
        test_input = series[-min_len:-horizon_len]
        ground_truth = series[-horizon_len:]
        
        test_tensor = torch.tensor(
            test_input, dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            # 零样本预测
            base_out = base_model(past_values=test_tensor)
            base_forecast = base_out.mean_predictions[0, :horizon_len].float().cpu().numpy()
            
            # 微调后预测
            ft_out = ft_model(past_values=test_tensor)
            ft_forecast = ft_out.mean_predictions[0, :horizon_len].float().cpu().numpy()
        
        base_mae = float(np.abs(base_forecast - ground_truth).mean())
        ft_mae = float(np.abs(ft_forecast - ground_truth).mean())
        
        base_maes.append(base_mae)
        ft_maes.append(ft_mae)
        
        logger.info(
            f"序列 {i} — Zero-shot MAE: {base_mae:.2f}, Fine-tuned MAE: {ft_mae:.2f}"
        )
    
    if base_maes:
        avg_base = np.mean(base_maes)
        avg_ft = np.mean(ft_maes)
        improvement = (avg_base - avg_ft) / avg_base * 100
        
        logger.info("=" * 60)
        logger.info(f"平均 Zero-shot MAE: {avg_base:.2f}")
        logger.info(f"平均 Fine-tuned MAE: {avg_ft:.2f}")
        logger.info(f"改进幅度: {improvement:.1f}%")
        logger.info("=" * 60)


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='TimesFM 2.5 微调 - 高速公路交通流量预测（8输入8输出）'
    )
    
    # 模型配置
    parser.add_argument(
        '--model_id',
        default=config.MODEL_ID,
        help='HuggingFace 模型 ID'
    )
    
    # 数据配置
    parser.add_argument(
        '--data_dir',
        default=config.PREPROCESSED_DIR,
        help='预处理数据目录'
    )
    parser.add_argument('--context_len', type=int, default=config.CONTEXT_LEN,
                       help=f'上下文长度（输入窗口，默认{config.CONTEXT_LEN}小时）')
    parser.add_argument('--horizon_len', type=int, default=config.HORIZON_LEN,
                       help=f'预测 horizon（默认{config.HORIZON_LEN}小时）')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--num_samples', type=int, default=config.NUM_SAMPLES,
                       help='预采样的训练窗口数')
    
    # LoRA 配置
    parser.add_argument('--lora_r', type=int, default=config.LORA_R,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=config.LORA_ALPHA,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=config.LORA_DROPOUT,
                       help='LoRA dropout')
    
    # 输出配置
    parser.add_argument(
        '--output_dir',
        default=config.CHECKPOINT_DIR,
        help='LoRA 适配器保存路径'
    )
    parser.add_argument('--seed', type=int, default=config.SEED,
                       help='随机种子')
    
    # 模式选择
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='仅评估，跳过训练'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建输出目录
    config.ensure_directories()
    
    logger.info("=" * 60)
    logger.info("TimesFM 微调训练 - 8输入8输出配置")
    logger.info("=" * 60)
    logger.info(f"输入窗口: {args.context_len} 小时")
    logger.info(f"预测窗口: {args.horizon_len} 小时")
    logger.info("=" * 60)
    
    if not args.eval_only:
        train(args)
    
    # 评估
    if os.path.isdir(args.output_dir):
        logger.info("\n" + "=" * 60)
        logger.info("开始评估")
        logger.info("=" * 60)
        evaluate(args)
    else:
        logger.warning(f"在 {args.output_dir} 未找到适配器，跳过评估")


if __name__ == '__main__':
    main()
