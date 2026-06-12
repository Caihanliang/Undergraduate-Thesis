#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOMENT Train Predictions Visualization - Clean version without annotations
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========== Path Configuration ==========
PROJECT_ROOT = "/home/user/Downloads/cai/moment-main/moment-main"
os.chdir(PROJECT_ROOT)

# ========== Configuration ==========
CONFIG = {
    'predictions_csv': 'visualization-results/predictions/train_predictions.csv',
    'station_mapping': 'station_mapping.json',
    'output_dir': 'visualization-results/train_figures',
    'figsize': (14, 5),
    'dpi': 150,
    'start_date': '2023-09-01 08:00:00',  # Start time for predictions
    'time_step_hours': 1,
}

# Feature names - English only
FEATURE_NAMES = {
    'Passenger_Car_Up': 'Passenger Car Up',
    'Passenger_Car_Down': 'Passenger Car Down',
    'Non_Passenger_Car_Up': 'Non-Passenger Car Up',
    'Non_Passenger_Car_Down': 'Non-Passenger Car Down'
}


def load_data():
    """Load and prepare data"""
    print("\n" + "="*60)
    print("MOMENT Train Predictions Visualization")
    print("="*60)
    
    print(f"\nLoading predictions from: {CONFIG['predictions_csv']}")
    df = pd.read_csv(CONFIG['predictions_csv'])
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def parse_time_from_samples(df):
    """Extract real time from Sample_X strings"""
    print("\nParsing time information...")
    
    # Extract sample numbers
    df['sample_num'] = df['时间'].str.extract(r'Sample_(\d+)')[0].astype(int)
    
    # Calculate real datetime: Sample_0 -> 2023-09-01 08:00:00
    start_date = pd.Timestamp(CONFIG['start_date'])
    df['datetime'] = start_date + pd.to_timedelta(df['sample_num'] * CONFIG['time_step_hours'], unit='h')
    
    stations = sorted(df['站点编号'].unique())
    features = sorted(df['特征'].unique())
    
    print(f"Stations: {len(stations)}")
    print(f"Features: {len(features)} - {features}")
    
    samples_per_station = df.groupby(['站点编号', '特征']).size().iloc[0]
    print(f"Samples per (station, feature): {samples_per_station}")
    print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df, stations, features


def plot_station_timeseries(df, station, feature, output_dir):
    """Plot time series for a single station and feature - no annotations"""
    
    station_df = df[(df['站点编号'] == station) & (df['特征'] == feature)].copy()
    
    if station_df.empty:
        return None, None
    
    station_df = station_df.sort_values('datetime')
    
    time_vals = station_df['datetime'].values
    true_vals = station_df['真实值'].values
    pred_vals = station_df['预测值'].values
    
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    
    # Create figure - similar to FaST style
    fig, ax = plt.subplots(figsize=CONFIG['figsize'])
    
    # Plot with FaST-like styling
    ax.plot(time_vals, true_vals, label='Actual', linewidth=1.8, 
            color='#1f77b4', alpha=0.8)
    ax.plot(time_vals, pred_vals, label='Predicted', linewidth=1.8, 
            color='#ff4b5c', linestyle='--', alpha=0.8)
    
    feature_name = FEATURE_NAMES.get(feature, feature)
    station_name = f"Station_{station:03d}"
    
    ax.set_title(f"{station_name} - {feature_name}", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Traffic Flow', fontsize=11)
    
    # Format x-axis - similar to FaST
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    feature_dir = feature_name.replace(' ', '_')
    save_dir = os.path.join(output_dir, 'figures', 'all_stations', feature_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'station_{station:03d}.png')
    plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    
    return mae, rmse


def plot_overview(df, features, output_dir):
    """Generate overview plots (mean across all stations) - similar to FaST"""
    print("\nGenerating overview plots...")
    
    overview_dir = os.path.join(output_dir, 'figures', 'overview')
    os.makedirs(overview_dir, exist_ok=True)
    
    for feature in features:
        feature_df = df[df['特征'] == feature]
        
        # Calculate mean across stations for each time step
        mean_by_time = feature_df.groupby('datetime').agg({
            '真实值': 'mean',
            '预测值': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        time_vals = mean_by_time['datetime'].values
        true_mean = mean_by_time['真实值'].values
        pred_mean = mean_by_time['预测值'].values
        
        # Plot mean values
        ax.plot(time_vals, true_mean, label='True', linewidth=2, 
               color='#1f77b4', alpha=0.8)
        ax.plot(time_vals, pred_mean, label='Pred', linewidth=2, 
               color='#ff4b5c', linestyle='--', alpha=0.8)
        
        feature_name = FEATURE_NAMES.get(feature, feature)
        
        # Calculate overall metrics
        mae = mean_absolute_error(true_mean, pred_mean)
        rmse = np.sqrt(mean_squared_error(true_mean, pred_mean))
        
        ax.set_title(f"[train] {feature_name} Traffic Overview", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Traffic Flow (veh/h)', fontsize=12)
        
        # Format x-axis like FaST
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        
        plt.tight_layout()
        
        save_path = os.path.join(overview_dir, f'{feature_name.replace(" ", "_")}_overview.png')
        plt.savefig(save_path, dpi=CONFIG['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {feature_name.replace(' ', '_')}_overview.png")


def save_metrics(all_metrics, output_dir):
    """Save metrics to JSON"""
    metrics_path = os.path.join(output_dir, 'metrics.json')
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")


def main():
    """Main function"""
    
    df = load_data()
    df, stations, features = parse_time_from_samples(df)
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    all_metrics = {}
    
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)
    
    for feature in features:
        feature_name = FEATURE_NAMES.get(feature, feature)
        print(f"\nProcessing: {feature_name}")
        
        feature_metrics = {}
        for station in stations:
            mae, rmse = plot_station_timeseries(df, station, feature, CONFIG['output_dir'])
            if mae is not None:
                feature_metrics[f'Station_{station:03d}'] = {'MAE': round(float(mae), 3), 'RMSE': round(float(rmse), 3)}
            
            if station % 20 == 0 and station > 0:
                print(f"  Processed {station}/{stations[-1]} stations")
        
        if feature_metrics:
            avg_mae = np.mean([m['MAE'] for m in feature_metrics.values()])
            avg_rmse = np.mean([m['RMSE'] for m in feature_metrics.values()])
            
            all_metrics[feature] = {
                'average_MAE': round(float(avg_mae), 3),
                'average_RMSE': round(float(avg_rmse), 3),
                'per_station': feature_metrics
            }
            
            print(f"  Feature {feature_name} - Avg MAE: {avg_mae:.3f}, Avg RMSE: {avg_rmse:.3f}")
    
    plot_overview(df, features, CONFIG['output_dir'])
    save_metrics(all_metrics, CONFIG['output_dir'])
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print(f"  - Station plots: figures/all_stations/")
    print(f"  - Overview plots: figures/overview/")
    print(f"  - Metrics: metrics.json")
    print("\nDone!")


if __name__ == "__main__":
    main()