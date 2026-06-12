#!/usr/bin/env python3
"""
LLM微调质量诊断工具

功能:
1. 分析训练日志，评估收敛质量
2. 检查推理结果，评估预测性能
3. 生成可视化报告
4. 提供优化建议

使用方法:
    python 诊断训练质量.py [--mode train|inference|both]
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ====================== 配置 ======================
PROJECT_ROOT = "/home/user/Downloads/cai/FaST-main-8_8MO63Delete/FaST-main"
TRAINING_LOG_DIR = os.path.join(PROJECT_ROOT, "config", "cai")
INFERENCE_RESULT_DIR = os.path.join(PROJECT_ROOT, "config", "cai", 
                                     "inference_results_all_stations_with_predictions")

# ====================== 训练质量分析 ======================
def analyze_training_quality():
    """分析训练日志，评估模型收敛质量"""
    print("=" * 90)
    print("📊 训练质量分析")
    print("=" * 90)
    
    # 查找训练日志文件
    log_files = list(Path(TRAINING_LOG_DIR).glob("quick518_optimized_*.txt"))
    
    if not log_files:
        print("❌ 未找到训练日志文件")
        return None
    
    # 使用最新的日志
    latest_log = sorted(log_files)[-1]
    print(f"📄 分析日志: {latest_log.name}")
    
    # 解析日志
    losses = []
    grad_norms = []
    learning_rates = []
    epochs = []
    
    with open(latest_log, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配loss行
            match = re.search(r"\{'loss': ([\d.]+), 'grad_norm': ([\d.]+), "
                            r"'learning_rate': ([\de.-]+), 'epoch': ([\d.]+)\}", line)
            if match:
                loss = float(match.group(1))
                grad_norm = float(match.group(2))
                lr = float(match.group(3))
                epoch = float(match.group(4))
                
                losses.append(loss)
                grad_norms.append(grad_norm)
                learning_rates.append(lr)
                epochs.append(epoch)
    
    if not losses:
        print("❌ 日志中未找到有效的训练指标")
        return None
    
    # 统计分析
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    avg_grad_norm = np.mean(grad_norms[-100:])  # 最后100步的平均梯度
    max_grad_norm = np.max(grad_norms)
    
    print(f"\n✅ 训练数据统计:")
    print(f"   总步数: {len(losses)}")
    print(f"   最终Epoch: {epochs[-1]:.2f}")
    print(f"   初始Loss: {initial_loss:.4f}")
    print(f"   最终Loss: {final_loss:.4f}")
    print(f"   Loss下降幅度: {loss_reduction:.1f}%")
    print(f"   平均梯度范数(最后100步): {avg_grad_norm:.4f}")
    print(f"   最大梯度范数: {max_grad_norm:.4f}")
    
    # 质量评估
    print(f"\n🎯 质量评估:")
    
    # Loss评估
    if final_loss < 0.15:
        print(f"   ✅ Loss收敛: 优秀 ({final_loss:.4f} < 0.15)")
        loss_score = 5
    elif final_loss < 0.25:
        print(f"   ⚠️  Loss收敛: 良好 ({final_loss:.4f} < 0.25)")
        loss_score = 4
    else:
        print(f"   ❌ Loss收敛: 需优化 ({final_loss:.4f} > 0.25)")
        loss_score = 2
    
    # Loss下降幅度评估
    if loss_reduction > 90:
        print(f"   ✅ Loss下降幅度: 优秀 ({loss_reduction:.1f}% > 90%)")
    elif loss_reduction > 70:
        print(f"   ⚠️  Loss下降幅度: 良好 ({loss_reduction:.1f}% > 70%)")
    else:
        print(f"   ❌ Loss下降幅度: 不足 ({loss_reduction:.1f}% < 70%)")
    
    # 梯度稳定性评估
    if 0.1 <= avg_grad_norm <= 1.0 and max_grad_norm < 5.0:
        print(f"   ✅ 梯度稳定性: 优秀 (avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f})")
        grad_score = 5
    elif max_grad_norm < 10.0:
        print(f"   ⚠️  梯度稳定性: 良好 (avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f})")
        grad_score = 4
    else:
        print(f"   ❌ 梯度稳定性: 需优化 (avg={avg_grad_norm:.4f}, max={max_grad_norm:.4f})")
        grad_score = 2
    
    # 综合评分
    overall_score = (loss_score + grad_score) / 2
    print(f"\n🏆 综合评分: {overall_score:.1f}/5.0")
    
    if overall_score >= 4.5:
        print("   🎉 训练质量: 优秀！可以进行推理测试")
    elif overall_score >= 3.5:
        print("   👍 训练质量: 良好，建议进行推理验证")
    else:
        print("   ⚠️  训练质量: 一般，建议调整超参数重新训练")
    
    # 生成可视化
    generate_training_plots(epochs, losses, grad_norms, learning_rates)
    
    return {
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'avg_grad_norm': avg_grad_norm,
        'overall_score': overall_score
    }


def generate_training_plots(epochs, losses, grad_norms, learning_rates):
    """生成训练过程可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss曲线
    axes[0, 0].plot(epochs, losses, linewidth=2, color='steelblue')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.15, color='green', linestyle='--', label='Excellent (<0.15)')
    axes[0, 0].axhline(y=0.25, color='orange', linestyle='--', label='Good (<0.25)')
    axes[0, 0].legend()
    
    # 2. 梯度范数曲线
    axes[0, 1].plot(epochs, grad_norms, linewidth=2, color='coral')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Gradient Norm', fontsize=12)
    axes[0, 1].set_title('Gradient Norm Stability', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', label='Stable (<1.0)')
    axes[0, 1].axhline(y=5.0, color='red', linestyle='--', label='Unstable (>5.0)')
    axes[0, 1].legend()
    
    # 3. 学习率曲线
    axes[1, 0].plot(epochs, learning_rates, linewidth=2, color='mediumseagreen')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. Loss下降速率
    loss_diff = np.diff(losses)
    smooth_window = min(50, len(loss_diff))
    if smooth_window > 0:
        smoothed_diff = np.convolve(loss_diff, np.ones(smooth_window)/smooth_window, mode='valid')
        smoothed_epochs = epochs[smooth_window:]
        axes[1, 1].plot(smoothed_epochs, smoothed_diff, linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss Change per Step', fontsize=12)
        axes[1, 1].set_title('Loss Convergence Rate', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(TRAINING_LOG_DIR, "training_quality_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 训练曲线图已保存: {plot_path}")
    plt.close()


# ====================== 推理质量分析 ======================
def analyze_inference_quality():
    """分析推理结果，评估LLM预测性能"""
    print("\n" + "=" * 90)
    print("🔍 推理质量分析")
    print("=" * 90)
    
    # 检查结果文件
    metrics_file = os.path.join(INFERENCE_RESULT_DIR, "global_overall_mean_metrics.csv")
    compare_file = os.path.join(INFERENCE_RESULT_DIR, "llm_gnn_prediction_compare.csv")
    
    if not os.path.exists(metrics_file):
        print("❌ 推理结果文件不存在，请先运行推理脚本")
        return None
    
    # 读取全局指标
    df_metrics = pd.read_csv(metrics_file)
    print(f"\n📄 全局汇总指标:")
    print(df_metrics.to_string(index=False))
    
    # 提取关键指标
    if len(df_metrics) > 0:
        row = df_metrics.iloc[0]
        gnn_mae = row['GNN_MAE']
        llm_mae = row['LLM_MAE']
        mae_improvement = row['MAE_improvement']
        
        mae_improvement_pct = (mae_improvement / gnn_mae * 100) if gnn_mae > 0 else 0
        
        print(f"\n🎯 LLM vs GNN 对比:")
        print(f"   GNN MAE: {gnn_mae:.4f}")
        print(f"   LLM MAE: {llm_mae:.4f}")
        print(f"   绝对改善: {mae_improvement:.4f}")
        print(f"   相对改善: {mae_improvement_pct:.1f}%")
        
        # 质量评估
        print(f"\n🏆 推理质量评估:")
        
        if mae_improvement_pct > 30:
            print(f"   🎉 MAE改善: 优秀 ({mae_improvement_pct:.1f}% > 30%)")
            inference_score = 5
        elif mae_improvement_pct > 10:
            print(f"   👍 MAE改善: 良好 ({mae_improvement_pct:.1f}% > 10%)")
            inference_score = 4
        elif mae_improvement_pct > 0:
            print(f"   ⚠️  MAE改善: 一般 ({mae_improvement_pct:.1f}% > 0%)")
            inference_score = 3
        else:
            print(f"   ❌ MAE改善: 失败 ({mae_improvement_pct:.1f}% ≤ 0%)")
            inference_score = 1
        
        # 检查fallback率（如果有详细数据）
        if os.path.exists(compare_file):
            df_compare = pd.read_csv(compare_file)
            total_samples = len(df_compare)
            fallback_count = df_compare['is_fallback'].sum()
            fallback_rate = (fallback_count / total_samples * 100) if total_samples > 0 else 0
            
            print(f"\n📊 解析成功率:")
            print(f"   总样本数: {total_samples}")
            print(f"   Fallback次数: {fallback_count}")
            print(f"   Fallback率: {fallback_rate:.2f}%")
            
            if fallback_rate < 5:
                print(f"   ✅ Fallback率: 优秀 ({fallback_rate:.2f}% < 5%)")
            elif fallback_rate < 10:
                print(f"   ⚠️  Fallback率: 良好 ({fallback_rate:.2f}% < 10%)")
            else:
                print(f"   ❌ Fallback率: 需优化 ({fallback_rate:.2f}% ≥ 10%)")
        
        return {
            'gnn_mae': gnn_mae,
            'llm_mae': llm_mae,
            'mae_improvement_pct': mae_improvement_pct,
            'inference_score': inference_score
        }
    
    return None


# ====================== 站点级分析 ======================
def analyze_station_performance():
    """分析各站点的性能分布"""
    print("\n" + "=" * 90)
    print("📍 站点级性能分析")
    print("=" * 90)
    
    station_file = os.path.join(INFERENCE_RESULT_DIR, "all_station_feature_mean_metrics.csv")
    
    if not os.path.exists(station_file):
        print("❌ 站点级指标文件不存在")
        return None
    
    df = pd.read_csv(station_file)
    
    # 统计各特征的改善情况
    features = df['feature_id'].unique()
    
    print(f"\n📊 各特征平均改善幅度:")
    for feat_id in sorted(features):
        feat_data = df[df['feature_id'] == feat_id]
        avg_gnn_mae = feat_data['GNN_MAE'].mean()
        avg_llm_mae = feat_data['LLM_MAE'].mean()
        improvement = ((avg_gnn_mae - avg_llm_mae) / avg_gnn_mae * 100) if avg_gnn_mae > 0 else 0
        
        feat_name = feat_data['feature_name'].iloc[0]
        print(f"   Feature {feat_id} ({feat_name}): {improvement:.1f}% 改善")
    
    # 找出表现最好和最差的站点
    df['improvement'] = df['GNN_MAE'] - df['LLM_MAE']
    
    top_5 = df.nlargest(5, 'improvement')
    bottom_5 = df.nsmallest(5, 'improvement')
    
    print(f"\n🏆 Top 5 改善最大的站点-特征组合:")
    for _, row in top_5.iterrows():
        print(f"   站点{row['station_id']} ({row['station_short_name']}) - "
              f"Feature {row['feature_id']}: 改善 {row['improvement']:.2f}")
    
    print(f"\n⚠️  Top 5 改善最小的站点-特征组合:")
    for _, row in bottom_5.iterrows():
        print(f"   站点{row['station_id']} ({row['station_short_name']}) - "
              f"Feature {row['feature_id']}: 改善 {row['improvement']:.2f}")
    
    # 生成站点性能分布图
    plt.figure(figsize=(12, 6))
    plt.hist(df['improvement'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Improvement')
    plt.xlabel('MAE Improvement (GNN - LLM)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Station-Level MAE Improvement', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(INFERENCE_RESULT_DIR, "station_improvement_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 站点性能分布图已保存: {plot_path}")
    plt.close()
    
    return df


# ====================== 综合报告 ======================
def generate_comprehensive_report():
    """生成综合质量评估报告"""
    print("\n" + "=" * 90)
    print("📋 综合质量评估报告")
    print("=" * 90)
    
    # 分析训练质量
    train_result = analyze_training_quality()
    
    # 分析推理质量
    inference_result = analyze_inference_quality()
    
    # 分析站点性能
    station_df = analyze_station_performance()
    
    # 综合评分
    print(f"\n" + "=" * 90)
    print("🏆 最终评估结论")
    print("=" * 90)
    
    if train_result and inference_result:
        train_score = train_result['overall_score']
        inference_score = inference_result['inference_score']
        final_score = (train_score + inference_score) / 2
        
        print(f"\n训练质量评分: {train_score:.1f}/5.0")
        print(f"推理质量评分: {inference_score:.1f}/5.0")
        print(f"综合评分: {final_score:.1f}/5.0")
        
        if final_score >= 4.5:
            print("\n🎉🎉🎉 整体评价: 优秀！")
            print("   ✅ 训练充分收敛")
            print("   ✅ LLM显著优于GNN基线")
            print("   ✅ 可以投入生产使用")
        elif final_score >= 3.5:
            print("\n👍 整体评价: 良好")
            print("   ✅ 训练质量可接受")
            print("   ⚠️  LLM有一定改善但仍有优化空间")
            print("   💡 建议: 尝试增加样本数或调整超参数")
        else:
            print("\n⚠️  整体评价: 需优化")
            print("   ❌ 训练或推理存在问题")
            print("   💡 建议:")
            print("      - 检查Prompt是否与微调一致")
            print("      - 调整temperature或batch size")
            print("      - 考虑重新微调模型")
    
    # 保存报告
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training': train_result,
        'inference': inference_result,
        'recommendation': '优秀' if final_score >= 4.5 else ('良好' if final_score >= 3.5 else '需优化')
    }
    
    report_path = os.path.join(TRAINING_LOG_DIR, "quality_assessment_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 完整报告已保存: {report_path}")


# ====================== 主函数 ======================
if __name__ == "__main__":
    print("🚀 LLM微调质量诊断工具")
    print("=" * 90)
    
    try:
        generate_comprehensive_report()
    except Exception as e:
        print(f"\n❌ 诊断过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()