"""
FaST-MV 快速启动脚本
一键完成：数据准备 → 模型测试 → 训练启动
"""
import os
import sys
from pathlib import Path

# 添加路径
script_dir = Path(__file__).parent
main_master_dir = script_dir.parent
sys.path.insert(0, str(main_master_dir))

def step1_prepare_data():
    """步骤1：准备多变量数据"""
    print("\n" + "=" * 70)
    print("步骤 1/3：准备多变量数据集")
    print("=" * 70)
    
    prepare_script = script_dir / "prepare_multivariate_data.py"
    
    if not prepare_script.exists():
        print("❌ 找不到 prepare_multivariate_data.py")
        return False
    
    # 运行数据准备脚本
    import subprocess
    result = subprocess.run(
        [sys.executable, str(prepare_script)],
        cwd=str(script_dir)
    )
    
    if result.returncode == 0:
        print("\n✅ 数据准备完成！")
        return True
    else:
        print("\n❌ 数据准备失败！")
        return False


def step2_test_model():
    """步骤2：测试模型"""
    print("\n" + "=" * 70)
    print("步骤 2/3：测试 FaST-MV 模型")
    print("=" * 70)
    
    # 直接导入测试
    try:
        import torch
        from FaST.arch.fast_arch_mv import FaST_MV
        
        print("\n🔧 创建模型...")
        model = FaST_MV(
            num_nodes=160,
            num_features=2,
            input_len=8,
            output_len=8,
            layers=3,
            num_experts=8,
            hidden_dim=64,
            num_agent=32,
            use_revIN=True,
            channel_independent=True
        )
        
        print(f"✅ 模型创建成功！参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        print("\n🧪 运行前向传播测试...")
        test_input = torch.randn(2, 8, 160, 2)
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"✅ 前向传播成功！")
        print(f"   输入: {test_input.shape}")
        print(f"   输出: {output.shape}")
        
        assert output.shape == (2, 8, 160, 2), "输出形状错误！"
        print("\n✅ 模型测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def step3_start_training():
    """步骤3：启动训练"""
    print("\n" + "=" * 70)
    print("步骤 3/3：启动训练")
    print("=" * 70)
    
    config_file = script_dir / "HNGS_8_8MV.py"
    
    if not config_file.exists():
        print(f"❌ 找不到配置文件: {config_file}")
        return False
    
    print(f"\n📝 配置文件: {config_file.name}")
    print("\n💡 训练命令:")
    print(f"cd {main_master_dir}")
    print(f"python experiments/train_seed.py -c FaST/HNGS_8_8MV.py -g 0")
    
    # 询问是否立即开始训练
    response = input("\n是否现在开始训练？(y/n): ").strip().lower()
    
    if response == 'y':
        print("\n🚀 开始训练...")
        train_script = main_master_dir / "experiments" / "train_seed.py"
        
        import subprocess
        result = subprocess.run(
            [sys.executable, str(train_script), "-c", "FaST/HNGS_8_8MV.py", "-g", "0"],
            cwd=str(main_master_dir)
        )
        
        if result.returncode == 0:
            print("\n✅ 训练完成！")
            return True
        else:
            print("\n❌ 训练失败！")
            return False
    else:
        print("\n⏸️  训练已跳过。你可以稍后手动运行上面的命令。")
        return True


def main():
    print("\n" + "🚀" * 35)
    print("FaST-MV 快速启动")
    print("多变量交通流量预测（小客车 + 非小客车）")
    print("🚀" * 35)
    
    # 步骤1：准备数据
    if not step1_prepare_data():
        print("\n❌ 数据准备失败，流程中止")
        sys.exit(1)
    
    # 步骤2：测试模型
    if not step2_test_model():
        print("\n❌ 模型测试失败，流程中止")
        sys.exit(1)
    
    # 步骤3：启动训练
    if not step3_start_training():
        print("\n❌ 训练启动失败")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("🎉 所有步骤完成！")
    print("=" * 70)
    print("\n📚 相关文档:")
    print(f"   - 使用指南: {script_dir / 'README_FAST_MV.md'}")
    print(f"   - 模型代码: {script_dir / 'arch' / 'fast_arch_mv.py'}")
    print(f"   - 配置文件: {script_dir / 'HNGS_8_8MV.py'}")
    print("\n💡 提示:")
    print("   - 训练日志保存在 checkpoints/ 目录")
    print("   - 使用 tensorboard 查看训练曲线")
    print("   - 训练完成后可运行推理脚本进行测试")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
