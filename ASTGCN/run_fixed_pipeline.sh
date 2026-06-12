#!/bin/bash
# ASTGCN 多变量预测完整修复流程

echo "============================================================"
echo "ASTGCN 多变量预测 - 完整修复流程"
echo "============================================================"

cd /home/user/Downloads/cai/ASTGCN

echo ""
echo "步骤1: 清理旧数据..."
rm -f dataset/train.npz dataset/val.npz dataset/test.npz
echo "✅ 旧数据已清理"

echo ""
echo "步骤2: 重新预处理数据（归一化y）..."
python preprocess_highway_data.py \
    --dataset_dir ./dataset \
    --input_len 8 \
    --output_len 8 \
    --adj_method distance

if [ $? -ne 0 ]; then
    echo "❌ 数据预处理失败"
    exit 1
fi
echo "✅ 数据预处理完成"

echo ""
echo "步骤3: 验证数据格式..."
python verify_data.py

if [ $? -ne 0 ]; then
    echo "❌ 数据验证失败"
    exit 1
fi
echo "✅ 数据验证通过"

echo ""
echo "步骤4: 清理旧模型..."
rm -rf experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03
echo "✅ 旧模型已清理"

echo ""
echo "步骤5: 开始训练..."
python train_ASTGCN_r.py --config configurations/highway_traffic.conf

if [ $? -ne 0 ]; then
    echo "❌ 训练失败"
    exit 1
fi
echo "✅ 训练完成"

echo ""
echo "步骤6: 生成训练集预测结果..."
python predict_and_save_train.py --config configurations/highway_traffic.conf

if [ $? -ne 0 ]; then
    echo "❌ 预测结果生成失败"
    exit 1
fi
echo "✅ 预测结果生成完成"

echo ""
echo "============================================================"
echo "✅ 全部流程完成！"
echo "============================================================"
echo ""
echo "结果文件位置:"
echo "  - 最佳模型: experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/epoch_*.params"
echo "  - 测试评估: experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/output_epoch_*_test.npz"
echo "  - 训练预测: experiments/highway_traffic/astgcn_r_h1d0w0_channel4_1.000000e-03/train_predictions.csv"
echo ""
