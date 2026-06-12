"""
带 mask（掩码）的损失函数
用于时空预测 / 交通预测
支持处理 NaN 或指定的无效值（null_val）
自动分批次计算，防止显存爆炸
"""
import numpy as np
import torch

def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    """
    带掩码的平均绝对误差（MAE）
    忽略无效值（null_val / NaN）
    大张量自动分batch，防止OOM
    """
    device = prediction.device   # 获取当前设备（CPU/GPU）
    # 累计损失与有效权重
    total_loss = 0.0
    total_weight = 0.0

    # 样本总数
    num_samples = prediction.size(0)
    # 最大分批大小，防止显存爆炸
    batch_size_limit = 64
    # 如果样本数 <= 64，直接计算
    if num_samples <= batch_size_limit:
        # 生成掩码：标记哪些是有效值  
        if np.isnan(null_val):
            # 忽略 NaN
            mask = ~torch.isnan(target)
        else:
            # 忽略指定的无效值（如 0）
            eps = 5e-5
            null_tensor = torch.full_like(target, null_val, device=device)
            mask = ~torch.isclose(target, null_tensor, atol=eps, rtol=0.0)
        
        # 转浮点型并归一化掩码
        mask = mask.float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        # 防止出现 NaN/Inf
        mask = torch.nan_to_num(mask)
        # 计算 MAE 损失
        loss = torch.abs(prediction - target)
        loss = loss * mask
        loss = torch.nan_to_num(loss)
        # 累加损失和有效权重
        total_loss = loss.sum()
        total_weight = mask.sum()
    # 如果样本数 >64，分批计算
    else:
        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            # 同上面的掩码逻辑
            if np.isnan(null_val):
                mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_tensor = torch.full_like(target_batch, null_val, device=device)
                mask = ~torch.isclose(target_batch, null_tensor, atol=eps, rtol=0.0)

            mask = mask.float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)
            
            # 计算当前批次损失
            loss = torch.abs(pred_batch - target_batch)
            loss = loss * mask
            loss = torch.nan_to_num(loss)
            # 累加
            total_loss += loss.sum()
            total_weight += mask.sum()

            # 释放显存
            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()
     # 最终返回平均损失
    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_mape(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    """
    带掩码的平均绝对百分比误差（MAPE）
    自动忽略：
    1. NaN / 指定无效值
    2. target=0 的情况（避免除以0）
    """
    device = prediction.device
    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    num_samples = prediction.size(0)
    batch_size_limit = 64

    if num_samples <= batch_size_limit:
        # 过滤掉 target=0，防止除以0
        zero_mask = ~torch.isclose(target, torch.tensor(0.0, device=device), atol=5e-5)
        # 过滤无效值
        if np.isnan(null_val):
            null_mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            null_mask = ~torch.isclose(
                target, torch.tensor(null_val, device=device), atol=eps, rtol=0.0
            )
        # 最终有效掩码
        mask = (zero_mask & null_mask).float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        mask = torch.nan_to_num(mask)
        
        # MAPE 公式
        loss = torch.abs((prediction - target) / target)
        loss = loss * mask
        loss = torch.nan_to_num(loss)

        total_loss = loss.sum()
        total_weight = mask.sum()
    else:
        # 分批计算
        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            zero_mask = ~torch.isclose(
                target_batch, torch.tensor(0.0, device=device), atol=5e-5
            )

            if np.isnan(null_val):
                null_mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_mask = ~torch.isclose(
                    target_batch,
                    torch.tensor(null_val, device=device),
                    atol=eps,
                    rtol=0.0,
                )

            mask = (zero_mask & null_mask).float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)

            loss = torch.abs((pred_batch - target_batch) / target_batch)
            loss = loss * mask
            loss = torch.nan_to_num(loss)

            total_loss += loss.sum()
            total_weight += mask.sum()

            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_mse(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan
) -> torch.Tensor:
    """
    带掩码的均方误差（MSE）
    忽略无效值，大张量自动分批
    """
    device = prediction.device
    total_loss = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    num_samples = prediction.size(0)

    batch_size_limit = 64

    if num_samples <= batch_size_limit:
        if np.isnan(null_val):
            mask = ~torch.isnan(target)
        else:
            eps = 5e-5
            null_tensor = torch.full_like(target, null_val, device=device)
            mask = ~torch.isclose(target, null_tensor, atol=eps, rtol=0.0)

        mask = mask.float()
        mask_mean = torch.mean(mask)
        if mask_mean > 0:
            mask = mask / mask_mean
        mask = torch.nan_to_num(mask)
        # MSE 损失
        loss = (prediction - target) ** 2
        loss = loss * mask
        loss = torch.nan_to_num(loss)

        total_loss += loss.sum()
        total_weight += mask.sum()
    else:
        # 分批计算
        for i in range(0, num_samples, batch_size_limit):
            pred_batch = prediction[i : i + batch_size_limit]
            target_batch = target[i : i + batch_size_limit]

            if np.isnan(null_val):
                mask = ~torch.isnan(target_batch)
            else:
                eps = 5e-5
                null_tensor = torch.full_like(target_batch, null_val, device=device)
                mask = ~torch.isclose(target_batch, null_tensor, atol=eps, rtol=0.0)

            mask = mask.float()
            mask_mean = torch.mean(mask)
            if mask_mean > 0:
                mask = mask / mask_mean
            mask = torch.nan_to_num(mask)

            loss = (pred_batch - target_batch) ** 2
            loss = loss * mask
            loss = torch.nan_to_num(loss)
            total_loss += loss.sum()
            total_weight += mask.sum()

            del pred_batch, target_batch, mask, loss
            torch.cuda.empty_cache()

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)

def masked_rmse(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    带掩码的均方根误差（RMSE）
    直接调用 masked_mse 后开根号
    """
    return torch.sqrt(masked_mse(prediction=prediction, target=target, null_val=null_val))
