"""
GMAN 高速公路交通流量预测 - 4特征版本
支持: 小客车上行、小客车下行、非小客车上行、非小客车下行
时序配置: 8输入8输出
"""

import os
# TensorFlow 1.x 兼容性配置
os.environ['TF_XLA_FLAGS'] = ''
os.environ['TF_MLIR_OPTIMIZE_CONCRETE_FUNCTIONS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU避免GPU兼容问题

import math
import argparse
import numpy as np
import time
import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ==================== 配置参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=8, help='历史步长')
parser.add_argument('--Q', type=int, default=8, help='预测步长')
parser.add_argument('--L', type=int, default=5, help='STAtt Block数量')
parser.add_argument('--K', type=int, default=8, help='注意力头数')
parser.add_argument('--d', type=int, default=8, help='每个注意力头的维度')
parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--max_epoch', type=int, default=100, help='最大epoch数')
parser.add_argument('--patience', type=int, default=10, help='早停耐心度')
parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率')
parser.add_argument('--decay_epoch', type=int, default=5, help='学习率衰减周期')

# 路径配置
DATA_DIR = '/home/user/Downloads/cai/GMAN/data/highway_4feat'
MODEL_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
LOG_FILE = '/home/user/Downloads/cai/GMAN/logs/highway_4feat.log'

args = parser.parse_args()

# 创建目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_string(log, string):
    """日志记录"""
    log.write(string + '\n')
    log.flush()
    print(string)


def metric(pred, label):
    """评估指标计算 (MAE, RMSE, MAPE)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, np.abs(label) + 1e-8)  # 防止除零
        
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    
    return mae, rmse, mape


def load_data():
    """加载预处理好的数据"""
    print("📊 加载数据...")
    data = np.load(os.path.join(DATA_DIR, 'highway_data.npz'))
    
    trainX = data['trainX']
    trainTE = data['trainTE']
    trainY = data['trainY']
    valX = data['valX']
    valTE = data['valTE']
    valY = data['valY']
    testX = data['testX']
    testTE = data['testTE']
    testY = data['testY']
    SE = data['SE']
    # 修复NumPy弃用警告：正确提取标量值
    mean = float(data['mean'].item())
    std = float(data['std'].item())
    
    print(f"   训练集: X={trainX.shape}, Y={trainY.shape}")
    print(f"   验证集: X={valX.shape}, Y={valY.shape}")
    print(f"   测试集: X={testX.shape}, Y={testY.shape}")
    print(f"   空间嵌入: SE={SE.shape}")
    print(f"   归一化: mean={mean:.4f}, std={std:.4f}")
    
    return (trainX, trainTE, trainY, valX, valTE, valY, 
            testX, testTE, testY, SE, mean, std)


# ==================== GMAN模型定义 ====================
def placeholder(P, Q, N):
    """占位符定义 - 支持展平的多变量"""
    X = tf.placeholder(shape=(None, P, N), dtype=tf.float32, name='X')
    TE = tf.placeholder(shape=(None, P + Q, 2), dtype=tf.int32, name='TE')
    label = tf.placeholder(shape=(None, Q, N), dtype=tf.float32, name='label')
    is_training = tf.placeholder(shape=(), dtype=tf.bool, name='is_training')
    return X, TE, label, is_training


def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None, scope_name='fc'):
    """
    全连接层（完全手动实现，兼容TF 1.x和2.x）
    使用 tf.Variable + tf.nn.conv2d 手动构建
    
    Args:
        scope_name: 变量作用域名称，确保每次调用有独立的作用域
    """
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    
    assert type(units) == list
    
    for layer_idx, (num_unit, activation) in enumerate(zip(units, activations)):
        if drop is not None and is_training:
            x = tf.nn.dropout(x, keep_prob=1-drop)
        
        # 获取输入维度
        input_shape = x.get_shape().as_list()
        in_channels = input_shape[-1]
        
        # 创建唯一的变量作用域
        with tf.variable_scope(f'{scope_name}_layer_{layer_idx}', reuse=tf.AUTO_REUSE):
            # 创建权重 (1x1卷积核)
            kernel = tf.get_variable(
                name='kernel',
                shape=[1, 1, in_channels, num_unit],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer()
            )
            
            if use_bias:
                bias = tf.get_variable(
                    name='bias',
                    shape=[num_unit],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
            
            # 执行1x1卷积
            x = tf.nn.conv2d(
                input=x,
                filters=kernel,
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            
            if use_bias:
                x = tf.nn.bias_add(x, bias)
            
            # 应用激活函数
            if activation is not None:
                x = activation(x)
            
            # Batch Normalization（手动实现）
            if bn and activation is not None:
                # 获取通道数
                channels = x.get_shape().as_list()[-1]
                
                # 创建BN参数
                beta = tf.get_variable(
                    name=f'bn_beta',
                    shape=[channels],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                gamma = tf.get_variable(
                    name=f'bn_gamma',
                    shape=[channels],
                    dtype=tf.float32,
                    initializer=tf.ones_initializer()
                )
                
                # 计算均值和方差
                mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
                
                # 应用Batch Normalization
                x = tf.nn.batch_normalization(
                    x=x,
                    mean=mean,
                    variance=variance,
                    offset=beta,
                    scale=gamma,
                    variance_epsilon=1e-5
                )
    
    return x


def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    """
    时空嵌入
    SE: [N*F, D] 或 [N, D] - 需要适配输入维度
    TE: [batch_size, P + Q, 2] (dayofweek, timeofday)
    T: num of time steps in one day
    D: output dims
    return: [batch_size, P + Q, N*F, D]
    """
    # 获取SE的实际维度
    se_shape = SE.get_shape().as_list()
    
    # 空间嵌入 - 直接使用传入的SE（已经是展平后的N*F维度）
    SE_expanded = tf.expand_dims(tf.expand_dims(SE, axis=0), axis=0)
    SE_processed = FC(SE_expanded, units=[D, D], activations=[tf.nn.relu, None],
                      bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='SE')
    
    # 时间嵌入
    dayofweek = tf.one_hot(TE[..., 0], depth=7)
    timeofday = tf.one_hot(TE[..., 1], depth=T)
    TE_concat = tf.concat((dayofweek, timeofday), axis=-1)
    TE_expanded = tf.expand_dims(TE_concat, axis=2)
    TE_processed = FC(TE_expanded, units=[D, D], activations=[tf.nn.relu, None],
                      bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='TE')
    
    # 广播相加：SE_processed形状[1,1,N*F,D] + TE_processed形状[B,P+Q,1,D]
    # 结果形状：[B, P+Q, N*F, D]
    return tf.add(SE_processed, TE_processed)


def spatialAttention(X, STE, K, d, bn, bn_decay, is_training, scope_suffix=''):
    """空间注意力机制"""
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    
    query = FC(X, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'spatial_query{scope_suffix}')
    key = FC(X, units=D, activations=tf.nn.relu,
             bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'spatial_key{scope_suffix}')
    value = FC(X, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'spatial_value{scope_suffix}')
    
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    
    attention = tf.matmul(query, key, transpose_b=True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, -1)
    
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'spatial_output{scope_suffix}')
    
    return X


def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True, scope_suffix=''):
    """时间注意力机制"""
    D = K * d
    X = tf.concat((X, STE), axis=-1)
    
    query = FC(X, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'temporal_query{scope_suffix}')
    key = FC(X, units=D, activations=tf.nn.relu,
             bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'temporal_key{scope_suffix}')
    value = FC(X, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'temporal_value{scope_suffix}')
    
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape().as_list()[1]
        N = X.get_shape().as_list()[2]
        
        # 创建下三角mask矩阵
        mask_tensor = tf.ones(shape=(num_step, num_step), dtype=tf.float32)
        mask_tensor = tf.linalg.LinearOperatorLowerTriangular(mask_tensor).to_dense()
        
        # 扩展维度以匹配attention的形状 [K*batch, N, num_step, num_step]
        mask_tensor = tf.expand_dims(tf.expand_dims(mask_tensor, axis=0), axis=0)
        mask_tensor = tf.tile(mask_tensor, multiples=[K * batch_size, N, 1, 1])
        
        # 使用-1e9作为负无穷大值（与attention形状一致）
        neg_inf = tf.fill(tf.shape(attention), -1e9)
        attention = tf.where(mask_tensor > 0, attention, neg_inf)
    
    attention = tf.nn.softmax(attention, axis=-1)
    
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'temporal_output{scope_suffix}')
    
    return X


def gatedFusion(HS, HT, D, bn, bn_decay, is_training, scope_suffix=''):
    """门控融合"""
    XS = FC(HS, units=D, activations=None,
            bn=bn, bn_decay=bn_decay, is_training=is_training, use_bias=False, scope_name=f'fusion_xs{scope_suffix}')
    XT = FC(HT, units=D, activations=None,
            bn=bn, bn_decay=bn_decay, is_training=is_training, use_bias=True, scope_name=f'fusion_xt{scope_suffix}')
    
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(H, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name=f'fusion_output{scope_suffix}')
    
    return H


def STAttBlock(X, STE, K, d, bn, bn_decay, is_training, mask=True, layer_idx=0):
    """时空注意力块"""
    scope_suffix = f'_L{layer_idx}'
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training, scope_suffix=scope_suffix)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=mask, scope_suffix=scope_suffix)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training, scope_suffix=scope_suffix)
    return tf.add(X, H)


def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    """转换注意力机制"""
    D = K * d
    
    query = FC(STE_Q, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='trans_query')
    key = FC(STE_P, units=D, activations=tf.nn.relu,
             bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='trans_key')
    value = FC(X, units=D, activations=tf.nn.relu,
               bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='trans_value')
    
    query = tf.concat(tf.split(query, K, axis=-1), axis=0)
    key = tf.concat(tf.split(key, K, axis=-1), axis=0)
    value = tf.concat(tf.split(value, K, axis=-1), axis=0)
    
    query = tf.transpose(query, perm=(0, 2, 1, 3))
    key = tf.transpose(key, perm=(0, 2, 3, 1))
    value = tf.transpose(value, perm=(0, 2, 1, 3))
    
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis=-1)
    
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis=0), axis=-1)
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='trans_output')
    
    return X


def GMAN(X, TE, SE, P, Q, T, L, K, d, bn, bn_decay, is_training):
    """
    GMAN主模型
    X: [batch_size, P, N*F] (展平的多变量，N*F=98*4=392)
    TE: [batch_size, P+Q, 2]
    SE: [N*F, D] 或 [N, D] numpy数组 (需要与X的第二维匹配，即392)
    返回: [batch_size, Q, N*F]
    """
    D = K * d
    
    # 将SE从numpy数组转换为TensorFlow常量
    if isinstance(SE, np.ndarray):
        SE = tf.constant(SE, dtype=tf.float32)
    
    # 检查并调整SE维度以匹配输入
    se_shape = SE.get_shape().as_list()
    input_shape = X.get_shape().as_list()
    
    if len(se_shape) == 2 and se_shape[0] != input_shape[2]:
        # SE是[N, D]格式，需要扩展为[N*F, D]
        # 假设F=4，重复每个站点的嵌入4次
        N = se_shape[0]
        F = input_shape[2] // N
        if F > 1 and input_shape[2] % N == 0:
            # 重复每个站点的嵌入F次
            SE_repeated = tf.repeat(SE, repeats=F, axis=0)
            SE = SE_repeated
            print(f"   ⚠️  SE维度从 {se_shape} 调整为 {SE.get_shape().as_list()}")
        else:
            raise ValueError(f"SE维度 {se_shape[0]} 无法匹配输入维度 {input_shape[2]}")
    
    # 输入投影
    X = tf.expand_dims(X, axis=-1)  # [B, P, N*F, 1]
    X = FC(X, units=[D, D], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training, scope_name='input_proj')
    
    # 时空嵌入
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, :P]
    STE_Q = STE[:, P:]
    
    # 编码器
    for i in range(L):
        X = STAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    
    # 转换注意力
    X = transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training)
    
    # 解码器
    for i in range(L):
        X = STAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    
    # 输出层
    X = FC(X, units=[D, 1], activations=[tf.nn.relu, None],
           bn=bn, bn_decay=bn_decay, is_training=is_training,
           use_bias=True, drop=0.1, scope_name='output_proj')
    
    return tf.squeeze(X, axis=3)


def mae_loss(pred, label):
    """MAE损失函数"""
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(condition=tf.math.is_nan(mask), x=0., y=mask)
    
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.where(condition=tf.math.is_nan(loss), x=0., y=loss)
    loss = tf.reduce_mean(loss)
    
    return loss


# ==================== 主训练流程 ====================
def main():
    start = time.time()
    
    log = open(LOG_FILE, 'w')
    log_string(log, str(args))
    
    # 加载数据
    log_string(log, '\n📊 加载数据...')
    (trainX, trainTE, trainY, valX, valTE, valY, 
     testX, testTE, testY, SE, mean, std) = load_data()
    
    num_train, P, N = trainX.shape
    T = 24  # 一天24小时
    
    # 构建模型
    log_string(log, '\n🔧 编译模型...')
    X, TE, label, is_training = placeholder(P, args.Q, N)
    
    global_step = tf.Variable(0, trainable=False)
    bn_momentum = tf.train.exponential_decay(
        0.5, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.5, staircase=True
    )
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    
    pred = GMAN(X, TE, SE, P, args.Q, T, args.L, args.K, args.d,
                bn=True, bn_decay=bn_decay, is_training=is_training)
    
    # 反归一化
    pred = pred * std + mean
    label_original = label * std + mean
    
    loss = mae_loss(pred, label_original)
    
    # 优化器
    learning_rate = tf.train.exponential_decay(
        args.learning_rate, global_step,
        decay_steps=args.decay_epoch * num_train // args.batch_size,
        decay_rate=0.7, staircase=True
    )
    learning_rate = tf.maximum(learning_rate, 1e-5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    # 统计参数量
    parameters = 0
    for variable in tf.trainable_variables():
        parameters += np.prod([x.value for x in variable.get_shape()])
    log_string(log, f'可训练参数: {parameters:,}')
    
    # 初始化会话
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # 训练循环
    log_string(log, '\n**** 开始训练 ****')
    num_val = valX.shape[0]
    wait = 0
    val_loss_min = np.inf
    
    for epoch in range(args.max_epoch):
        if wait >= args.patience:
            log_string(log, f'早停于 epoch: {epoch:04d}')
            break
        
        # Shuffle训练数据
        permutation = np.random.permutation(num_train)
        trainX_shuffled = trainX[permutation]
        trainTE_shuffled = trainTE[permutation]
        trainY_shuffled = trainY[permutation]
        
        # 训练
        start_train = time.time()
        train_loss = 0
        num_batch = math.ceil(num_train / args.batch_size)
        
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
            
            feed_dict = {
                X: trainX_shuffled[start_idx:end_idx],
                TE: trainTE_shuffled[start_idx:end_idx],
                label: trainY_shuffled[start_idx:end_idx],
                is_training: True
            }
            
            _, loss_batch = sess.run([train_op, loss], feed_dict=feed_dict)
            train_loss += loss_batch * (end_idx - start_idx)
        
        train_loss /= num_train
        end_train = time.time()
        
        # 验证
        start_val = time.time()
        val_loss = 0
        num_batch = math.ceil(num_val / args.batch_size)
        
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
            
            feed_dict = {
                X: valX[start_idx:end_idx],
                TE: valTE[start_idx:end_idx],
                label: valY[start_idx:end_idx],
                is_training: False
            }
            
            loss_batch = sess.run(loss, feed_dict=feed_dict)
            val_loss += loss_batch * (end_idx - start_idx)
        
        val_loss /= num_val
        end_val = time.time()
        
        log_string(log, 
                   f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                   f'epoch: {epoch+1}/{args.max_epoch}, '
                   f'train time: {end_train-start_train:.1f}s, '
                   f'val time: {end_val-start_val:.1f}s')
        log_string(log, f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss <= val_loss_min:
            log_string(log, 
                      f'val loss 从 {val_loss_min:.4f} 降到 {val_loss:.4f}, '
                      f'保存模型到 {MODEL_DIR}')
            wait = 0
            val_loss_min = val_loss
            saver.save(sess, MODEL_DIR)
        else:
            wait += 1
    
    # 测试
    log_string(log, '\n**** 测试模型 ****')
    log_string(log, f'从 {MODEL_DIR} 加载模型')
    saver = tf.train.import_meta_graph(MODEL_DIR + '.meta')
    saver.restore(sess, MODEL_DIR)
    log_string(log, '模型已恢复!')
    
    num_test = testX.shape[0]
    
    # 测试集预测
    testPred = []
    start_test = time.time()
    num_batch = math.ceil(num_test / args.batch_size)
    
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
        
        feed_dict = {
            X: testX[start_idx:end_idx],
            TE: testTE[start_idx:end_idx],
            is_training: False
        }
        
        pred_batch = sess.run(pred, feed_dict=feed_dict)
        testPred.append(pred_batch)
    
    end_test = time.time()
    testPred = np.concatenate(testPred, axis=0)
    
    # 计算指标
    test_mae, test_rmse, test_mape = metric(testPred, testY * std + mean)
    
    log_string(log, f'测试时间: {end_test-start_test:.1f}s')
    log_string(log, '                MAE\t\tRMSE\t\tMAPE')
    log_string(log, f'test             {test_mae:.2f}\t\t{test_rmse:.2f}\t\t{test_mape*100:.2f}%')
    
    # 每个预测步的指标
    log_string(log, '\n各预测步性能:')
    MAE, RMSE, MAPE = [], [], []
    
    for q in range(args.Q):
        mae, rmse, mape = metric(testPred[:, q], testY[:, q] * std + mean)
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, f'step: {q+1:02d}         {mae:.2f}\t\t{rmse:.2f}\t\t{mape*100:.2f}%')
    
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    
    log_string(log, f'\naverage:         {average_mae:.2f}\t\t{average_rmse:.2f}\t\t{average_mape*100:.2f}%')
    
    end = time.time()
    log_string(log, f'\n总耗时: {(end-start)/60:.1f}min')
    
    sess.close()
    log.close()
    
    print("\n✅ 训练完成!")
    print(f"   模型保存: {MODEL_DIR}")
    print(f"   日志文件: {LOG_FILE}")


if __name__ == '__main__':
    main()
