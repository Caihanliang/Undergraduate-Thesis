"""
GMAN 高速公路交通流量预测 - 4特征版本 (TF 2.x 原生实现)
支持: 小客车上行、小客车下行、非小客车上行、非小客车下行
时序配置: 8输入8输出
"""

import os
# 启用GPU（移除之前的禁用配置）
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 注释掉这行以使用GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import argparse
import numpy as np
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks

# ==================== 配置参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=8, help='历史步长')
parser.add_argument('--Q', type=int, default=8, help='预测步长')
parser.add_argument('--L', type=int, default=3, help='STAtt Block数量（增加到3）')
parser.add_argument('--K', type=int, default=8, help='注意力头数（增加到8）')
parser.add_argument('--d', type=int, default=16, help='每个注意力头的维度（增加到16）')
parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--max_epoch', type=int, default=50, help='最大epoch数（增加到50）')
parser.add_argument('--patience', type=int, default=10, help='早停耐心度')
parser.add_argument('--learning_rate', type=float, default=0.002, help='初始学习率（提高到0.002）')
parser.add_argument('--decay_epoch', type=int, default=10, help='学习率衰减周期')
parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')

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


# ==================== 数据加载 ====================
def load_data():
    """加载预处理的数据"""
    print("📊 加载数据...")
    
    # 加载所有数据（单个npz文件）
    data = np.load(os.path.join(DATA_DIR, 'highway_data.npz'))
    
    X_train = data['trainX']
    Y_train = data['trainY']
    trainTE = data['trainTE']
    
    X_val = data['valX']
    Y_val = data['valY']
    valTE = data['valTE']
    
    X_test = data['testX']
    Y_test = data['testY']
    testTE = data['testTE']
    
    SE = data['SE']
    mean = float(data['mean'][0]) if data['mean'].ndim > 0 else float(data['mean'])
    std = float(data['std'][0]) if data['std'].ndim > 0 else float(data['std'])
    
    print(f"   训练集: X={X_train.shape}, Y={Y_train.shape}, TE={trainTE.shape}")
    print(f"   验证集: X={X_val.shape}, Y={Y_val.shape}, TE={valTE.shape}")
    print(f"   测试集: X={X_test.shape}, Y={Y_test.shape}, TE={testTE.shape}")
    print(f"   空间嵌入: SE={SE.shape}")
    print(f"   归一化: mean={mean:.4f}, std={std:.4f}")
    
    return (X_train, Y_train, trainTE, 
            X_val, Y_val, valTE, 
            X_test, Y_test, testTE,
            SE, mean, std)


# ==================== GMAN模型组件 (TF 2.x 原生实现) ====================
class FC(layers.Layer):
    """全连接层（1x1卷积实现）"""
    def __init__(self, units, activation=None, use_bias=True, bn=False, name='fc'):
        super(FC, self).__init__(name=name)
        self.units = units if isinstance(units, list) else [units]
        self.activation = activation
        self.use_bias = use_bias
        self.bn = bn
        
        # 为每一层创建独立的Conv2D
        self.conv_layers = []
        self.bn_layers = []
        
        for i, units in enumerate(self.units):
            self.conv_layers.append(
                layers.Conv2D(units, kernel_size=(1, 1), use_bias=use_bias, 
                             name=f'{name}_conv_{i}')
            )
            if bn and activation is not None:
                self.bn_layers.append(
                    layers.BatchNormalization(name=f'{name}_bn_{i}')
                )
    
    def call(self, x, training=False):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            
            # Batch Normalization
            if self.bn and self.activation is not None and i < len(self.bn_layers):
                x = self.bn_layers[i](x, training=training)
            
            # Activation
            if self.activation is not None:
                x = self.activation(x)
        
        return x


class STEmbedding(layers.Layer):
    """时空嵌入层"""
    def __init__(self, D, T, bn=False, name='st_embedding'):
        super(STEmbedding, self).__init__(name=name)
        self.D = D
        self.T = T
        self.bn = bn
        
        self.se_fc = FC([D, D], activation=tf.nn.relu, bn=bn, name='SE')
        self.te_fc = FC([D, D], activation=tf.nn.relu, bn=bn, name='TE')
    
    def call(self, se, te, training=False):
        """
        se: [N*F, D] 空间嵌入
        te: [B, P+Q, 2] 时间特征 (dayofweek, timeofday) - 可能是float或int
        返回: [B, P+Q, N*F, D]
        """
        # 空间嵌入处理
        se_expanded = tf.expand_dims(tf.expand_dims(se, axis=0), axis=0)  # [1, 1, N*F, D]
        se_processed = self.se_fc(se_expanded, training=training)  # [1, 1, N*F, D]
        
        # 时间嵌入处理 - 确保te是整数类型
        te_int = tf.cast(te, tf.int32)  # 转换为int32
        
        dayofweek = tf.one_hot(te_int[..., 0], depth=7)  # [B, P+Q, 7]
        timeofday = tf.one_hot(te_int[..., 1], depth=self.T)  # [B, P+Q, T]
        te_concat = tf.concat([dayofweek, timeofday], axis=-1)  # [B, P+Q, 7+T]
        te_expanded = tf.expand_dims(te_concat, axis=2)  # [B, P+Q, 1, 7+T]
        te_processed = self.te_fc(te_expanded, training=training)  # [B, P+Q, 1, D]
        
        # 广播相加
        return se_processed + te_processed  # [B, P+Q, N*F, D]


class SpatialAttention(layers.Layer):
    """空间注意力机制"""
    def __init__(self, K, d, bn=False, name='spatial_attention'):
        super(SpatialAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K * d
        self.bn = bn
        
        self.query_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='query')
        self.key_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='key')
        self.value_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='value')
        self.output_fc = FC([self.D, self.D], activation=tf.nn.relu, bn=bn, name='output')
    
    def call(self, x, ste, training=False):
        """
        x: [B, P, N*F, D]
        ste: [B, P, N*F, D]
        返回: [B, P, N*F, D]
        """
        x_concat = tf.concat([x, ste], axis=-1)  # [B, P, N*F, 2D]
        
        query = self.query_fc(x_concat, training=training)  # [B, P, N*F, D]
        key = self.key_fc(x_concat, training=training)  # [B, P, N*F, D]
        value = self.value_fc(x_concat, training=training)  # [B, P, N*F, D]
        
        # 多头分割
        query_heads = tf.split(query, self.K, axis=-1)  # K * [B, P, N*F, d]
        key_heads = tf.split(key, self.K, axis=-1)
        value_heads = tf.split(value, self.K, axis=-1)
        
        # 合并batch和head维度
        query_merged = tf.concat(query_heads, axis=0)  # [K*B, P, N*F, d]
        key_merged = tf.concat(key_heads, axis=0)
        value_merged = tf.concat(value_heads, axis=0)
        
        # 计算attention
        attention = tf.matmul(query_merged, key_merged, transpose_b=True)  # [K*B, P, N*F, N*F]
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)
        
        # 应用attention
        out = tf.matmul(attention, value_merged)  # [K*B, P, N*F, d]
        out_heads = tf.split(out, self.K, axis=0)  # K * [B, P, N*F, d]
        out = tf.concat(out_heads, axis=-1)  # [B, P, N*F, D]
        
        return self.output_fc(out, training=training)


class TemporalAttention(layers.Layer):
    """时间注意力机制"""
    def __init__(self, K, d, bn=False, mask=True, name='temporal_attention'):
        super(TemporalAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K * d
        self.bn = bn
        self.mask = mask
        
        self.query_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='query')
        self.key_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='key')
        self.value_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='value')
        self.output_fc = FC([self.D, self.D], activation=tf.nn.relu, bn=bn, name='output')
    
    def call(self, x, ste, training=False):
        """
        x: [B, P, N*F, D]
        ste: [B, P, N*F, D]
        返回: [B, P, N*F, D]
        """
        x_concat = tf.concat([x, ste], axis=-1)
        
        query = self.query_fc(x_concat, training=training)
        key = self.key_fc(x_concat, training=training)
        value = self.value_fc(x_concat, training=training)
        
        # 多头分割
        query_heads = tf.split(query, self.K, axis=-1)
        key_heads = tf.split(key, self.K, axis=-1)
        value_heads = tf.split(value, self.K, axis=-1)
        
        # 合并batch和head维度
        query_merged = tf.concat(query_heads, axis=0)  # [K*B, P, N*F, d]
        key_merged = tf.concat(key_heads, axis=0)
        value_merged = tf.concat(value_heads, axis=0)
        
        # 转置以进行时间维度attention
        query_t = tf.transpose(query_merged, perm=[0, 2, 1, 3])  # [K*B, N*F, P, d]
        key_t = tf.transpose(key_merged, perm=[0, 2, 3, 1])  # [K*B, N*F, d, P]
        value_t = tf.transpose(value_merged, perm=[0, 2, 1, 3])  # [K*B, N*F, P, d]
        
        # 计算attention
        attention = tf.matmul(query_t, key_t)  # [K*B, N*F, P, P]
        attention /= (self.d ** 0.5)
        
        # Causal mask
        if self.mask:
            batch_size = tf.shape(x)[0]
            num_step = tf.shape(x)[1]
            n_feat = tf.shape(x)[2]
            
            # 创建下三角mask
            mask_matrix = tf.linalg.band_part(
                tf.ones([num_step, num_step]), 
                num_lower=-1, 
                num_upper=0
            )  # 下三角矩阵
            
            # 扩展维度并重复
            mask_expanded = tf.expand_dims(tf.expand_dims(mask_matrix, 0), 0)  # [1, 1, P, P]
            mask_tiled = tf.tile(mask_expanded, [self.K * batch_size, n_feat, 1, 1])
            
            # 应用mask
            neg_inf = tf.fill(tf.shape(attention), -1e9)
            attention = tf.where(mask_tiled > 0, attention, neg_inf)
        
        attention = tf.nn.softmax(attention, axis=-1)
        
        # 应用attention
        out = tf.matmul(attention, value_t)  # [K*B, N*F, P, d]
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # [K*B, P, N*F, d]
        
        # 分割head并合并
        out_heads = tf.split(out, self.K, axis=0)
        out = tf.concat(out_heads, axis=-1)
        
        return self.output_fc(out, training=training)


class GatedFusion(layers.Layer):
    """门控融合"""
    def __init__(self, D, bn=False, name='gated_fusion'):
        super(GatedFusion, self).__init__(name=name)
        self.D = D
        self.bn = bn
        
        self.xs_fc = FC(D, activation=None, use_bias=False, bn=bn, name='xs')
        self.xt_fc = FC(D, activation=None, use_bias=True, bn=bn, name='xt')
        self.output_fc = FC([D, D], activation=tf.nn.relu, bn=bn, name='output')
    
    def call(self, hs, ht, training=False):
        """
        hs: [B, P, N*F, D] 空间特征
        ht: [B, P, N*F, D] 时间特征
        返回: [B, P, N*F, D]
        """
        xs = self.xs_fc(hs, training=training)
        xt = self.xt_fc(ht, training=training)
        
        z = tf.nn.sigmoid(xs + xt)
        h = z * hs + (1 - z) * ht
        
        return self.output_fc(h, training=training)


class TransformAttention(layers.Layer):
    """转换注意力（从encoder到decoder）"""
    def __init__(self, K, d, bn=False, name='transform_attention'):
        super(TransformAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K * d
        self.bn = bn
        
        self.query_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='query')
        self.key_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='key')
        self.value_fc = FC(self.D, activation=tf.nn.relu, bn=bn, name='value')
        self.output_fc = FC([self.D, self.D], activation=tf.nn.relu, bn=bn, name='output')
    
    def call(self, x, ste_p, ste_q, training=False):
        """
        x: [B, P, N*F, D] encoder输出
        ste_p: [B, P, N*F, D] encoder时空嵌入
        ste_q: [B, Q, N*F, D] decoder时空嵌入
        返回: [B, Q, N*F, D]
        """
        query = self.query_fc(ste_q, training=training)  # [B, Q, N*F, D]
        key = self.key_fc(ste_p, training=training)  # [B, P, N*F, D]
        value = self.value_fc(x, training=training)  # [B, P, N*F, D]
        
        # 多头分割
        query_heads = tf.split(query, self.K, axis=-1)
        key_heads = tf.split(key, self.K, axis=-1)
        value_heads = tf.split(value, self.K, axis=-1)
        
        # 合并batch和head
        query_merged = tf.concat(query_heads, axis=0)  # [K*B, Q, N*F, d]
        key_merged = tf.concat(key_heads, axis=0)  # [K*B, P, N*F, d]
        value_merged = tf.concat(value_heads, axis=0)  # [K*B, P, N*F, d]
        
        # 转置
        query_t = tf.transpose(query_merged, perm=[0, 2, 1, 3])  # [K*B, N*F, Q, d]
        key_t = tf.transpose(key_merged, perm=[0, 2, 3, 1])  # [K*B, N*F, d, P]
        value_t = tf.transpose(value_merged, perm=[0, 2, 1, 3])  # [K*B, N*F, P, d]
        
        # Attention
        attention = tf.matmul(query_t, key_t)  # [K*B, N*F, Q, P]
        attention /= (self.d ** 0.5)
        attention = tf.nn.softmax(attention, axis=-1)
        
        out = tf.matmul(attention, value_t)  # [K*B, N*F, Q, d]
        out = tf.transpose(out, perm=[0, 2, 1, 3])  # [K*B, Q, N*F, d]
        
        out_heads = tf.split(out, self.K, axis=0)
        out = tf.concat(out_heads, axis=-1)
        
        return self.output_fc(out, training=training)


class STAttBlock(layers.Layer):
    """时空注意力块"""
    def __init__(self, K, d, bn=False, name='st_att_block'):
        super(STAttBlock, self).__init__(name=name)
        self.spatial_att = SpatialAttention(K, d, bn=bn, name='spatial')
        self.temporal_att = TemporalAttention(K, d, bn=bn, name='temporal')
        self.gated_fusion = GatedFusion(K * d, bn=bn, name='fusion')
    
    def call(self, x, ste, training=False):
        hs = self.spatial_att(x, ste, training=training)
        ht = self.temporal_att(x, ste, training=training)
        return self.gated_fusion(hs, ht, training=training)


class GMANModel(tf.keras.Model):
    """GMAN主模型"""
    def __init__(self, P, Q, T, L, K, d, D_se, bn=False, name='GMAN'):
        super(GMANModel, self).__init__(name=name)
        self.P = P
        self.Q = Q
        self.L = L
        self.K = K
        self.d = d
        self.D = K * d
        
        # 输入投影
        self.input_proj = FC([self.D, self.D], activation=tf.nn.relu, bn=bn, name='input_proj')
        
        # 时空嵌入
        self.st_embedding = STEmbedding(self.D, T, bn=bn, name='st_embedding')
        
        # Encoder blocks
        self.encoder_blocks = [
            STAttBlock(K, d, bn=bn, name=f'encoder_block_{i}') 
            for i in range(L)
        ]
        
        # Transform attention
        self.transform_att = TransformAttention(K, d, bn=bn, name='transform_att')
        
        # Decoder blocks
        self.decoder_blocks = [
            STAttBlock(K, d, bn=bn, name=f'decoder_block_{i}') 
            for i in range(L)
        ]
        
        # 输出层 - 关键修复: 回归任务应该使用线性激活（无激活函数）
        # 之前使用ReLU导致模型退化为均值预测器（所有预测值都是265.11）
        self.output_proj = FC([self.D, 1], activation=None, bn=False, name='output_proj')

    def get_config(self):
        """实现序列化配置，支持model.save()"""
        base_config = super().get_config()
        config = {
            'P': self.P,
            'Q': self.Q,
            'T': self.st_embedding.T if hasattr(self.st_embedding, 'T') else 24,
            'L': self.L,
            'K': self.K,
            'd': self.d,
            'D_se': self.encoder_blocks[0].se_conv.units if hasattr(self.encoder_blocks[0], 'se_conv') else 64,
            'bn': True  # 默认使用BatchNorm
        }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        """从配置重建模型"""
        return cls(**config)
    
    def call(self, inputs, training=False):
        """
        inputs: dict with keys:
            - X: [B, P, N*F] 输入序列 (N*F = 98*4 = 392)
            - TE: [B, P+Q, 2] 时间特征
            - SE: [N, D_se] 空间嵌入 (N=98站点，需要扩展到N*F=392)
        返回: [B, Q, N*F] 预测结果 (392维，包含4个特征的展平)
        """
        X = inputs['X']
        TE = inputs['TE']
        SE = inputs['SE']
        
        # 调整SE维度以匹配输入
        # SE原始形状: [98, D_se]，需要扩展为 [392, D_se]（每个站点重复4次对应4个特征）
        se_shape = SE.shape.as_list() if hasattr(SE, 'shape') else list(SE.shape)
        input_n_feat = X.shape[2]  # N*F = 392
        
        if len(se_shape) == 2 and se_shape[0] != input_n_feat:
            # 计算每个站点对应的特征数
            n_stations = se_shape[0]  # 98
            features_per_station = input_n_feat // n_stations  # 392/98 = 4
            
            if features_per_station > 1:
                # 重复每个站点的嵌入F次
                SE_expanded = tf.repeat(SE, repeats=features_per_station, axis=0)
                SE = SE_expanded
        
        # 输入投影
        X = tf.expand_dims(X, axis=-1)  # [B, P, N*F, 1]
        X = self.input_proj(X, training=training)  # [B, P, N*F, D]
        
        # 时空嵌入
        STE = self.st_embedding(SE, TE, training=training)  # [B, P+Q, N*F, D]
        STE_P = STE[:, :self.P]  # [B, P, N*F, D]
        STE_Q = STE[:, self.P:]  # [B, Q, N*F, D]
        
        # Encoder
        for block in self.encoder_blocks:
            X = block(X, STE_P, training=training)
        
        # Transform attention
        X = self.transform_att(X, STE_P, STE_Q, training=training)
        
        # Decoder
        for block in self.decoder_blocks:
            X = block(X, STE_Q, training=training)
        
        # 输出
        X = self.output_proj(X, training=training)  # [B, Q, N*F, 1]
        return tf.squeeze(X, axis=-1)  # [B, Q, N*F] 保持展平格式


# ==================== 训练函数 ====================
def train_model():
    """训练GMAN模型"""
    print(args)
    
    # 加载数据
    (X_train, Y_train, trainTE, 
     X_val, Y_val, valTE, 
     X_test, Y_test, testTE,
     SE, mean, std) = load_data()
    
    # 转换为TensorFlow Dataset（不包含SE）
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'X': X_train.astype(np.float32), 'TE': trainTE.astype(np.float32)},
        Y_train.astype(np.float32)
    ))
    train_dataset = train_dataset.shuffle(1000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'X': X_val.astype(np.float32), 'TE': valTE.astype(np.float32)},
        Y_val.astype(np.float32)
    ))
    val_dataset = val_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'X': X_test.astype(np.float32), 'TE': testTE.astype(np.float32)},
        Y_test.astype(np.float32)
    ))
    test_dataset = test_dataset.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 构建模型
    print("\n🔧 编译模型...")
    model = GMANModel(
        P=args.P, Q=args.Q, T=24, L=args.L, K=args.K, d=args.d,
        D_se=SE.shape[1], bn=True
    )
    
    # 将SE转换为常量并设置为模型属性
    SE_tensor = tf.constant(SE, dtype=tf.float32)
    
    # 初始化模型（通过一次前向传播）
    sample_input = {
        'X': tf.constant(X_train[:1].astype(np.float32)),
        'TE': tf.constant(trainTE[:1].astype(np.float32)),
        'SE': SE_tensor
    }
    _ = model(sample_input, training=False)
    
    # 打印模型摘要
    model.build(input_shape=None)
    total_params = sum([tf.reduce_prod(v.shape).numpy() for v in model.trainable_variables])
    print(f"   可训练参数: {total_params:,}")
    
    # 编译模型
    optimizer = optimizers.Adam(learning_rate=args.learning_rate)
    
    # 定义损失函数（使用Huber Loss增强鲁棒性）
    def huber_loss(y_true, y_pred, delta=1.0):
        """Huber Loss - 对异常值更鲁棒"""
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * tf.square(quadratic) + delta * linear
        
        # 应用mask忽略0值
        mask = tf.not_equal(y_true, 0)
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    
    def masked_mae(y_true, y_pred):
        mask = tf.not_equal(y_true, 0)
        mask = tf.cast(mask, tf.float32)
        loss = tf.abs(y_true - y_pred) * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    # 自定义训练步骤（添加梯度裁剪）
    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as tape:
            # 添加SE到输入
            batch_x_with_se = {**batch_x, 'SE': SE_tensor}
            predictions = model(batch_x_with_se, training=True)
            loss = huber_loss(batch_y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        # 梯度裁剪防止不稳定
        gradients, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss
    
    @tf.function
    def val_step(batch_x, batch_y):
        batch_x_with_se = {**batch_x, 'SE': SE_tensor}
        predictions = model(batch_x_with_se, training=False)
        loss = masked_mae(batch_y, predictions)
        return loss, predictions
    
    # 训练循环
    print("\n🚀 开始训练...\n")
    log = open(LOG_FILE, 'w')
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.max_epoch):
        start_time = time.time()
        
        # 训练阶段
        train_losses = []
        for batch_x, batch_y in train_dataset:
            loss = train_step(batch_x, batch_y)
            train_losses.append(loss.numpy())
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证阶段
        val_losses = []
        for batch_x, batch_y in val_dataset:
            loss, _ = val_step(batch_x, batch_y)
            val_losses.append(loss.numpy())
        
        avg_val_loss = np.mean(val_losses)
        
        epoch_time = time.time() - start_time
        
        log_msg = f"Epoch {epoch+1:3d}/{args.max_epoch} | " \
                  f"Train Loss: {avg_train_loss:.4f} | " \
                  f"Val Loss: {avg_val_loss:.4f} | " \
                  f"Time: {epoch_time:.1f}s"
        
        log_string(log, log_msg)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型（使用TensorFlow checkpoint格式）
            try:
                checkpoint_path = os.path.join(MODEL_DIR, 'best_model_tf')
                checkpoint = tf.train.Checkpoint(model=model)
                checkpoint.save(checkpoint_path)
                log_string(log, f"   ✅ 保存TF Checkpoint (val_loss: {best_val_loss:.4f})")
            except Exception as e:
                log_string(log, f"   ⚠️  TF Checkpoint保存失败: {e}")
            
            # 同时保存NumPy格式作为备份
            try:
                weights_dict = {}
                for var in model.variables:
                    weights_dict[var.name.replace('/', '_').replace(':', '_')] = var.numpy()
                
                np.savez(os.path.join(MODEL_DIR, 'best_model_weights.npz'), **weights_dict)
                log_string(log, f"   ✅ 保存NumPy权重备份")
            except Exception as e:
                log_string(log, f"   ⚠️  NumPy权重保存失败: {e}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log_string(log, f"\n⏹️  早停触发！在epoch {epoch+1}停止训练")
                break
        
        # 学习率衰减
        if (epoch + 1) % args.decay_epoch == 0:
            new_lr = args.learning_rate * (0.5 ** ((epoch + 1) // args.decay_epoch))
            optimizer.learning_rate.assign(new_lr)
            log_string(log, f"   📉 学习率调整为: {new_lr:.6f}")
    
    log.close()
    
    # 测试阶段
    print("\n📊 测试阶段...")
    
    # 优先加载NumPy权重
    np_weights_path = os.path.join(MODEL_DIR, 'best_model_weights.npz')
    h5_weights_path = os.path.join(MODEL_DIR, 'best_model.weights.h5')
    
    if os.path.exists(np_weights_path):
        try:
            np_weights = np.load(np_weights_path, allow_pickle=True)
            # 按顺序将权重赋值给模型层
            weight_idx = 0
            for layer in model.layers:
                if layer.weights:
                    num_weights = len(layer.weights)
                    layer_weights = []
                    for i in range(num_weights):
                        key = f'{layer.name}_{i}'
                        if key in np_weights:
                            layer_weights.append(np_weights[key])
                        else:
                            # 如果找不到对应的key，保持原权重或报错，这里选择跳过该层更新
                            print(f"Warning: Key {key} not found in npz file")
                            break
                    
                    if len(layer_weights) == num_weights:
                        layer.set_weights(layer_weights)
                        weight_idx += num_weights
            
            log_string(open(LOG_FILE, 'a'), "   ✅ 已从NumPy文件加载权重")
            print("   ✅ 已从NumPy文件加载权重")
        except Exception as e:
            print(f"   ⚠️  NumPy权重加载失败: {e}，尝试加载H5...")
            if os.path.exists(h5_weights_path):
                model.load_weights(h5_weights_path)
    elif os.path.exists(h5_weights_path):
        model.load_weights(h5_weights_path)
    else:
        print("   ⚠️  未找到任何预训练权重文件，将使用随机初始化的模型进行测试")
    
    test_predictions = []
    test_actuals = []
    
    for batch_x, batch_y in test_dataset:
        batch_x_with_se = {**batch_x, 'SE': SE_tensor}
        pred = model(batch_x_with_se, training=False)
        test_predictions.append(pred.numpy())
        test_actuals.append(batch_y.numpy())
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_actuals = np.concatenate(test_actuals, axis=0)
    
    # 诊断：检查归一化空间的预测质量
    norm_mae = np.mean(np.abs(test_predictions - test_actuals))
    print(f"\n   📈 归一化空间 MAE: {norm_mae:.4f}")
    print(f"   📊 预测值范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    print(f"   📊 真实值范围: [{test_actuals.min():.2f}, {test_actuals.max():.2f}]")
    print(f"   📊 预测值均值: {test_predictions.mean():.2f}, 标准差: {test_predictions.std():.2f}")
    
    # 反归一化
    test_predictions_denorm = test_predictions * std + mean
    test_actuals_denorm = test_actuals * std + mean
    
    # 计算MAE
    mae = np.mean(np.abs(test_predictions_denorm - test_actuals_denorm))
    rmse = np.sqrt(np.mean((test_predictions_denorm - test_actuals_denorm) ** 2))
    mape = np.mean(np.abs((test_actuals_denorm - test_predictions_denorm) / (test_actuals_denorm + 1e-8))) * 100
    
    print(f"\n✅ 测试集指标:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # 保存预测结果
    np.savez(os.path.join(MODEL_DIR, 'predictions.npz'),
             predictions=test_predictions_denorm,
             actuals=test_actuals_denorm,
             predictions_norm=test_predictions,
             actuals_norm=test_actuals,
             mae=mae,
             rmse=rmse,
             mape=mape)
    
    print(f"\n💾 结果已保存到: {MODEL_DIR}")
    
    # ========== 新增: 保存所有数据集的预测结果到CSV ==========
    print("\n" + "="*60)
    print("📊 开始保存所有数据集的预测结果到CSV...")
    print("="*60)
    
    from datetime import datetime, timedelta
    import pandas as pd
    
    START_TIME = datetime(2023, 9, 1, 0, 0, 0)
    P, Q = args.P, args.Q
    N = 98
    F = 4
    
    def generate_timestamps(num_samples):
        """生成时间戳"""
        timestamps = []
        for i in range(num_samples):
            pred_start = START_TIME + timedelta(hours=i)
            sample_timestamps = [pred_start + timedelta(hours=j) for j in range(Q)]
            timestamps.append(sample_timestamps)
        return np.array(timestamps)
    
    def save_dataset_predictions(model, dataset_name, X, Y, TE, SE_tensor_expanded, mean, std):
        """保存单个数据集的预测结果"""
        print(f"\n🚀 处理 {dataset_name} 集...")
        
        # 创建Dataset
        ds = tf.data.Dataset.from_tensor_slices((
            {'X': X.astype(np.float32), 'TE': TE.astype(np.float32)},
            Y.astype(np.float32)
        ))
        ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        
        # 预测
        all_preds = []
        all_actuals = []
        for batch_x, batch_y in ds:
            batch_input = {
                'X': batch_x['X'],
                'TE': batch_x['TE'],
                'SE': SE_tensor_expanded  # 直接使用 [N*F, D_se]，不需要batch维度
            }
            pred = model(batch_input, training=False)
            all_preds.append(pred.numpy())
            all_actuals.append(batch_y.numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)
        
        # 反归一化
        predictions_denorm = predictions * std + mean
        actuals_denorm = actuals * std + mean
        
        # Reshape
        samples = predictions_denorm.shape[0]
        pred_reshaped = predictions_denorm.reshape(samples, Q, N, F)
        actual_reshaped = actuals_denorm.reshape(samples, Q, N, F)
        
        # 展平
        pred_flat = pred_reshaped.reshape(-1, F)
        actual_flat = actual_reshaped.reshape(-1, F)
        
        # 生成索引和时间戳
        timestamps = generate_timestamps(samples)
        sample_indices = np.repeat(np.arange(samples), Q * N)
        time_steps = np.tile(np.repeat(np.arange(Q), N), samples)
        station_indices = np.tile(np.arange(N), samples * Q)
        timestamps_flat = timestamps.reshape(-1)
        timestamps_expanded = np.repeat(timestamps_flat, N)
        timestamp_strs = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps_expanded]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'sample_index': sample_indices,
            'time_step': time_steps,
            'station_index': station_indices,
            'timestamp': timestamp_strs,
            '小客车上行_预测值': pred_flat[:, 0],
            '小客车上行_真实값': actual_flat[:, 0],
            '小客车下行_预测值': pred_flat[:, 1],
            '小客车下行_真实값': actual_flat[:, 1],
            '非小客车上行_预测값': pred_flat[:, 2],
            '非小客车上行_真实값': actual_flat[:, 2],
            '非小客车下行_预测값': pred_flat[:, 3],
            '非小客车下行_真实값': actual_flat[:, 3]
        })
        
        # 计算MAE
        feature_names = ['小客车上行', '小客车下行', '非小客车上行', '非小客车下行']
        mae_dict = {}
        for i, feat in enumerate(feature_names):
            mae = np.mean(np.abs(pred_flat[:, i] - actual_flat[:, i]))
            mae_dict[feat] = mae
            print(f"   {feat} MAE: {mae:.2f}")
        
        # 保存CSV
        csv_path = os.path.join(MODEL_DIR, f'{dataset_name}_predictions.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 已保存: {csv_path} (形状: {df.shape})")
        
        return mae_dict
    
    # 扩展SE张量以匹配输入维度（注意：SE不需要batch维度）
    SE_tensor_for_model = tf.constant(np.repeat(SE, F, axis=0), dtype=tf.float32)  # (392, 64)
    
    # 保存三个数据集
    all_mae = {}
    all_mae['train'] = save_dataset_predictions(model, 'train', X_train, Y_train, trainTE, SE_tensor_for_model, mean, std)
    all_mae['val'] = save_dataset_predictions(model, 'val', X_val, Y_val, valTE, SE_tensor_for_model, mean, std)
    all_mae['test'] = save_dataset_predictions(model, 'test', X_test, Y_test, testTE, SE_tensor_for_model, mean, std)
    
    # 保存汇总
    summary_df = pd.DataFrame(all_mae)
    summary_path = os.path.join(MODEL_DIR, 'mae_summary.csv')
    summary_df.to_csv(summary_path, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print("✅ 所有预测结果已保存完成!")
    print(f"{'='*60}")
    print(f"\n📊 MAE汇总:")
    print(summary_df)


if __name__ == '__main__':
    train_model()