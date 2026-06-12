"""
GMAN 高速公路交通流量预测 - 4特征版本 (TF 2.x 原生实现)
支持: 小客车上行、小客车下行、非小客车上行、非小客车下行
时序配置: 8输入8输出
"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
parser.add_argument('--L', type=int, default=3, help='STAtt Block数量')
parser.add_argument('--K', type=int, default=8, help='注意力头数')
parser.add_argument('--d', type=int, default=16, help='每个注意力头的维度')
parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例')
parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
parser.add_argument('--test_ratio', type=float, default=0.2, help='测试集比例')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--max_epoch', type=int, default=50, help='最大epoch数')
parser.add_argument('--patience', type=int, default=10, help='早停耐心度')
parser.add_argument('--learning_rate', type=float, default=0.002, help='初始学习率')
parser.add_argument('--decay_epoch', type=int, default=10, help='学习率衰减周期')
parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')

# 路径配置
DATA_DIR = '/home/user/Downloads/cai/GMAN/data/highway_4feat'
MODEL_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
LOG_FILE = '/home/user/Downloads/cai/GMAN/logs/highway_4feat.log'

args = parser.parse_args()

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# ==================== 数据加载 ====================
def load_data():
    print("📊 加载数据...")
    data = np.load(os.path.join(DATA_DIR, 'highway_data.npz'))
    
    X_train = data['trainX'].astype(np.float32)
    Y_train = data['trainY'].astype(np.float32)
    trainTE = data['trainTE'].astype(np.float32)
    
    X_val = data['valX'].astype(np.float32)
    Y_val = data['valY'].astype(np.float32)
    valTE = data['valTE'].astype(np.float32)
    
    X_test = data['testX'].astype(np.float32)
    Y_test = data['testY'].astype(np.float32)
    testTE = data['testTE'].astype(np.float32)
    
    SE = data['SE'].astype(np.float32)
    mean = float(data['mean'])
    std = float(data['std'])
    
    print(f"   训练集: X={X_train.shape}, Y={Y_train.shape}, TE={trainTE.shape}")
    print(f"   验证集: X={X_val.shape}, Y={Y_val.shape}, TE={valTE.shape}")
    print(f"   测试集: X={X_test.shape}, Y={Y_test.shape}, TE={testTE.shape}")
    print(f"   空间嵌入: SE={SE.shape}")
    print(f"   归一化: mean={mean:.4f}, std={std:.4f}")
    
    return (X_train, Y_train, trainTE, 
            X_val, Y_val, valTE, 
            X_test, Y_test, testTE,
            SE, mean, std)


# ==================== GMAN模型组件 ====================
class FC(layers.Layer):
    def __init__(self, units, activation=None, use_bias=True, bn=False, name='fc'):
        super(FC, self).__init__(name=name)
        self.units = units if isinstance(units, list) else [units]
        self.activation = activation
        self.use_bias = use_bias
        self.bn = bn
        
        self.conv_layers = []
        self.bn_layers = []
        
        for i, units in enumerate(self.units):
            self.conv_layers.append(
                layers.Conv2D(units, kernel_size=(1, 1), use_bias=use_bias, 
                             name=f'{name}_conv_{i}')
            )
            if bn and activation is not None:
                self.bn_layers.append(layers.BatchNormalization(name=f'{name}_bn_{i}'))
    
    def call(self, x, training=False):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.bn and self.activation is not None and i < len(self.bn_layers):
                x = self.bn_layers[i](x, training=training)
            if self.activation is not None:
                x = self.activation(x)
        return x


class STEmbedding(layers.Layer):
    def __init__(self, D, T, bn=False, name='st_embedding'):
        super(STEmbedding, self).__init__(name=name)
        self.D = D
        self.T = T
        self.bn = bn
        self.se_fc = FC([D, D], activation=tf.nn.relu, bn=bn, name='SE')
        self.te_fc = FC([D, D], activation=tf.nn.relu, bn=bn, name='TE')
    
    def call(self, se, te, training=False):
        se_expanded = tf.expand_dims(tf.expand_dims(se, 0), 0)
        se_processed = self.se_fc(se_expanded, training=training)
        
        te_int = tf.cast(te, tf.int32)
        dayofweek = tf.one_hot(te_int[..., 0], depth=7)
        timeofday = tf.one_hot(te_int[..., 1], depth=self.T)
        te_concat = tf.concat([dayofweek, timeofday], axis=-1)
        te_expanded = tf.expand_dims(te_concat, axis=2)
        te_processed = self.te_fc(te_expanded, training=training)
        return se_processed + te_processed


class SpatialAttention(layers.Layer):
    def __init__(self, K, d, bn=False, name='spatial_attention'):
        super(SpatialAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K*d
        self.bn = bn
        self.query_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.key_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.value_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.output_fc = FC([self.D, self.D], tf.nn.relu, bn=bn)
    
    def call(self, x, ste, training=False):
        x_concat = tf.concat([x, ste], -1)
        q = self.query_fc(x_concat, training=training)
        k = self.key_fc(x_concat, training=training)
        v = self.value_fc(x_concat, training=training)
        
        qs = tf.split(q, self.K, -1)
        ks = tf.split(k, self.K, -1)
        vs = tf.split(v, self.K, -1)
        
        qm = tf.concat(qs, 0)
        km = tf.concat(ks, 0)
        vm = tf.concat(vs, 0)
        
        att = tf.matmul(qm, km, transpose_b=True)/(self.d**0.5)
        att = tf.nn.softmax(att)
        out = tf.matmul(att, vm)
        out = tf.concat(tf.split(out, self.K, 0), -1)
        return self.output_fc(out, training=training)


class TemporalAttention(layers.Layer):
    def __init__(self, K, d, bn=False, mask=True, name='temporal_attention'):
        super(TemporalAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K*d
        self.bn = bn
        self.mask = mask
        self.query_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.key_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.value_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.output_fc = FC([self.D, self.D], tf.nn.relu, bn=bn)
    
    def call(self, x, ste, training=False):
        x_concat = tf.concat([x, ste], -1)
        q = self.query_fc(x_concat, training=training)
        k = self.key_fc(x_concat, training=training)
        v = self.value_fc(x_concat, training=training)
        
        qs = tf.split(q, self.K, -1)
        ks = tf.split(k, self.K, -1)
        vs = tf.split(v, self.K, -1)
        
        qm = tf.concat(qs,0)
        km = tf.concat(ks,0)
        vm = tf.concat(vs,0)
        
        qt = tf.transpose(qm, [0,2,1,3])
        kt = tf.transpose(km, [0,2,3,1])
        vt = tf.transpose(vm, [0,2,1,3])
        
        att = tf.matmul(qt, kt)/(self.d**0.5)
        
        if self.mask:
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            N = tf.shape(x)[2]
            m = tf.linalg.band_part(tf.ones([T,T], dtype=tf.float32),-1,0)
            m = tf.tile(tf.expand_dims(tf.expand_dims(m,0),0),[self.K*B,N,1,1])
            att = tf.where(m>0, att, -1e9)
        
        att = tf.nn.softmax(att)
        out = tf.matmul(att, vt)
        out = tf.transpose(out, [0,2,1,3])
        out = tf.concat(tf.split(out, self.K,0),-1)
        return self.output_fc(out, training=training)


class GatedFusion(layers.Layer):
    def __init__(self, D, bn=False, name='gated_fusion'):
        super(GatedFusion, self).__init__(name=name)
        self.xs_fc = FC(D, None, False, bn=bn)
        self.xt_fc = FC(D, None, True, bn=bn)
        self.output_fc = FC([D,D], tf.nn.relu, bn=bn)
    
    def call(self, hs, ht, training=False):
        xs = self.xs_fc(hs, training=training)
        xt = self.xt_fc(ht, training=training)
        z = tf.nn.sigmoid(xs+xt)
        h = z*hs + (1-z)*ht
        return self.output_fc(h, training=training)


class TransformAttention(layers.Layer):
    def __init__(self, K, d, bn=False, name='transform_attention'):
        super(TransformAttention, self).__init__(name=name)
        self.K = K
        self.d = d
        self.D = K*d
        self.query_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.key_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.value_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.output_fc = FC([self.D,self.D], tf.nn.relu, bn=bn)
    
    def call(self, x, ste_p, ste_q, training=False):
        q = self.query_fc(ste_q, training=training)
        k = self.key_fc(ste_p, training=training)
        v = self.value_fc(x, training=training)
        
        qs = tf.split(q,self.K,-1)
        ks = tf.split(k,self.K,-1)
        vs = tf.split(v,self.K,-1)
        
        qm = tf.concat(qs,0)
        km = tf.concat(ks,0)
        vm = tf.concat(vs,0)
        
        qt = tf.transpose(qm, [0,2,1,3])
        kt = tf.transpose(km, [0,2,3,1])
        vt = tf.transpose(vm, [0,2,1,3])
        
        att = tf.matmul(qt,kt)/(self.d**0.5)
        att = tf.nn.softmax(att)
        out = tf.matmul(att,vt)
        out = tf.transpose(out, [0,2,1,3])
        out = tf.concat(tf.split(out,self.K,0),-1)
        return self.output_fc(out, training=training)


class STAttBlock(layers.Layer):
    def __init__(self, K, d, bn=False, name='st_att_block'):
        super(STAttBlock, self).__init__(name=name)
        self.spatial_att = SpatialAttention(K,d,bn=bn)
        self.temporal_att = TemporalAttention(K,d,bn=bn)
        self.gated_fusion = GatedFusion(K*d,bn=bn)
    
    def call(self, x, ste, training=False):
        hs = self.spatial_att(x, ste, training=training)
        ht = self.temporal_att(x, ste, training=training)
        return self.gated_fusion(hs, ht, training=training)


class GMANModel(tf.keras.Model):
    def __init__(self, P, Q, T, L, K, d, D_se, bn=False, name='GMAN'):
        super(GMANModel, self).__init__(name=name)
        self.P=P; self.Q=Q; self.L=L; self.K=K; self.d=d; self.D=K*d
        
        self.input_proj = FC([self.D,self.D], tf.nn.relu, bn=bn)
        self.st_embedding = STEmbedding(self.D, T, bn=bn)
        
        self.encoder_blocks = [STAttBlock(K,d,bn) for _ in range(L)]
        self.transform_att = TransformAttention(K,d,bn)
        self.decoder_blocks = [STAttBlock(K,d,bn) for _ in range(L)]
        self.output_proj = FC([self.D,1], tf.nn.relu, bn=bn)
    
    def call(self, inputs, training=False):
        X = inputs['X']
        TE = inputs['TE']
        SE = inputs['SE']
        
        N, D = SE.shape
        NF = X.shape[2]
        F = NF//N
        SE = tf.repeat(SE, F, axis=0)
        
        X = tf.expand_dims(X,-1)
        X = self.input_proj(X, training=training)
        
        STE = self.st_embedding(SE, TE, training=training)
        STE_P = STE[:,:self.P]
        STE_Q = STE[:,self.P:]
        
        for blk in self.encoder_blocks:
            X = blk(X, STE_P, training=training)
        
        X = self.transform_att(X, STE_P, STE_Q, training=training)
        
        for blk in self.decoder_blocks:
            X = blk(X, STE_Q, training=training)
        
        X = self.output_proj(X, training=training)
        return tf.squeeze(X,-1)


# ==================== 训练函数 ====================
def train_model():
    print(args)
    (X_train, Y_train, trainTE, X_val, Y_val, valTE, X_test, Y_test, testTE, SE, mean, std) = load_data()
    
    train_dataset = tf.data.Dataset.from_tensor_slices(({'X':X_train,'TE':trainTE}, Y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(({'X':X_val,'TE':valTE}, Y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices(({'X':X_test,'TE':testTE}, Y_test)).batch(32).prefetch(tf.data.AUTOTUNE)
    
    model = GMANModel(P=args.P, Q=args.Q, T=24, L=args.L, K=args.K, d=args.d, D_se=SE.shape[1], bn=True)
    SE_tensor = tf.constant(SE, tf.float32)
    
    sample_input = {'X':X_train[:1], 'TE':trainTE[:1], 'SE':SE_tensor}
    model(sample_input, training=False)
    
    optimizer = optimizers.Adam(args.learning_rate)
    
    # ==================== 修复类型错误：统一转 float32 ====================
    def huber_loss(y_true, y_pred, delta=1.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        error = y_true - y_pred
        abs_err = tf.abs(error)
        quad = tf.minimum(abs_err, delta)
        lin = abs_err - quad
        loss = 0.5 * tf.square(quad) + delta * lin
        
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    
    def masked_mae(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        return tf.reduce_sum(tf.abs(y_true - y_pred) * mask) / tf.reduce_sum(mask)
    
    @tf.function
    def train_step(bx,by):
        with tf.GradientTape() as tape:
            pred = model({**bx,'SE':SE_tensor}, training=True)
            loss = huber_loss(by,pred)
        grad = tape.gradient(loss, model.trainable_variables)
        grad,_ = tf.clip_by_global_norm(grad, args.grad_clip)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return loss
    
    @tf.function
    def val_step(bx,by):
        pred = model({**bx,'SE':SE_tensor}, training=False)
        return masked_mae(by,pred)
    
    log = open(LOG_FILE,'w')
    best_val = 1e9
    patience = 0
    
    for ep in range(args.max_epoch):
        t0 = time.time()
        tl = []
        for bx,by in train_dataset:
            tl.append(train_step(bx,by).numpy())
        vl = []
        for bx,by in val_dataset:
            vl.append(val_step(bx,by).numpy())
        at = np.mean(tl)
        av = np.mean(vl)
        log_string(log,f"Epoch {ep+1:2d} | Train {at:.4f} | Val {av:.4f} | {time.time()-t0:.1f}s")
        
        if av < best_val:
            best_val = av
            patience = 0
            model.save_weights(os.path.join(MODEL_DIR,'best_model.weights.h5'))
            log_string(log,"   ✅ 保存最佳模型")
        else:
            patience +=1
            if patience >= args.patience:
                log_string(log,"⏹️ 早停")
                break
    
    log.close()
    model.load_weights(os.path.join(MODEL_DIR,'best_model.weights.h5'))
    
    # ==================== 预测并保存CSV ====================
    from datetime import datetime, timedelta
    import pandas as pd
    
    START = datetime(2023,9,1)
    N, F, P, Q = 98, 4, args.P, args.Q
    SE_expand = tf.repeat(SE_tensor, F, axis=0)
    
    def pred_and_save(name, X, Y, TE):
        preds = []
        for i in range(0,len(X),32):
            bx = X[i:i+32]
            bte = TE[i:i+32]
            p = model({'X':bx,'TE':bte,'SE':SE_expand}, training=False)
            preds.append(p.numpy())
        pred = np.concatenate(preds,0)
        
        pred = pred*std+mean
        true = Y*std+mean
        
        samples = pred.shape[0]
        pred = pred.reshape(samples,Q,N,F)
        true = true.reshape(samples,Q,N,F)
        
        rows = []
        for s in range(samples):
            for t in range(Q):
                ts = START + timedelta(hours=s+t)
                for n in range(N):
                    rows.append([
                        s, t, n, ts.strftime('%Y-%m-%d %H:%M'),
                        pred[s,t,n,0], true[s,t,n,0],
                        pred[s,t,n,1], true[s,t,n,1],
                        pred[s,t,n,2], true[s,t,n,2],
                        pred[s,t,n,3], true[s,t,n,3],
                    ])
        
        df = pd.DataFrame(rows, columns=[
            'sample','time_step','station','time',
            '小客车上行_预测','小客车上行_真实',
            '小客车下行_预测','小客车下行_真实',
            '非小客车上行_预测','非小客车上行_真实',
            '非小客车下行_预测','非小客车下行_真实',
        ])
        path = os.path.join(MODEL_DIR,f'{name}_predictions.csv')
        df.to_csv(path,index=False,encoding='utf-8-sig')
        print(f'✅ {name} 集保存完成 → {path}')
    
    print("\n🚀 开始保存所有预测结果...")
    pred_and_save('train', X_train, Y_train, trainTE)
    pred_and_save('val', X_val, Y_val, valTE)
    pred_and_save('test', X_test, Y_test, testTE)
    print("\n🎉 全部完成！")


if __name__ == '__main__':
    train_model()