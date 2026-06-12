"""
GMAN 高速公路交通流量预测 - 4特征版本 (TF 2.x 原生实现)
✅ 已修复：真实值 错误导出问题
✅ 不重新训练，直接加载已有模型
✅ 预测 + 正确保存 CSV
python predictonly.py

"""

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras import layers, optimizers

# ==================== 配置参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=8, help='历史步长')
parser.add_argument('--Q', type=int, default=8, help='预测步长')
parser.add_argument('--L', type=int, default=3, help='STAtt Block数量')
parser.add_argument('--K', type=int, default=8, help='注意力头数')
parser.add_argument('--d', type=int, default=16, help='每个注意力头的维度')
parser.add_argument('--train_ratio', type=int, default=0.7)
parser.add_argument('--val_ratio', type=int, default=0.1)
parser.add_argument('--test_ratio', type=int, default=0.2)

# 路径配置
DATA_DIR = '/home/user/Downloads/cai/GMAN/data/highway_4feat'
MODEL_DIR = '/home/user/Downloads/cai/GMAN/models/highway_4feat'
args = parser.parse_args()

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
    
    print(f"训练集: X={X_train.shape}, Y={Y_train.shape}")
    print(f"验证集: X={X_val.shape}, Y={Y_val.shape}")
    print(f"测试集: X={X_test.shape}, Y={Y_test.shape}")
    print(f"归一化: mean={mean:.4f}, std={std:.4f}")
    
    return (X_train, Y_train, trainTE, 
            X_val, Y_val, valTE, 
            X_test, Y_test, testTE, SE, mean, std)


# ==================== GMAN模型结构 ====================
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
            self.conv_layers.append(layers.Conv2D(units, (1,1), use_bias=use_bias, name=f'{name}_conv_{i}'))
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
        self.se_fc = FC([D, D], tf.nn.relu, bn=bn, name='SE')
        self.te_fc = FC([D, D], tf.nn.relu, bn=bn, name='TE')
    
    def call(self, se, te, training=False):
        se_expanded = tf.expand_dims(tf.expand_dims(se, 0), 0)
        se_processed = self.se_fc(se_expanded, training=training)
        te_int = tf.cast(te, tf.int32)
        dayofweek = tf.one_hot(te_int[...,0], 7)
        timeofday = tf.one_hot(te_int[...,1], self.T)
        te_concat = tf.concat([dayofweek, timeofday], -1)
        te_expanded = tf.expand_dims(te_concat, 2)
        te_processed = self.te_fc(te_expanded, training=training)
        return se_processed + te_processed

class SpatialAttention(layers.Layer):
    def __init__(self, K, d, bn=False):
        super(SpatialAttention, self).__init__()
        self.K=K; self.d=d; self.D=K*d
        self.query_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.key_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.value_fc = FC(self.D, tf.nn.relu, bn=bn)
        self.output_fc = FC([self.D,self.D], tf.nn.relu, bn=bn)
    
    def call(self, x, ste, training=False):
        x_concat = tf.concat([x,ste],-1)
        q = self.query_fc(x_concat,training=training)
        k = self.key_fc(x_concat,training=training)
        v = self.value_fc(x_concat,training=training)
        qs = tf.split(q,self.K,-1)
        ks = tf.split(k,self.K,-1)
        vs = tf.split(v,self.K,-1)
        qm = tf.concat(qs,0)
        km = tf.concat(ks,0)
        vm = tf.concat(vs,0)
        att = tf.matmul(qm,km,transpose_b=True)/(self.d**0.5)
        att = tf.nn.softmax(att)
        out = tf.matmul(att,vm)
        out = tf.concat(tf.split(out,self.K,0),-1)
        return self.output_fc(out,training=training)

class TemporalAttention(layers.Layer):
    def __init__(self, K, d, bn=False, mask=True):
        super(TemporalAttention,self).__init__()
        self.K=K; self.d=d; self.D=K*d; self.mask=mask
        self.query_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.key_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.value_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.output_fc = FC([self.D,self.D],tf.nn.relu,bn=bn)
    
    def call(self,x,ste,training=False):
        x_concat = tf.concat([x,ste],-1)
        q = self.query_fc(x_concat,training=training)
        k = self.key_fc(x_concat,training=training)
        v = self.value_fc(x_concat,training=training)
        qs = tf.split(q,self.K,-1)
        ks = tf.split(k,self.K,-1)
        vs = tf.split(v,self.K,-1)
        qm = tf.concat(qs,0)
        km = tf.concat(ks,0)
        vm = tf.concat(vs,0)
        qt = tf.transpose(qm,[0,2,1,3])
        kt = tf.transpose(km,[0,2,3,1])
        vt = tf.transpose(vm,[0,2,1,3])
        att = tf.matmul(qt,kt)/(self.d**0.5)
        if self.mask:
            T = tf.shape(x)[1]
            m = tf.linalg.band_part(tf.ones([T,T],tf.float32),-1,0)
            m = tf.tile(tf.expand_dims(tf.expand_dims(m,0),0),[self.K*tf.shape(x)[0],tf.shape(x)[2],1,1])
            att = tf.where(m>0,att,-1e9)
        att = tf.nn.softmax(att)
        out = tf.matmul(att,vt)
        out = tf.transpose(out,[0,2,1,3])
        out = tf.concat(tf.split(out,self.K,0),-1)
        return self.output_fc(out,training=training)

class GatedFusion(layers.Layer):
    def __init__(self,D,bn=False):
        super(GatedFusion,self).__init__()
        self.xs_fc = FC(D,None,False,bn=bn)
        self.xt_fc = FC(D,None,True,bn=bn)
        self.output_fc = FC([D,D],tf.nn.relu,bn=bn)
    
    def call(self,hs,ht,training=False):
        xs = self.xs_fc(hs,training=training)
        xt = self.xt_fc(ht,training=training)
        z = tf.nn.sigmoid(xs+xt)
        h = z*hs + (1-z)*ht
        return self.output_fc(h,training=training)

class TransformAttention(layers.Layer):
    def __init__(self,K,d,bn=False):
        super(TransformAttention,self).__init__()
        self.K=K; self.d=d; self.D=K*d
        self.query_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.key_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.value_fc = FC(self.D,tf.nn.relu,bn=bn)
        self.output_fc = FC([self.D,self.D],tf.nn.relu,bn=bn)
    
    def call(self,x,ste_p,ste_q,training=False):
        q = self.query_fc(ste_q,training=training)
        k = self.key_fc(ste_p,training=training)
        v = self.value_fc(x,training=training)
        qs = tf.split(q,self.K,-1)
        ks = tf.split(k,self.K,-1)
        vs = tf.split(v,self.K,-1)
        qm = tf.concat(qs,0)
        km = tf.concat(ks,0)
        vm = tf.concat(vs,0)
        qt = tf.transpose(qm,[0,2,1,3])
        kt = tf.transpose(km,[0,2,3,1])
        vt = tf.transpose(vm,[0,2,1,3])
        att = tf.matmul(qt,kt)/(self.d**0.5)
        att = tf.nn.softmax(att)
        out = tf.matmul(att,vt)
        out = tf.transpose(out,[0,2,1,3])
        out = tf.concat(tf.split(out,self.K,0),-1)
        return self.output_fc(out,training=training)

class STAttBlock(layers.Layer):
    def __init__(self,K,d,bn=False):
        super(STAttBlock,self).__init__()
        self.spatial_att = SpatialAttention(K,d,bn)
        self.temporal_att = TemporalAttention(K,d,bn)
        self.gated_fusion = GatedFusion(K*d,bn)
    
    def call(self,x,ste,training=False):
        hs = self.spatial_att(x,ste,training=training)
        ht = self.temporal_att(x,ste,training=training)
        return self.gated_fusion(hs,ht,training=training)

class GMANModel(tf.keras.Model):
    def __init__(self,P,Q,T,L,Kd,D_se,bn=False):
        super(GMANModel,self).__init__()
        self.P=P; self.Q=Q; self.L=L; self.K=Kd[0]; self.d=Kd[1]; self.D=self.K*self.d
        self.input_proj = FC([self.D,self.D],tf.nn.relu,bn=bn)
        self.st_embedding = STEmbedding(self.D,T,bn)
        self.encoder_blocks = [STAttBlock(self.K,self.d,bn) for _ in range(L)]
        self.transform_att = TransformAttention(self.K,self.d,bn)
        self.decoder_blocks = [STAttBlock(self.K,self.d,bn) for _ in range(L)]
        self.output_proj = FC([self.D,1],tf.nn.relu,bn=bn)
    
    def call(self,inputs,training=False):
        X = inputs['X']
        TE = inputs['TE']
        SE = inputs['SE']
        N,_ = SE.shape
        F = X.shape[2]//N
        SE = tf.repeat(SE,F,axis=0)
        X = tf.expand_dims(X,-1)
        X = self.input_proj(X,training=training)
        STE = self.st_embedding(SE,TE,training=training)
        STE_P = STE[:,:self.P]
        STE_Q = STE[:,self.P:]
        for blk in self.encoder_blocks:
            X = blk(X,STE_P,training=training)
        X = self.transform_att(X,STE_P,STE_Q,training=training)
        for blk in self.decoder_blocks:
            X = blk(X,STE_Q,training=training)
        X = self.output_proj(X,training=training)
        return tf.squeeze(X,-1)

# ==================== 【核心】只预测，不训练，直接导出正确CSV ====================
def predict_and_save():
    from datetime import datetime, timedelta
    import pandas as pd

    # 加载数据（归一化的用于模型输入）
    X_train, Y_train_norm, trainTE, X_val, Y_val_norm, valTE, X_test, Y_test_norm, testTE, SE, mean, std = load_data()

    # 构建模型
    model = GMANModel(P=args.P, Q=args.Q, T=24, L=args.L, Kd=[args.K, args.d], D_se=SE.shape[1], bn=True)
    SE_tensor = tf.constant(SE, tf.float32)
    model({'X': X_train[:1], 'TE': trainTE[:1], 'SE': SE_tensor}, training=False)

    # 加载你已经训练好的模型 ✅
    model.load_weights(os.path.join(MODEL_DIR, 'best_model.weights.h5'))
    print("✅ 模型加载成功！")

    # 配置
    START = datetime(2023, 9, 1)
    N, F, Q = 98, 4, args.Q
    SE_expand = tf.repeat(SE_tensor, F, axis=0)

    # ========== 导出函数（简化逻辑：直接反归一化 Y_norm）==========
    def pred_and_save(name, X, Y_norm, TE):
        preds = []
        for i in range(0, len(X), 32):
            bx = X[i:i+32]
            bte = TE[i:i+32]
            p = model({'X': bx, 'TE': bte, 'SE': SE_expand}, training=False)
            preds.append(p.numpy())
        pred = np.concatenate(preds, 0)

        # ✅ 关键修复：预测值和真实值都反归一化到原始空间
        pred_denorm = pred * std + mean
        true_denorm = Y_norm * std + mean  # Y_norm是归一化的，需要反归一化

        # 形状对齐
        pred_denorm = pred_denorm.reshape(pred_denorm.shape[0], Q, N, F)
        true_denorm = true_denorm.reshape(true_denorm.shape[0], Q, N, F)

        rows = []
        for s in range(pred_denorm.shape[0]):
            for t in range(Q):
                ts = START + timedelta(hours=s + t)
                for n in range(N):
                    rows.append([
                        s, t, n, ts.strftime('%Y-%m-%d %H:%M'),
                        pred_denorm[s,t,n,0], true_denorm[s,t,n,0],
                        pred_denorm[s,t,n,1], true_denorm[s,t,n,1],
                        pred_denorm[s,t,n,2], true_denorm[s,t,n,2],
                        pred_denorm[s,t,n,3], true_denorm[s,t,n,3],
                    ])

        df = pd.DataFrame(rows, columns=[
            'sample','time_step','station','time',
            '小客车上行_预测','小客车上行_真实',
            '小客车下行_预测','小客车下行_真实',
            '非小客车上行_预测','非小客车上行_真实',
            '非小客车下行_预测','非小客车下行_真实',
        ])
        path = os.path.join(MODEL_DIR, f'{name}_predictions.csv')
        df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f'✅ {name} 集已保存 → {path}')
        
        # 计算并打印指标（在原始空间计算）
        mae = np.mean(np.abs(pred_denorm - true_denorm))
        rmse = np.sqrt(np.mean((pred_denorm - true_denorm) ** 2))
        print(f'   MAE: {mae:.2f}, RMSE: {rmse:.2f}')

    # 导出全部
    print("\n🚀 开始导出预测结果...")
    pred_and_save('train', X_train, Y_train_norm, trainTE)
    pred_and_save('val', X_val, Y_val_norm, valTE)
    pred_and_save('test', X_test, Y_test_norm, testTE)
    print("\n🎉 全部导出完成！")

if __name__ == '__main__':
    predict_and_save()