# FaST-MV (Fast Adaptive Spatio-Temporal with Multi-Variable)
# 支持多变量输入输出的 FaST 模型
# 借鉴 PatchTST 的 Channel Independence 思想
#
# 数据格式：[B, L, N, C]
#   - B: batch size
#   - L: input length (e.g., 8)
#   - N: num_nodes (e.g., 160)
#   - C: num_features (e.g., 2 = [小客车, 非小客车])
#
# 核心改造：
# 1. 移除 TOD/DOW 日历特征依赖
# 2. 引入可学习位置编码 (Learnable Positional Encoding)
# 3. 支持多变量输入输出（每个变量独立处理或联合建模）

import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from easytorch.device import to_device
import math

# ---------------------------
# RMSNorm：均方根归一化
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
            
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed

# ---------------------------
# 可学习参数层
# ---------------------------
class TrainableParameterLayer(nn.Module):
    def __init__(self, shape):
        super(TrainableParameterLayer, self).__init__()
        self.parameter = nn.Parameter(torch.empty(*shape))
        nn.init.xavier_uniform_(self.parameter)

    def forward(self, indices):
        return self.parameter[indices]

# ---------------------------
# 🔥 可学习位置编码 (借鉴 PatchTST)
# ---------------------------
class LearnablePositionalEncoding(nn.Module):
    """
    可学习的绝对位置编码，替代 TOD/DOW 外部时间特征
    为每个时间步提供独立的位置信息
    """
    """ 移除了日内 周内 的时间特征  """
    def __init__(self, seq_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        # 为每个时间步学习一个 d_model 维的向量
        self.pos_encoding = nn.Parameter(torch.empty(seq_len, d_model))
        nn.init.uniform_(self.pos_encoding, -0.02, 0.02)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, L, D] 批次,节点,时间步,维度
        Returns:
            x + pos_encoding: 加入位置信息
        """
        if x.dim() == 4:
            # x: [B, N, L, D]
            pos = self.pos_encoding.unsqueeze(0).unsqueeze(1)  # [1, 1, L, D]
            return x + pos
        else:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

# ---------------------------
# 简化版路由层（移除时间/星期偏置）
# ---------------------------
class SimplifiedRoutingLayer(nn.Module):
    """
    简化路由层：仅保留节点偏置和基础路由
    移除 TOD/DOW 依赖，实现 Channel Independence
    """
    def __init__(self, router_fea_dim, num_experts, num_nodes):
        super(SimplifiedRoutingLayer, self).__init__()
        self.num_nodes = num_nodes
        # 基础路由分数
        self.router_logit_layer = nn.Linear(router_fea_dim, num_experts)
        # 仅保留节点偏置
        self.adaptive_router_node = TrainableParameterLayer([num_nodes, num_experts])

    def forward(self, x, node_idx):
        """
        Args:
            x: [B, N, D] 历史数据特征
            node_idx: [1, N] 节点索引
        Returns:
            router: [B, N, E] 专家选择概率
        """
        router = self.router_logit_layer(x)
        router += self.adaptive_router_node(node_idx)
        router = F.softmax(router, dim=-1)
        return router

# ---------------------------
# GLU：门控线性单元
# ---------------------------
class GLU(nn.Module):
    def __init__(self, in_dim, out_dim=-1):
        super(GLU, self).__init__()
        if out_dim < 0:
            out_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, g = torch.chunk(self.linear(x), chunks=2, dim=-1)
        return x * F.sigmoid(g)

# ---------------------------
# 并行混合专家层
# ---------------------------
class ParallelMoEWithGLU(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, num_nodes, res_flag=True):
        super(ParallelMoEWithGLU, self).__init__()
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_nodes = num_nodes
        self.res_flag = res_flag
        
        self.GLU_Experts = GLU(in_dim, num_experts * out_dim)
        if res_flag:
            self.norm = RMSNorm(d=out_dim)

    def forward(self, x, router):
        """
        Args:
            x: [B, N, D]
            router: [B, N, E]
        Returns:
            output: [B, N, D]
        """
        res = x
        x = self.GLU_Experts(x).view(-1, self.num_nodes, self.num_experts, self.out_dim)
        x = torch.einsum("bne,bned->bnd", router, x)
        
        if self.res_flag:
            return self.norm(x + res), x
        return x

# ---------------------------
# 双向图注意力机制 (AAGA)
# ---------------------------
class AAGA(nn.Module):
    def __init__(self, dim):
        super(AAGA, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.agent = nn.Linear(dim, dim * 2)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = RMSNorm(d=dim)

    def forward(self, agent, x):
        """
        Args:
            agent: [K, D] 代理节点
            x: [B, N, D] 节点特征
        """
        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        q_agent, k_agent = torch.chunk(self.agent(agent), chunks=2, dim=-1)

        # Graph-to-Agent
        attn = torch.einsum("kd,bnd->bkn", (q_agent, k))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc1(v)

        # Agent-to-Graph
        attn = torch.einsum("bnd,kd->bnk", (q, k_agent))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc2(v)

        return self.norm(v + x)

# ---------------------------
# MLP 输出层
# ---------------------------
class mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mlp, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

# ---------------------------
# 🔥🔥 FaST-MV 主模型（多变量版本）
# ---------------------------
class FaST_MV(nn.Module):
    """
    FaST with Multi-Variable Support
    支持多变量输入输出（如：小客车 + 非小客车）
    
    输入格式：
    - [B, L, N, C]
    - B: batch size
    - L: input length (e.g., 8)
    - N: num_nodes (e.g., 160)
    - C: num_features (e.g., 2 = [小客车, 非小客车])
    
    输出格式：
    - [B, P, N, C]
    - P: output length (e.g., 8)
    """
    def __init__(
        self,
        num_nodes,          # 站点数 (e.g., 160)
        num_features=2,     # 🔥 特征数 (e.g., 2: 小客车+非小客车)
        input_len=8,
        output_len=8,
        layers=3,
        num_experts=8,
        hidden_dim=64,
        num_agent=32,
        use_revIN=True,     # 是否使用 RevIN 归一化
        channel_independent=True,  # 🔥 是否使用 Channel Independence
    ):
        super(FaST_MV, self).__init__()
        self.L = input_len
        self.layers = layers
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.use_revIN = use_revIN
        self.channel_independent = channel_independent
        
        # 节点索引
        self.node_idx = to_device(torch.arange(self.num_nodes)).unsqueeze(0)  # [1, N]
        
        # 🔥 关键：根据是否使用 CI 调整输入维度
        if channel_independent:
            # Channel Independent：每个特征独立处理
            # 输入维度 = input_len（时间序列长度）
            input_dim = input_len
        else:
            # 联合建模：将所有特征拼接
            # 输入维度 = input_len × num_features
            input_dim = input_len * num_features
        
        # 输入层路由（简化版，无时间偏置）
        self.input_layer = nn.ModuleList([
            SimplifiedRoutingLayer(input_dim, num_experts, num_nodes),
            ParallelMoEWithGLU(input_dim, hidden_dim, num_experts, num_nodes, res_flag=False)
        ])
        
        # 堆叠多层 AAGA + 路由 + MoE
        self.AAGA = nn.ModuleList()
        self.Router = nn.ModuleList()
        self.MoE = nn.ModuleList()
        for _ in range(layers):
            self.AAGA.append(AAGA(hidden_dim))
            self.Router.append(SimplifiedRoutingLayer(input_dim, num_experts, num_nodes))
            self.MoE.append(ParallelMoEWithGLU(hidden_dim, hidden_dim, num_experts, num_nodes))
        
        # 🔥 输出层：预测所有特征
        if channel_independent:
            # 每个特征独立预测
            self.output_layer = mlp(hidden_dim * (layers + 1), output_len)
        else:
            # 联合预测所有特征
            self.output_layer = mlp(hidden_dim * (layers + 1), output_len * num_features)
        
        # 可学习代理节点
        self.agent = nn.Parameter(torch.empty([num_agent, hidden_dim]))
        nn.init.xavier_uniform_(self.agent)
        
        # 🔥 可学习位置编码（替代 TOD/DOW 嵌入）
        self.pos_encoding = LearnablePositionalEncoding(input_len, hidden_dim)
        
        # 🔥 节点嵌入（区分不同站点）
        self.node_emb = TrainableParameterLayer([num_nodes, hidden_dim])
        
        # 🔥 特征嵌入（区分不同车辆类型）
        if channel_independent:
            self.feature_emb = TrainableParameterLayer([num_features, hidden_dim])
        else:
            self.feature_emb = None
        
        # 🔥 RevIN 归一化（每个特征独立归一化）
        if use_revIN:
            self.revin = RevIN(num_features=num_features)
        else:
            self.revin = None

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data: [B, L, N, C]
                - B: batch size
                - L: input length (e.g., 8)
                - N: num_nodes (e.g., 160)
                - C: num_features (e.g., 2 = [小客车, 非小客车])
        
        Returns:
            prediction: [B, P, N, C]
                - P: output length (e.g., 8)
        """
        B, L, N, C = history_data.shape
        
        # 验证输入维度
        assert N == self.num_nodes, f"节点数不匹配: {N} vs {self.num_nodes}"
        assert C == self.num_features, f"特征数不匹配: {C} vs {self.num_features}"
        
        # 转置为 [B, N, L, C] 便于处理
        raw = history_data.permute(0, 2, 1, 3).contiguous()  # [B, N, L, C]
        
        # 🔥 步骤1：RevIN 归一化（如果使用）
        if self.use_revIN and self.revin is not None:
            raw = self.revin(raw, 'norm')  # [B, N, L, C]
        else:
            # fallback：使用 Instance Norm
            seq_mean = torch.mean(raw, dim=2, keepdim=True)  # [B, N, 1, C]
            seq_var = torch.var(raw, dim=2, keepdim=True) + 1e-5
            raw = (raw - seq_mean) / torch.sqrt(seq_var)
        
        # 🔥 步骤2：根据 CI 模式处理
        if self.channel_independent:
            # Channel Independent：每个特征独立处理
            predictions = []
            
            for c in range(self.num_features):
                # 提取第 c 个特征
                raw_c = raw[:, :, :, c]  # [B, N, L]
                
                # 输入层 MoE
                router = self.input_layer[0](raw_c, self.node_idx)  # [B, N, E]
                x = self.input_layer[1](raw_c, router)  # [B, N, D]
                
                # 加入位置编码 + 节点嵌入 + 特征嵌入
                x = x.unsqueeze(2).expand(-1, -1, self.L, -1)  # [B, N, L, D]
                x = self.pos_encoding(x)  # 加入位置信息
                x = x.mean(dim=2)  # 聚合回 [B, N, D]
                
                x += self.node_emb(self.node_idx)  # [B, N, D]
                x += self.feature_emb(torch.tensor([c], device=x.device))  # 加入特征嵌入
                
                # 堆叠核心层
                skip = [x]
                for i in range(self.layers):
                    x = self.AAGA[i](self.agent, x)  # 图注意力
                    router = self.Router[i](raw_c, self.node_idx)  # 路由
                    x, s = self.MoE[i](x, router)  # MoE
                    skip.append(s)
                
                # 拼接所有层输出
                x = torch.cat(skip, dim=-1)  # [B, N, D*(L+1)]
                
                # 预测
                pred_c = self.output_layer(x)  # [B, N, P]
                predictions.append(pred_c)
            
            # 拼接所有特征的预测
            x = torch.stack(predictions, dim=-1)  # [B, N, P, C]
            x = x.permute(0, 2, 1, 3).contiguous()  # [B, P, N, C]
            
        else:
            # 联合建模：将所有特征拼接
            # 拼接为 [B, N, L*C]
            raw_flat = raw.reshape(B, N, -1)  # [B, N, L*C]
            
            # 输入层 MoE
            router = self.input_layer[0](raw_flat, self.node_idx)
            x = self.input_layer[1](raw_flat, router)
            
            # 加入位置编码 + 节点嵌入
            x = x.unsqueeze(2).expand(-1, -1, self.L, -1)
            x = self.pos_encoding(x)
            x = x.mean(dim=2)
            
            x += self.node_emb(self.node_idx)
            
            # 堆叠核心层
            skip = [x]
            for i in range(self.layers):
                x = self.AAGA[i](self.agent, x)
                router = self.Router[i](raw_flat, self.node_idx)
                x, s = self.MoE[i](x, router)
                skip.append(s)
            
            # 拼接输出
            x = torch.cat(skip, dim=-1)
            
            # 预测所有特征
            x = self.output_layer(x)  # [B, N, P*C]
            x = x.view(B, N, self.output_len, self.num_features)  # [B, N, P, C]
            x = x.permute(0, 2, 1, 3).contiguous()  # [B, P, N, C]
        
        # 🔥 步骤3：反归一化
        if self.use_revIN and self.revin is not None:
            # x: [B, P, N, C] → RevIN 期望 [B, N, L, C]，其中 L=P
            x = x.permute(0, 2, 1, 3)  # [B, P, N, C] → [B, N, P, C]
            x = self.revin(x, 'denorm')
            x = x.permute(0, 2, 1, 3)  # [B, N, P, C] → [B, P, N, C]
        else:
            x = x * torch.sqrt(seq_var) + seq_mean

        return x  # [B, P, N, C]


# ---------------------------
# RevIN 归一化层（从 PatchTST 借鉴）
# ---------------------------
class RevIN(nn.Module):
    """Reversible Instance Normalization for Multi-Variable"""
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def _init_params(self):
        # 为每个特征学习独立的 affine 参数
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        """
        Args:
            x: [B, N, L, C]
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        # x: [B, N, L, C]
        # 沿时间维度计算统计量
        dim2reduce = (2,)  # 仅沿 L 维度
        if self.subtract_last:
            self.last = x[:, :, -1:, :]  # [B, N, 1, C]
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # [B, N, 1, C]
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()  # [B, N, 1, C]

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # affine_weight: [C] → [1, 1, 1, C]
            x = x * self.affine_weight.view(1, 1, 1, -1)
            x = x + self.affine_bias.view(1, 1, 1, -1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.view(1, 1, 1, -1)
            x = x / (self.affine_weight.view(1, 1, 1, -1) + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("FaST-MV 模型测试（多变量版本）")
    print("=" * 70)
    
    # 测试1：Channel Independent 模式
    print("\n📌 测试1：Channel Independent 模式")
    model_ci = FaST_MV(
        num_nodes=160,
        num_features=2,  # 小客车 + 非小客车
        input_len=8,
        output_len=8,
        layers=3,
        num_experts=8,
        hidden_dim=64,
        num_agent=32,
        use_revIN=True,
        channel_independent=True
    )
    
    # 模拟输入：[B, L, N, C]
    test_input = torch.randn(2, 8, 160, 2)
    print(f"输入形状: {test_input.shape}")
    print(f"  - B=2, L=8, N=160, C=2")
    
    model_ci.eval()
    with torch.no_grad():
        output_ci = model_ci(test_input)
    
    print(f"输出形状: {output_ci.shape}")
    print(f"预期形状: [2, 8, 160, 2]")
    assert output_ci.shape == (2, 8, 160, 2), "❌ 输出形状错误！"
    print("✅ Channel Independent 测试通过！")
    
    # 测试2：联合建模模式
    print("\n📌 测试2：联合建模模式")
    model_joint = FaST_MV(
        num_nodes=160,
        num_features=2,
        input_len=8,
        output_len=8,
        layers=3,
        num_experts=8,
        hidden_dim=64,
        num_agent=32,
        use_revIN=True,
        channel_independent=False
    )
    
    with torch.no_grad():
        output_joint = model_joint(test_input)
    
    print(f"输出形状: {output_joint.shape}")
    assert output_joint.shape == (2, 8, 160, 2), "❌ 输出形状错误！"
    print("✅ 联合建模测试通过！")
    
    # 统计信息
    print(f"\n📊 模型参数统计:")
    params_ci = sum(p.numel() for p in model_ci.parameters())
    params_joint = sum(p.numel() for p in model_joint.parameters())
    print(f"   Channel Independent: {params_ci:,}")
    print(f"   联合建模: {params_joint:,}")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！FaST-MV 模型准备就绪！")
    print("=" * 70)
