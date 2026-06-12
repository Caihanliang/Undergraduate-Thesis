# FaST (Fast Adaptive Spatio-Temporal) 是一个基于混合专家系统 (MoE) 和自适应图注意力机制的时间序列预测模型
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from easytorch.device import to_device

#  只使用 RMS 进行归一化
"""
移除均值中心化，仅通过均方根缩放
支持部分 RMSNorm（参数 p 控制）
计算效率更高，适合大规模模型
"""
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
            Zhang B, Sennrich R. Root mean square layer normalization. Advances in neural information processing systems, 2019, 32.
        :param d: model size 特征维度
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled) 部分RMSNorm比例，默认-1表示关闭
        :param eps:  epsilon value, default 1e-8 防止除0的极小值
        :param bias: whether use bias term for RMSNorm, disabled by   是否加偏置
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps  # 防止分母为0
        self.d = d   # 输入维度 特征维度
        self.p = p   # 部分归一化比例 部分RMSNorm比例，默认-1表示关闭
        self.bias = bias  # 是否加偏置
        """
        假设输入特征为 x∈Rd
        则经过缩放和偏置后的输出为：y=scale⋅x+offset
        scale和 offset是可学习的参数，维度与输入特征相同（或可以广播）
        """
        # 可学习缩放参数
        self.scale = nn.Parameter(torch.ones(d))  #创建一个维度为d的可学习参数scale，初始值全为1
        self.register_parameter("scale", self.scale) #将self.scale这个参数注册到当前模块中，命名为"scale"
         # 如果需要偏置，创建可学习偏置
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        # 不使用部分归一化
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True) # 计算L2范数（在最后一个维度上）
            d_x = self.d
        else:
            # 部分维度归一化（很少用）
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        # 计算均方根
        rms_x = norm_x * d_x ** (-1. / 2)
        # 归一化
        x_normed = x / (rms_x + self.eps)
        # 缩放 + 偏置
        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

# ---------------------------
# 可学习参数层：用于时间/节点/星期嵌入
# 作用：存储可学习向量（如节点嵌入、时间嵌入）
# ---------------------------
class TrainableParameterLayer(nn.Module):
    def __init__(self, shape):  # shape = [num_embeddings, dim]  # shape = [数量, 维度]
        super(TrainableParameterLayer, self).__init__()
        self.parameter = nn.Parameter(torch.empty(*shape)) # 新建可学习参数
        nn.init.xavier_uniform_(self.parameter)# 初始化

    def forward(self, indices):
        return self.parameter[indices] # 根据索引取出对应嵌入

# 分层自适应路由层  动态选择专家网络，实现条件计算
"""
历史数据特征 [batch, nodes, input_len]
时间索引：日内时刻 (day_idx)、周内日期 (week_idx)、节点编号 (node_idx)
"""
class HARoutingLayer(nn.Module):
    def __init__(self, router_fea_dim, num_experts, daily_steps, weekly_days, num_nodes):
        super(HARoutingLayer, self).__init__()
        self.daily_steps = daily_steps # 一天有多少个时间步
        self.weekly_days = weekly_days # 一周7天
        self.num_nodes = num_nodes     # 节点数
        # 输出专家概率的线性层
        self.router_logit_layer = nn.Linear(router_fea_dim, num_experts)
        # 三个自适应偏置：时间、星期、节点
        self.adaptive_router_day  = TrainableParameterLayer([daily_steps, num_experts])
        self.adaptive_router_week = TrainableParameterLayer([weekly_days, num_experts])
        self.adaptive_router_node = TrainableParameterLayer([num_nodes, num_experts])

    def forward(self, x, day_idx, week_idx, node_idx):
        # router logit 基础路由分数
        router = self.router_logit_layer(x)
        # +adaptive_router_day bias 加入时间偏置
        router += self.adaptive_router_day(day_idx)
        # +adaptive_router_week bias 加入星期偏置
        router += self.adaptive_router_week(week_idx)
        # +adaptive_router_node bias 加入节点偏置
        router += self.adaptive_router_node(node_idx)
        # Probabilistic 归一化为概率（和为1）
        router = F.softmax(router, dim=-1)
        return router
# ---------------------------
# GLU：门控线性单元
# 作用：增强特征表达、过滤无效信息
# ---------------------------
class GLU(nn.Module):
    def __init__(self, in_dim, out_dim=-1):
        super(GLU, self).__init__()
        if out_dim<0: out_dim = in_dim
        self.linear = nn.Linear(in_dim, out_dim*2) # 输出2倍维度，用于拆分

    def forward(self, x):
        # x:b,n,d # 拆成两部分：特征x + 门控g
        x, g = torch.chunk(self.linear(x), chunks=2, dim=-1)
        return x * F.sigmoid(g) # 门控加权

# 并行混合专家层  多个专家并行计算 → 路由加权融合
class ParallelMoEWithGLU(nn.Module):
    def __init__(self,in_dim, out_dim, num_experts, num_nodes, res_flag=True):
        super(ParallelMoEWithGLU, self).__init__()
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.num_nodes = num_nodes
        self.res_flag = res_flag # 是否残差连接
        # 用GLU同时表示所有专家
        self.GLU_Experts = GLU(in_dim, num_experts * out_dim)
        if res_flag:
            self.norm = RMSNorm(d=out_dim)

    def forward(self, x, router):
        """x:b,n,d  批次, 节点数, 维度 """

        res = x # 残差
        # reshape: b,n,ed->b,n,e,d
        # 把输出 reshape 成：批次,节点,专家,维度
        x = self.GLU_Experts(x).view(-1, self.num_nodes, self.num_experts, self.out_dim)
        x = torch.einsum("bne,bned->bnd", router, x)
        if self.res_flag:
            return self.norm(x + res),x
        return x

# 双向图注意力机制
"""
Graph-to-Agent：所有节点信息 → 聚合到代理节点
Agent-to-Graph：代理节点信息 → 分发回所有节点
"""
class AAGA(nn.Module):
    """Adaptive Graph Agent Attention (AGAA) 自适应图代理注意力 """

    def __init__(self, dim):
        super(AAGA, self).__init__()
        self.dim = dim
        self.scale = dim**-0.5  # 注意力缩放因子
        self.qkv = nn.Linear(dim, dim * 3) # 生成QKV
        self.agent = nn.Linear(dim, dim * 2) # 代理节点QK
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = RMSNorm(d=dim)

    def forward(self, agent, x):
        # agent: (k, d) 代理节点
        # x: (b, n, d)批次,节点数,维度
        
        q, k, v = torch.chunk(self.qkv(x), chunks=3, dim=-1)
        q_agent, k_agent = torch.chunk(self.agent(agent), chunks=2, dim=-1)

        # Graph-to-Agent Attention 步骤1：节点信息 → 代理节点
        attn = torch.einsum("kd,bnd->bkn", (q_agent, k))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc1(v)

        # Agent-to-Graph Attention 代理节点信息 → 所有节点
        attn = torch.einsum("bnd,kd->bnk", (q, k_agent))
        attn = F.softmax(attn * self.scale, dim=-1)
        v = torch.matmul(attn, v)
        v = self.fc2(v)

        return self.norm(v + x) # 残差 + 归一化

# ---------------------------
# MLP：简单的两层全连接
# ---------------------------
class mlp(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(mlp, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),nn.Linear(in_dim, out_dim))

    def forward(self, x):
        # x:b,n,d
        return self.layers(x)

# ---------------------------
# 🔥 FaST 主模型
# ---------------------------
class FaST(nn.Module):
    def __init__(
        self,
        num_nodes,# 图节点数
        input_len=96,# 输入历史长度
        output_len=48, # 预测长度
        layers=3,# 堆叠层数
        num_experts=8,# 专家数量
        daily_steps=96,# 一天时间步（15分钟则96）
        # daily_steps=24, #这里是不是要改  这些json文件里面可以读取然后改吗
        weekly_days=7, # 一周7天
        hidden_dim=64,# 隐藏维度
        num_agent=32,# 代理节点数
    ):
        super(FaST, self).__init__()
        self.L = input_len
        self.layers = layers
        self.daily_steps = daily_steps
        self.weekly_days = weekly_days
        self.num_nodes = num_nodes

        # 创造节点索引（0~N-1）
        self.node_idx = to_device(torch.arange(self.num_nodes)).unsqueeze(0)  # (N,)->(1, N)
        # 创造输入层：路由 + MoE
        self.input_layer = nn.ModuleList([
            HARoutingLayer(input_len, num_experts, daily_steps, weekly_days, num_nodes),
            ParallelMoEWithGLU(input_len, hidden_dim, num_experts, num_nodes, res_flag=False)]
            )
        # 堆叠多层 AAGA + 路由 + MoE
        self.AAGA = nn.ModuleList()
        self.Router = nn.ModuleList()
        self.MoE = nn.ModuleList()
        for _ in range(layers):
            self.AAGA.append(AAGA(hidden_dim))
            self.Router.append(HARoutingLayer(input_len, num_experts, daily_steps, weekly_days, num_nodes))
            self.MoE.append(ParallelMoEWithGLU(hidden_dim, hidden_dim, num_experts, num_nodes))
        # 输出层：拼接所有层输出 → 预测未来
        self.output_layer = mlp(hidden_dim * (layers + 1), output_len)


        # adaptive agent 可学习代理节点
        self.agent = nn.Parameter(torch.empty([num_agent, hidden_dim]))
        nn.init.xavier_uniform_(self.agent)
        # time of day 三个可学习嵌入：时间、星期、节点
        self.tod_emb = TrainableParameterLayer([daily_steps, hidden_dim])
        # day of week
        self.dow_emb = TrainableParameterLayer([weekly_days, hidden_dim])
        # node embedding
        self.node_emb = TrainableParameterLayer([num_nodes, hidden_dim])

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            history_data (torch.Tensor): shape (b, l, n, 3)
            输入：history_data = (批次B, 输入长度L, 节点数N, 特征数3)
            - 0: data  交通流量（真正要预测的值）
            - 1: index for time of day 一天中第几个时间步（time of day）
            - 2: index for day of week 星期几（day of week）

        Returns:
            torch.Tensor: (b, p, n, 1)  未来交通预测值

        b = batch_size 批次大小
        l = input_len 输入历史长度（你设的 24）
        n = num_nodes 节点数（506）
        p = output_len 预测长度（你设的 8）
        3 = 3 个输入特征
        1 = 只预测交通流量

        """
        B, L, N, C = history_data.shape
        
        """步骤 1：输入数据解析 
           输入形状：(batch, L, N, 3)
           batch：批次
           L：历史长度（96）
           N：节点数（716）
           3：[真实值，时刻，星期]
        """
        # 取出真正的交通数据（第 0 维特征） # 取出流量数据，并转置 → (B, N, L)
        raw = history_data[:, :, :, 0].transpose(2, 1).contiguous()
        
        """
        步骤 2：实例归一化
        """
        # instance norm  对每个样本单独计算均值和方差，进行标准化 归一化
        seq_mean = torch.mean(raw, dim=-1, keepdim=True)
        seq_var = torch.var(raw, dim=-1, keepdim=True) + 1e-5
        raw = (raw - seq_mean) / torch.sqrt(seq_var)
        
        """
        步骤 3：提取时间 / 星期索引
        """
        # 取出时间、星期索引
        day_idx  = (history_data[:, -1, :, 1] * self.daily_steps).long()   # (B, N)
        week_idx = (history_data[:, -1, :, 2] * self.weekly_days).long()   # (B, N)
        
        """
        步骤 4：输入层 MoE
        """
        # 输入层路由 + MoE  压缩维度
        router = self.input_layer[0](raw, day_idx, week_idx, self.node_idx)
        x = self.input_layer[1](raw, router)
        
        """
        步骤 5：加入时空嵌入
        """
        # + time of day embedding # 加入时间/星期/节点嵌入
        x += self.tod_emb(day_idx).contiguous()
        # + day of week embedding
        x += self.dow_emb(week_idx).contiguous()
        # + node embedding
        x += self.node_emb(self.node_idx).contiguous()
        
        """
        步骤 6：堆叠 L 层核心块
        """
        # 跳跃连接（保存每一层输出）
        skip = [x]
        for i in range(self.layers):
            x = self.AAGA[i](self.agent, x) # 图注意力 捕捉空间依赖
            router = self.Router[i](raw, day_idx, week_idx, self.node_idx)  # 路由 每个节点重新选专家
            x,s = self.MoE[i](x, router) # 混合专家 8个专家并行计算 加权融合
            skip.append(s) # 存入跳跃连接
        """
        步骤 7：拼接所有层输出
        """
        # 拼接所有层输出
        x = torch.cat(skip, dim=-1)

        """
        步骤 8：预测 + 反归一化
        """
        x = self.output_layer(x) # b,n,p
        # instance denorm   进行反归一化，将预测值恢复到原始数据的数值范围。
        x = x * torch.sqrt(seq_var) + seq_mean
        """
        步骤 9：输出形状调整
        """
        # 调整形状输出：(B, P, N, 1)
        return x.unsqueeze(-1).transpose(2, 1).contiguous()  # prediction:[b, p, n, 1]


if __name__ == "__main__":
    # 创建模型 → 运行 FaST.__init__
    model = FaST(716, 96, 720)   # 节点数 输入历史长度 预测长度
    # 第 2 步：summary 开始模拟输入 → 运行 forward
    # 第 3 步：进入 forward 函数（真正开始计算！）
    summary(model, [64, 96, 716, 3]) # 打印模型结构 参数量 计算量
