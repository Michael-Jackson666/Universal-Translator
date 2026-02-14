"""
================================================================================
                    MicroGPT: 最小化纯Python实现的GPT
================================================================================
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.
@karpathy

核心组件：
  1. Tokenizer - 文本与数字的转换
  2. Autograd - 自动微分引擎（反向传播的基础）
  3. Transformer - 注意力机制 + 前馈网络
  4. Adam优化器 - 参数更新算法
  5. 训练循环 - 前向传播 → 计算损失 → 反向传播 → 更新参数
  6. 推理 - 自回归文本生成

这200行代码包含了ChatGPT的所有核心算法！
其他的都是工程优化：GPU并行、分布式、量化、FlashAttention等
"""

# ============================================================================
# 第0部分：导入标准库（无任何深度学习依赖！）
# ============================================================================
import os       # os.path.exists - 文件操作
import math     # math.log, math.exp - 数学运算（用于softmax、损失函数等）
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # 设置随机种子，确保实验可复现

# ============================================================================
# 第1部分：数据集加载
# ============================================================================
# 数据集：人名列表，每个名字是一个"文档"，模型学习生成类似的名字
# 这是一个字符级语言模型的经典任务
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
# 读取所有名字，每行一个名字
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)  # 打乱顺序，避免训练时的顺序偏差
print(f"num docs: {len(docs)}")

# ============================================================================
# 第2部分：Tokenizer（分词器）
# ============================================================================
# 字符级分词：每个唯一字符映射到一个整数ID
# 例如：'a'->0, 'b'->1, ..., 'z'->25
uchars = sorted(set(''.join(docs)))  # 收集所有唯一字符并排序
BOS = len(uchars)  # BOS (Beginning Of Sequence) 特殊标记，用于标记序列的开始和结束
vocab_size = len(uchars) + 1  # 词汇表大小 = 字符数 + 1个BOS标记
print(f"vocab size: {vocab_size}")
# 编码示例："hello" -> [BOS, h_id, e_id, l_id, l_id, o_id, BOS]
# 解码示例：[7, 4, 11, 11, 14] -> "hello"

# ============================================================================
# 第3部分：自动微分引擎 (Autograd)
# ============================================================================
# 这是深度学习的核心！通过计算图自动计算梯度
# 原理：链式法则 (Chain Rule)
#   如果 z = f(y), y = g(x)，则 dz/dx = dz/dy * dy/dx
#
# 计算图示例：
#   x = 2, y = 3
#   z = x * y = 6
#   loss = z + 1 = 7
#
#   反向传播：
#   d(loss)/d(loss) = 1
#   d(loss)/dz = 1 (因为 loss = z + 1)
#   d(loss)/dx = d(loss)/dz * dz/dx = 1 * y = 3
#   d(loss)/dy = d(loss)/dz * dz/dy = 1 * x = 2

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # 内存优化

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向传播时计算的标量值
        self.grad = 0                   # 反向传播时计算的梯度：d(loss)/d(self)
        self._children = children       # 计算图中的子节点（输入）
        self._local_grads = local_grads # 局部梯度：d(self)/d(child) 对每个子节点

    # ======================== 运算符重载与局部梯度 ========================
    # 每个运算都记录：(1) 前向结果 (2) 子节点 (3) 局部梯度
    
    def __add__(self, other):
        # 加法：z = x + y
        # 局部梯度：dz/dx = 1, dz/dy = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # 乘法：z = x * y
        # 局部梯度：dz/dx = y, dz/dy = x（交叉相乘）
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # 幂运算：z = x^n
        # 局部梯度：dz/dx = n * x^(n-1)（幂函数求导法则）
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    
    def log(self):
        # 对数：z = log(x)
        # 局部梯度：dz/dx = 1/x
        return Value(math.log(self.data), (self,), (1/self.data,))
    
    def exp(self):
        # 指数：z = e^x
        # 局部梯度：dz/dx = e^x（指数函数的导数等于自身）
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def relu(self):
        # ReLU激活：z = max(0, x)
        # 局部梯度：dz/dx = 1 if x > 0 else 0（分段函数）
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    # ======================== 辅助运算符 ========================
    def __neg__(self): return self * -1              # 取负：-x = x * (-1)
    def __radd__(self, other): return self + other   # 右加：other + self
    def __sub__(self, other): return self + (-other) # 减法：x - y = x + (-y)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other   # 右乘：other * self
    def __truediv__(self, other): return self * other**-1  # 除法：x/y = x * y^(-1)
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        反向传播：从loss节点开始，逆序遍历计算图，计算每个节点的梯度
        
        算法步骤：
        1. 拓扑排序：确保在计算某节点梯度前，所有依赖它的节点已处理
        2. 设置loss节点梯度为1（d(loss)/d(loss) = 1）
        3. 逆序遍历，应用链式法则：child.grad += local_grad * parent.grad
        
        可视化示例：
            a → [*] → c → [+] → loss
            b ↗           d ↗
        
        反向传播顺序：loss → (+) → c, d → (*) → a, b
        """
        # 步骤1：拓扑排序（后序遍历）
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # 步骤2&3：反向传播
        self.grad = 1  # d(loss)/d(loss) = 1
        for v in reversed(topo):  # 从loss向输入方向遍历
            for child, local_grad in zip(v._children, v._local_grads):
                # 链式法则：d(loss)/d(child) += d(loss)/d(v) * d(v)/d(child)
                child.grad += local_grad * v.grad

# ============================================================================
# 第4部分：模型参数初始化
# ============================================================================
# 超参数定义
n_embd = 16      # 嵌入维度：每个token用16维向量表示
n_head = 4       # 注意力头数：多头注意力的并行数
n_layer = 1      # Transformer层数
block_size = 16  # 最大序列长度（上下文窗口）
head_dim = n_embd // n_head  # 每个注意力头的维度 = 16/4 = 4

# 参数初始化函数：创建高斯随机初始化的权重矩阵
# nout x nin 的矩阵，每个元素是Value对象（支持自动微分）
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 模型权重字典
state_dict = {
    # ===== 嵌入层 =====
    'wte': matrix(vocab_size, n_embd),  # Token Embedding: [vocab_size, n_embd]
                                         # 将每个token ID映射到n_embd维向量
    'wpe': matrix(block_size, n_embd),  # Position Embedding: [block_size, n_embd]
                                         # 将每个位置映射到n_embd维向量
    # ===== 输出层 =====
    'lm_head': matrix(vocab_size, n_embd),  # Language Model Head: [vocab_size, n_embd]
                                             # 将隐藏状态映射回词汇表大小的logits
}

# 为每一层添加Transformer块的参数
for i in range(n_layer):
    # ===== 多头注意力参数 =====
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query投影: x -> Q
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key投影: x -> K  
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value投影: x -> V
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output投影: concat(heads) -> x
    
    # ===== MLP前馈网络参数 =====
    # MLP结构：x -> fc1 -> ReLU -> fc2 -> x
    # 中间维度是4倍扩展，这是GPT的标准设计
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 升维: [n_embd] -> [4*n_embd]
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 降维: [4*n_embd] -> [n_embd]

# 将所有参数展平为一维列表，便于优化器更新
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ============================================================================
# 第5部分：模型架构组件
# ============================================================================
# 相比GPT-2的简化：LayerNorm -> RMSNorm, GeLU -> ReLU, 无偏置项

def linear(x, w):
    """
    线性变换（矩阵乘法）：y = x @ W^T
    
    数学原理：
        输入 x: [n_in] 维向量
        权重 w: [n_out, n_in] 矩阵
        输出 y: [n_out] 维向量
        
        y[i] = sum(w[i][j] * x[j] for j in range(n_in))
    
    示例：x=[1,2,3], w=[[1,0,0],[0,1,0]] -> y=[1,2]
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    """
    Softmax函数：将logits转换为概率分布
    
    数学公式：
        softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    
    数值稳定性技巧：
        减去最大值防止exp溢出：exp(x_i - max) / sum(exp(x_j - max))
        这不改变结果，因为 exp(a-c)/exp(b-c) = exp(a)/exp(b)
    
    示例：[2.0, 1.0, 0.1] -> [0.659, 0.242, 0.099]（和为1）
    """
    max_val = max(val.data for val in logits)  # 找最大值
    exps = [(val - max_val).exp() for val in logits]  # 减去最大值后取指数
    total = sum(exps)  # 求和作为分母
    return [e / total for e in exps]  # 归一化得到概率

def rmsnorm(x):
    """
    RMSNorm（Root Mean Square Normalization）
    
    数学公式：
        RMSNorm(x) = x / sqrt(mean(x^2) + eps)
    
    与LayerNorm的区别：
        - LayerNorm: (x - mean) / std，需要计算均值和方差
        - RMSNorm: x / rms，只需计算均方根，更简单高效
    
    作用：稳定训练，防止梯度爆炸/消失
    
    示例：[3, 4] -> rms = sqrt((9+16)/2) = 3.54 -> [0.85, 1.13]
    """
    ms = sum(xi * xi for xi in x) / len(x)  # 计算平方均值 (mean square)
    scale = (ms + 1e-5) ** -0.5  # 1/sqrt(ms + eps)，eps防止除零
    return [xi * scale for xi in x]  # 缩放每个元素

def gpt(token_id, pos_id, keys, values):
    """
    GPT前向传播：给定当前token和位置，预测下一个token的概率分布
    
    架构概览（单个Transformer块）：
    ┌─────────────────────────────────────────────────────┐
    │  Input: token_id, pos_id                            │
    │           ↓                                         │
    │  ┌─────────────────────┐                            │
    │  │ Token Emb + Pos Emb │  x = wte[token] + wpe[pos] │
    │  └─────────────────────┘                            │
    │           ↓                                         │
    │  ┌─────────────────────┐                            │
    │  │      RMSNorm        │                            │
    │  └─────────────────────┘                            │
    │           ↓                                         │
    │  ╔═══════════════════════════════════════╗          │
    │  ║     Multi-Head Self-Attention         ║          │
    │  ║  ┌───┐ ┌───┐ ┌───┐                    ║          │
    │  ║  │ Q │ │ K │ │ V │  线性投影           ║          │
    │  ║  └───┘ └───┘ └───┘                    ║          │
    │  ║     ↓     ↓     ↓                     ║          │
    │  ║  Attention = softmax(QK^T/√d) × V     ║          │
    │  ╚═══════════════════════════════════════╝          │
    │           ↓ (+残差连接)                              │
    │  ╔═══════════════════════════════════════╗          │
    │  ║           MLP (Feed Forward)          ║          │
    │  ║     x → Linear → ReLU → Linear        ║          │
    │  ╚═══════════════════════════════════════╝          │
    │           ↓ (+残差连接)                              │
    │  ┌─────────────────────┐                            │
    │  │   LM Head (Linear)  │  → logits [vocab_size]     │
    │  └─────────────────────┘                            │
    └─────────────────────────────────────────────────────┘
    
    参数：
        token_id: 当前token的ID
        pos_id: 当前位置索引
        keys, values: KV缓存，用于自回归生成时避免重复计算
    
    返回：
        logits: [vocab_size] 维向量，表示下一个token的未归一化概率
    """
    # ==================== 嵌入层 ====================
    # Token嵌入：将离散token ID转为连续向量
    tok_emb = state_dict['wte'][token_id]  # 查表得到 [n_embd] 维向量
    # 位置嵌入：编码token在序列中的位置信息
    pos_emb = state_dict['wpe'][pos_id]    # 查表得到 [n_embd] 维向量
    # 相加得到最终嵌入（GPT-2风格，也可以用拼接）
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)  # 归一化稳定训练

    # ==================== Transformer层 ====================
    for li in range(n_layer):
        # ========== 1) 多头自注意力块 ==========
        x_residual = x  # 保存残差
        x = rmsnorm(x)  # Pre-Norm（在注意力之前归一化）
        
        # 计算Q, K, V投影
        # Q (Query): "我在找什么？"
        # K (Key): "我有什么信息？"
        # V (Value): "我的内容是什么？"
        q = linear(x, state_dict[f'layer{li}.attn_wq'])  # [n_embd] -> [n_embd]
        k = linear(x, state_dict[f'layer{li}.attn_wk'])  # [n_embd] -> [n_embd]
        v = linear(x, state_dict[f'layer{li}.attn_wv'])  # [n_embd] -> [n_embd]
        
        # KV缓存：存储历史的K和V，用于自回归生成
        keys[li].append(k)
        values[li].append(v)
        
        # 多头注意力计算
        x_attn = []
        for h in range(n_head):  # 对每个注意力头
            hs = h * head_dim  # 当前头的起始索引
            
            # 分割出当前头的Q, K, V
            q_h = q[hs:hs+head_dim]                        # [head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # [seq_len, head_dim]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]# [seq_len, head_dim]
            
            # 计算注意力分数：Attention(Q,K,V) = softmax(QK^T / √d_k) V
            # attn_logits[t] = (q_h · k_h[t]) / √head_dim
            # 除以√d_k是为了防止点积过大导致softmax饱和
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            
            # Softmax得到注意力权重（和为1）
            attn_weights = softmax(attn_logits)  # [seq_len]，表示对每个位置的关注程度
            
            # 加权求和Value：head_out = sum(attn_weights[t] * v_h[t])
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)  # 拼接所有头的输出
        
        # 输出投影：将拼接的多头输出映射回原始维度
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        # 残差连接：防止梯度消失，让梯度可以直接流回
        x = [a + b for a, b in zip(x, x_residual)]
        
        # ========== 2) MLP前馈网络块 ==========
        x_residual = x  # 保存残差
        x = rmsnorm(x)  # Pre-Norm
        # 两层MLP：先升维到4倍，再降回原维度
        # 这种"瓶颈"结构增加了模型的表达能力
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # [n_embd] -> [4*n_embd]
        x = [xi.relu() for xi in x]                       # ReLU激活引入非线性
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # [4*n_embd] -> [n_embd]
        # 残差连接
        x = [a + b for a, b in zip(x, x_residual)]

    # ==================== 输出层 ====================
    # LM Head：将隐藏状态映射到词汇表大小的logits
    logits = linear(x, state_dict['lm_head'])  # [n_embd] -> [vocab_size]
    return logits

# ============================================================================
# 第6部分：Adam优化器
# ============================================================================
# Adam = Adaptive Moment Estimation，结合了动量和自适应学习率
#
# 核心思想：
#   1. 动量 (Momentum): 累积历史梯度方向，平滑更新
#   2. 自适应学习率: 根据参数的历史梯度大小调整学习率
#
# 更新公式：
#   m_t = β1 * m_{t-1} + (1-β1) * g_t          # 一阶矩（梯度的指数移动平均）
#   v_t = β2 * v_{t-1} + (1-β2) * g_t^2        # 二阶矩（梯度平方的指数移动平均）
#   m̂_t = m_t / (1 - β1^t)                     # 偏差修正（初期m接近0）
#   v̂_t = v_t / (1 - β2^t)                     # 偏差修正
#   θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)    # 参数更新
#
learning_rate = 0.01   # 学习率
beta1 = 0.85           # 一阶矩衰减率（动量系数）
beta2 = 0.99           # 二阶矩衰减率（自适应学习率系数）
eps_adam = 1e-8        # 防止除零的小常数
m = [0.0] * len(params)  # 一阶矩缓冲（梯度的移动平均）
v = [0.0] * len(params)  # 二阶矩缓冲（梯度平方的移动平均）

# ============================================================================
# 第7部分：训练循环
# ============================================================================
# 语言模型训练目标：给定前缀，预测下一个token
# 示例：输入 "hell"，目标输出 "o"
#
# 损失函数：交叉熵损失 (Cross-Entropy Loss)
#   loss = -log(P(target_token))
#   直观理解：如果模型预测正确token的概率高，loss就低
#
num_steps = 1000  # 训练步数
for step in range(num_steps):
    
    # ==================== 数据准备 ====================
    # 取一个文档（名字），添加BOS标记
    # 例如："emma" -> [BOS, 'e', 'm', 'm', 'a', BOS]
    # BOS既是开始标记，也是结束标记
    doc = docs[step % len(docs)]  # 循环使用数据集
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)  # 序列长度不超过block_size
    
    # ==================== 前向传播 ====================
    # 构建计算图，计算每个位置的损失
    # 自回归训练：每次用前面的token预测下一个
    #
    # 例如 "emma"：
    #   位置0: 输入BOS, 预测'e'   -> loss_0 = -log(P('e'))
    #   位置1: 输入'e', 预测'm'   -> loss_1 = -log(P('m'))
    #   位置2: 输入'm', 预测'm'   -> loss_2 = -log(P('m'))
    #   位置3: 输入'm', 预测'a'   -> loss_3 = -log(P('a'))
    #   位置4: 输入'a', 预测BOS   -> loss_4 = -log(P(BOS))
    #
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # KV缓存
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 输入和目标
        logits = gpt(token_id, pos_id, keys, values)  # 前向传播得到logits
        probs = softmax(logits)  # 转换为概率分布
        loss_t = -probs[target_id].log()  # 交叉熵：-log(P(target))
        losses.append(loss_t)
    
    # 平均损失（所有位置的loss取平均）
    loss = (1 / n) * sum(losses)
    
    # ==================== 反向传播 ====================
    # 计算所有参数的梯度：d(loss)/d(param)
    loss.backward()
    
    # ==================== 参数更新 (Adam) ====================
    lr_t = learning_rate * (1 - step / num_steps)  # 线性学习率衰减
    for i, p in enumerate(params):
        # 更新一阶矩（动量）
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # 更新二阶矩（自适应学习率）
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 偏差修正（消除初期偏向零的bias）
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # 参数更新：θ = θ - lr * m̂ / (√v̂ + ε)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        # 清零梯度，为下一轮准备
        p.grad = 0
    
    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ============================================================================
# 第8部分：推理（文本生成）
# ============================================================================
# 自回归生成：从BOS开始，每次预测下一个token，直到生成BOS（结束）
#
# Temperature采样：控制生成的"创造性"
#   - 低temperature (如0.1): 更确定性，选择概率最高的token
#   - 高temperature (如1.0): 更随机，给低概率token更多机会
#   - 数学原理: softmax(logits/T)，T越小分布越尖锐
#
#   示例：logits = [2.0, 1.0, 0.5]
#   T=1.0: probs = [0.51, 0.27, 0.22]（较均匀）
#   T=0.5: probs = [0.67, 0.24, 0.09]（更集中在最大值）
#   T=0.1: probs = [0.99, 0.01, 0.00]（几乎确定性）
#
temperature = 0.5  # 采样温度，范围(0, 1]，越低越确定

print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # 初始化KV缓存
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # 从BOS开始生成
    sample = []
    
    for pos_id in range(block_size):
        # 前向传播得到下一个token的概率分布
        logits = gpt(token_id, pos_id, keys, values)
        
        # Temperature采样：logits除以temperature后做softmax
        probs = softmax([l / temperature for l in logits])
        
        # 按概率采样下一个token（而非贪婪选择最大概率）
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # 如果生成BOS，表示序列结束
        if token_id == BOS:
            break
        
        # 将token ID转回字符
        sample.append(uchars[token_id])
    
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

# ============================================================================
# 总结：这就是GPT的全部核心！
# ============================================================================
# 1. Tokenizer: 文本 <-> 数字
# 2. Autograd: 自动计算梯度（链式法则）
# 3. Transformer: 注意力机制 + MLP
# 4. 训练: 前向传播 -> 计算损失 -> 反向传播 -> 更新参数
# 5. 推理: 自回归生成，每次预测下一个token
#
# 这200行代码包含了ChatGPT的所有核心算法！
# 剩下的都是工程优化：GPU并行、分布式、量化、FlashAttention等
# ============================================================================