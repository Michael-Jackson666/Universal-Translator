# MicroGPT

**~200行纯Python实现的完整GPT**，无任何深度学习框架依赖。

> "This file is the complete algorithm. Everything else is just efficiency." — @karpathy

## 文件说明

| 文件 | 行数 | 说明 |
|------|------|------|
| `microgpt.py` | ~200 | 原始简洁版，代码紧凑 |
| `microgpt-detail.py` | ~520 | 详细注释版，含数学推导和架构图 |

## 核心架构

```
┌─────────────────────────────────────────────────────────┐
│  Input: token_id, pos_id                                │
│           ↓                                             │
│  ┌─────────────────────┐                                │
│  │ Token Emb + Pos Emb │  x = wte[token] + wpe[pos]     │
│  └─────────────────────┘                                │
│           ↓                                             │
│  ╔═══════════════════════════════════════╗              │
│  ║     Multi-Head Self-Attention         ║              │
│  ║  Q, K, V 投影 → softmax(QK^T/√d) × V  ║              │
│  ╚═══════════════════════════════════════╝              │
│           ↓ (+残差)                                     │
│  ╔═══════════════════════════════════════╗              │
│  ║     MLP: Linear → ReLU → Linear       ║              │
│  ╚═══════════════════════════════════════╝              │
│           ↓ (+残差)                                     │
│  ┌─────────────────────┐                                │
│  │   LM Head → logits  │  [vocab_size]                  │
│  └─────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 自动微分引擎 (Autograd)

```python
class Value:
    # 每个运算记录：前向结果 + 子节点 + 局部梯度
    def __mul__(self, other):
        # z = x * y → dz/dx = y, dz/dy = x
        return Value(self.data * other.data, (self, other), (other.data, self.data))
```

**链式法则**：`d(loss)/d(x) = d(loss)/d(z) × d(z)/d(x)`

### 2. 注意力机制

```python
# Attention(Q, K, V) = softmax(QK^T / √d_k) × V
attn_logits = [sum(q[j] * k[t][j] for j in range(d)) / d**0.5 for t in range(T)]
attn_weights = softmax(attn_logits)  # 和为1的权重
output = weighted_sum(attn_weights, V)
```

**直觉**：
- Q (Query): "我在找什么？"
- K (Key): "我有什么信息？"  
- V (Value): "我的内容是什么？"

### 3. Adam 优化器

```python
m = β1 * m + (1-β1) * grad      # 动量（一阶矩）
v = β2 * v + (1-β2) * grad²     # 自适应学习率（二阶矩）
param -= lr * m̂ / (√v̂ + ε)      # 参数更新
```

## 超参数配置

```python
n_embd = 16      # 嵌入维度
n_head = 4       # 注意力头数
n_layer = 1      # Transformer 层数
block_size = 16  # 上下文窗口长度
num_steps = 1000 # 训练步数
```

**参数量计算**：
- Token Embedding: vocab_size × n_embd
- Position Embedding: block_size × n_embd
- Attention: 4 × n_embd × n_embd (Q, K, V, O)
- MLP: n_embd × 4n_embd + 4n_embd × n_embd
- LM Head: vocab_size × n_embd

## 运行示例

```bash
python microgpt.py
```

**输出**：
```
num docs: 32033
vocab size: 27
num params: 7883
step    1 / 1000 | loss 3.2958
step  100 / 1000 | loss 2.4521
step  500 / 1000 | loss 2.1893
step 1000 / 1000 | loss 1.9234

--- inference (new, hallucinated names) ---
sample  1: emma
sample  2: olivia
sample  3: jayden
...
```

## 与 GPT-2 的区别

| 特性 | MicroGPT | GPT-2 |
|------|----------|-------|
| 归一化 | RMSNorm | LayerNorm |
| 激活函数 | ReLU | GeLU |
| 偏置项 | 无 | 有 |
| 参数量 | ~8K | 117M-1.5B |
| 框架 | 纯 Python | PyTorch |

## 学习要点

1. **Autograd 是深度学习的核心** - 理解 `Value` 类如何实现反向传播
2. **注意力机制的本质** - QKV 的几何意义和 softmax 归一化
3. **残差连接的重要性** - 让梯度可以直接流回，防止梯度消失
4. **Temperature 采样** - 控制生成的"创造性"程度

## 扩展阅读

- [microgpt-detail.py](microgpt-detail.py) - 带完整数学推导的注释版本
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
