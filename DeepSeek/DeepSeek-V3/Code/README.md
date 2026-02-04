# DeepSeek-V3 代码实现

基于论文 [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) 的 PyTorch 实现。

## 模块结构

```
Code/
├── config.py          # 配置类 DeepSeekV3Config
├── layers.py          # 基础组件 (RMSNorm, RoPE, ExpertFFN)
├── mla.py             # Multi-Head Latent Attention
├── moe.py             # DeepSeek MoE 层 (共享专家 + 路由专家)
├── mtp.py             # Multi-Token Prediction
├── block.py           # Transformer Block (MLA + MoE)
├── model.py           # 完整模型 (DeepSeekV3Model, DeepSeekV3ForCausalLM)
└── deepseek_v3.py     # 主入口和测试
```

## 核心组件

### 1. Multi-Head Latent Attention (MLA)
- KV 低秩联合压缩：将 KV Cache 压缩到低维潜变量
- 解耦 RoPE：位置编码单独处理
- Query 压缩：减少训练时激活显存

### 2. DeepSeek MoE
- 共享专家 (Shared Experts)：所有 token 都经过
- 路由专家 (Routed Experts)：Top-K 选择
- 无辅助损失负载均衡：Bias 仅影响路由选择，不影响权重

### 3. Multi-Token Prediction (MTP)
- 预测未来 D 个 token
- 训练时提升数据效率
- 推理时可用于投机解码

## 使用方法

```python
import torch
from config import create_small_config
from model import DeepSeekV3ForCausalLM

# 创建小型测试模型
config = create_small_config()
model = DeepSeekV3ForCausalLM(config)

# 前向传播
input_ids = torch.randint(0, 1000, (2, 16))
loss, logits, mtp_logits = model(input_ids, labels=input_ids)

# 生成
generated = model.generate(input_ids[:1, :8], max_new_tokens=10)
```

## 运行测试

```bash
cd Code
python deepseek_v3.py
```

## 模型规模

| 配置 | 值 |
|------|-----|
| 总参数量 | 671B |
| 激活参数量 | 37B (per token) |
| 层数 | 61 |
| 隐藏维度 | 7168 |
| 注意力头数 | 128 |
| 路由专家数 | 256 |
| 激活专家数 | 8 |
