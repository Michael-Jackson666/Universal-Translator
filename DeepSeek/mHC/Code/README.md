# mHC 代码实现

基于论文 [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) 的 PyTorch 实现。

## 核心思想

mHC 通过将 Hyper-Connections 的残差矩阵投影到 **Birkhoff 多面体（双随机矩阵流形）** 上，解决了 HC 架构的数值不稳定问题：

- **问题**：HC 的 $\mathcal{H}^{\text{res}}$ 矩阵无约束，连乘后信号爆炸（增益达 3000）
- **解决**：双随机矩阵的谱范数 ≤ 1，保证非扩张映射，信号守恒

## 模块结构

```
Code/
├── config.py      # 配置类 mHCConfig
├── manifold.py    # Birkhoff 多面体投影 + Sinkhorn-Knopp 算法
├── layers.py      # mHC 层实现 (动态映射 + 流形投影)
├── model.py       # 完整模型 (mHCModel)
├── mhc.py         # 主入口和测试
└── README.md      # 说明文档
```

## 核心组件

### 1. Sinkhorn-Knopp 算法 (`manifold.py`)

将任意正矩阵投影到双随机矩阵：

```python
M^(t) = T_r(T_c(M^(t-1)))
```

其中 $T_c$ 为列归一化，$T_r$ 为行归一化，迭代 20 次收敛。

### 2. 流形投影 (Eq. 8)

```python
H_pre = sigmoid(H_tilde_pre)      # 非负
H_post = 2 * sigmoid(H_tilde_post) # 恢复尺度
H_res = Sinkhorn-Knopp(H_tilde_res) # 双随机
```

### 3. mHC 层 (Eq. 3)

```python
x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
```

### 4. 动态重计算 (Eq. 20)

最优 checkpoint 间隔：

$$L_r^* = \sqrt{\frac{nL}{n+2}}$$

## 使用方法

```python
import torch
from config import create_small_config
from model import mHCModel

# 创建模型
config = create_small_config()
model = mHCModel(config)

# 前向传播
x = torch.randn(2, config.hidden_size)  # [batch, C]
y = model(x)  # [batch, C]

# 使用动态重计算节省显存
y = model(x, use_recompute=True)
```

## 运行测试

```bash
cd Code
python mhc.py
```

## 关键数学性质

| 性质 | 说明 |
|------|------|
| 凸组合 | $\mathcal{H}^{\text{res}}\mathbf{x}$ 是加权平均，信号均值守恒 |
| 范数有界 | 双随机矩阵谱范数 ≤ 1，映射非扩张 |
| 组合封闭 | 双随机矩阵乘积仍是双随机矩阵 |
| 恒等恢复 | 连乘保持稳定，类似 ResNet |

## 实验结果 (论文)

- 信号增益：HC 3000 → mHC **1.6**
- BBH 任务：Baseline 43.8 → HC 48.9 → **mHC 51.0**
- 训练开销：仅增加 **6.7%**
