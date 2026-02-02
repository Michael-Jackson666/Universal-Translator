# ZeRO-Infinity 学习笔记

微软 DeepSpeed 团队的 **ZeRO-Infinity** 是 DeepSeek Engram 等现代"计算存储分离"架构的鼻祖级工作。

## 📄 论文信息

- **论文**: [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://dl.acm.org/doi/10.1145/3458817.3476205)
- **会议**: SC'21 (Supercomputing 2021)
- **团队**: Microsoft DeepSpeed

## 🎯 核心问题

随着模型规模呈指数级增长，GPU 显存（HBM）的增长速度远远跟不上：

| 年份 | 模型 | 参数量 | 显存需求 (训练) |
|:---:|:---|:---:|:---:|
| 2019 | GPT-2 | 1.5B | ~10 GB |
| 2020 | GPT-3 | 175B | ~2.8 TB |
| 2021 | 目标 | 1T | **16 TB** |

**传统方案的困境**：
- 3D 并行需要数百张 GPU + 高速互联
- 代码需要大量修改
- 成本高昂，只有大公司能负担

## 💡 核心创新

### 五大关键技术

| 技术 | 解决问题 | 核心思想 |
|:---|:---|:---|
| **带宽中心化切分** | PCIe 带宽瓶颈 | N 卡并行加载 → 带宽 ×N |
| **无限卸载引擎** | NVMe 管理复杂 | DeepNVMe 异步分层存储 |
| **显存中心化分块** | 单层参数超显存 | Tiling 切分大矩阵 |
| **重叠中心化设计** | IO 延迟 | 三级流水线预取 |
| **通信重叠优化** | 梯度同步开销 | 计算与通信并行 |

### 核心公式

**有效带宽公式**：
$$
\text{Effective Bandwidth} = N \times \text{Single PCIe Bandwidth}
$$

**训练效率公式**：
$$
\text{efficiency} = \frac{\text{ait} \times bw}{\text{ait} \times bw + \text{peak}_{tp}}
$$

## 📊 实验结果

| 配置 | 模型规模 | 意义 |
|:---|:---:|:---|
| 16 V100 (1节点) | **1T 参数** | 以前需要几百张卡 |
| 512 V100 (32节点) | **32T 参数** | 当时世界纪录 50 倍 |

**效率**：达到 GPU 峰值算力的 **40%+**

## 🔗 与 Engram 的关系

ZeRO-Infinity 证明了 PCIe/NVMe 卸载是可行的，为 Engram 的设计提供了理论基础：

| 对比 | ZeRO-Infinity | Engram |
|:---|:---|:---|
| 定位 | 通用传输管道 | 外挂大脑 |
| 卸载对象 | 模型权重 (动态) | N-gram 表 (静态) |
| 预取方式 | 按层顺序 | Token 确定后立即 |

> **一句话理解**：ZeRO-Infinity 造就了"能把大象装进冰箱"的**通用传输管道**，而 Engram 则是设计了一种**特殊的"外挂大脑"**

## 📁 文件结构

```
Zero-Infinity/
├── README.md              # 本文件
└── Zero-Infinity.md       # 📝 完整学习笔记
    ├── 0. 前置知识回顾
    │   ├── ZeRO 系列演进
    │   ├── 3D 并行技术详解
    │   └── 硬件带宽层级
    ├── 1. 显存与带宽账单
    │   ├── 模型状态详解 (16TB 怎么来的)
    │   ├── Activation Checkpointing
    │   ├── 为什么最大算子是 h→4h
    │   ├── MSWM 瓶颈详解
    │   └── 效率公式推导
    ├── 2. 核心架构 (5大创新)
    ├── 3. 性能分析与实验
    ├── 4. 与 Engram 深度对比
    └── 5. 总结与启示
```

## 🔧 相关概念索引

| 概念 | 笔记位置 | 简介 |
|:---|:---|:---|
| **Activation Checkpointing** | §1.2 | 用计算换显存，保存检查点重计算 |
| **h → 4h** | §1.3 | FFN 扩展层是最大参数矩阵 |
| **MSWM** | §1.4 | 单算子执行的最小显存需求 |
| **算术强度** | §1.5 | FLOPs/Byte，决定能否掩盖 IO |
| **带宽中心化切分** | §2.1 | 核心创新，N 卡并行 IO |
| **Memory-Centric Tiling** | §2.3 | 分块突破显存限制 |

## 📚 参考资料

1. [ZeRO-Infinity 论文](https://dl.acm.org/doi/10.1145/3458817.3476205)
2. [ZeRO 原始论文](https://arxiv.org/abs/1910.02054)
3. [DeepSpeed 官方文档](https://www.deepspeed.ai/tutorials/zero-offload/)
4. [Engram 论文](https://arxiv.org/abs/2601.07372)
