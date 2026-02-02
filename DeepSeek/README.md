# DeepSeek 论文精读与复现

本目录包含 DeepSeek 系列论文的学习笔记和代码实现。

## 📁 目录结构

```
DeepSeek/
├── DeepSeek-MoE/              # DeepSeekMoE 混合专家模型
│   ├── README.md              # 模块概述
│   ├── MoE简介.md             # 📝 MoE 基础知识
│   ├── DeepSeek-MoE.md        # 📝 DeepSeekMoE 详细笔记
│   ├── DeepSeekMoE.png        # 🖼️ 架构图
│   ├── MoE Layer.png          # 🖼️ MoE 层示意图
│   └── Code/                  # 💻 PyTorch 实现
│       ├── experts.py         # 专家网络 (SwiGLU FFN)
│       ├── router.py          # Top-K 路由与负载均衡
│       ├── moe_layer.py       # MoE 层 (共享+路由专家)
│       └── deepseek_moe.py    # 完整模型实现
│
├── Engram/                    # Engram 条件记忆架构
│   ├── README.md              # 模块概述
│   ├── Engram.md              # 📝 Engram 详细笔记
│   ├── Engram.png             # 🖼️ 架构图
│   ├── Sparsity allocation and Engram scaling.png
│   └── Code/                  # 💻 PyTorch 实现
│       ├── tokenizer_compression.py  # Token 压缩与 N-gram 提取
│       ├── multi_head_hashing.py     # 多头哈希与 Embedding 查找
│       ├── context_aware_gating.py   # 上下文感知门控
│       ├── fusion.py                 # 深度卷积融合层
│       └── engram.py                 # 完整 Engram 模块
│
├── Zero-Infinity/             # ZeRO-Infinity 异构系统技术 (Engram 鼻祖)
│   ├── README.md              # 模块概述
│   └── Zero-Infinity.md       # 📝 完整学习笔记
│       ├── 前置知识 (ZeRO系列/3D并行/硬件带宽)
│       ├── Activation Checkpointing 详解
│       ├── h→4h 最大算子分析
│       ├── MSWM 瓶颈详解
│       ├── 5大核心架构创新
│       └── 与 Engram 深度对比
│
├── mHC/                       # mHC 多头因果架构
│   ├── mHC.md                 # 📝 mHC 学习笔记
│   ├── mHC.png                # 🖼️ 架构图
│   └── Communication-Computation Overlapping for mHC.png
│
└── DeepThink.md               # 📝 深度思考笔记
```

## 📚 已完成内容

### 1. DeepSeekMoE - 极致专家特化

**论文**: [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)

| 内容 | 状态 |
|------|------|
| MoE 基础知识笔记 | ✅ 完成 |
| DeepSeekMoE 原理笔记 | ✅ 完成 |
| 代码实现 (前向传播) | ✅ 完成 |

**核心创新**:
- 细粒度专家分割 (1/m 大小)
- 共享专家隔离 (始终激活)
- Top-K 路由与负载均衡

### 2. Engram - 条件记忆

**论文**: [Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)

| 内容 | 状态 |
|------|------|
| Engram 原理笔记 | ✅ 完成 |
| 代码实现 (前向传播) | ✅ 完成 |

**核心创新**:
- 条件记忆 (与 MoE 条件计算互补)
- N-gram 多头哈希检索
- 上下文感知门控
- 零开销预取机制

### 3. mHC - 多头因果架构

**论文**: DeepSeek-V3 系列

| 内容 | 状态 |
|------|------|
| mHC 学习笔记 | ✅ 完成 |
| 代码实现 | 📋 计划中 |

### 4. ZeRO-Infinity - 异构系统技术 (Engram 鼻祖)

**论文**: [ZeRO-Infinity: Breaking the GPU Memory Wall](https://dl.acm.org/doi/10.1145/3458817.3476205) (Microsoft, SC'21)

| 内容 | 状态 |
|------|------|
| 完整学习笔记 | ✅ 完成 |
| 前置知识详解 | ✅ 完成 |
| 与 Engram 对比分析 | ✅ 完成 |

**核心贡献**:
- 带宽中心化切分 (N 卡并行 IO)
- 无限卸载引擎 (DeepNVMe)
- 显存中心化分块 (Memory-Centric Tiling)
- 重叠中心化设计 (三级流水线预取)
- 训练民主化 (单节点微调万亿模型)

**笔记亮点**:
- Activation Checkpointing 技术详解
- 为什么 Transformer 最大算子是 h→4h
- MSWM 瓶颈的来源与解决
- 算术强度与效率公式推导
- ZeRO-Infinity 与 Engram 的技术继承关系

## 🚀 快速开始

```bash
cd DeepSeek

# 运行 DeepSeekMoE 示例
python DeepSeek-MoE/Code/deepseek_moe.py

# 运行 Engram 示例
python Engram/Code/engram.py
```

## 📄 License

MIT License
