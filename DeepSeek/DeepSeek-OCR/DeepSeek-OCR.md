这份笔记严格遵循您要求的逻辑结构，对 DeepSeek 最近发布的两篇关于 OCR 与视觉语言模型（VLM）核心编码器的论文进行了穷尽详细的梳理。

这两篇论文展示了 DeepSeek 在多模态领域的演进路径：从 **“验证视觉作为文本压缩介质的可行性”** (OCR 1.0) 到 **“重构视觉编码的因果逻辑”** (OCR 2.0)。

---

# 第一篇：DeepSeek-OCR: Contexts Optical Compression
**论文核心**：探索“视觉模态”是否可以作为文本信息的高效压缩介质。
**核心假设**：一张包含文档图像的图片，所需的 Visual Tokens 数量可能远少于其对应的文本 Tokens，从而实现长上下文的“光学压缩”。

## 1. 核心理念：光学压缩 (Optical Compression)
当前的 LLM 处理长文本时计算量呈二次方增长。DeepSeek-OCR 提出了一种激进的思路：**将长文本渲染成图像，再通过视觉编码器压缩成少量的 Visual Tokens 喂给 LLM。**
*   **压缩比 (Compression Ratio)**：定义为 `Ground Truth Text Tokens / Vision Tokens`。
*   **发现**：当压缩比在 **10x** 以内时，模型能以 97% 的精度还原文本；即使压缩比达到 **20x**，仍保留 60% 的精度。这意味着视觉是极佳的信息压缩容器。

## 2. 模型架构：DeepEncoder + MoE Decoder

### 2.1 视觉编码器：DeepEncoder (混合架构)
为了在高分辨率下保持低显存占用（Low Activation）并实现高压缩率，作者设计了一个串行混合架构：
1.  **高分辨率感知 (Perception)**：
    *   使用 **SAM-B (Segment Anything Model)** 的 ViT 部分。
    *   **特点**：基于窗口注意力 (Window Attention)，计算量低，适合处理高分辨率（如 1024x1024）的局部细节，但缺乏全局语义。
    *   **参数**：仅 80M。
2.  **压缩桥梁 (Compression)**：
    *   一个 2 层的卷积模块。
    *   **作用**：将 SAM 输出的 tokens 进行 **16x 下采样**。例如 $1024 \times 1024$ 的图生成 4096 个 patch，经过压缩后变成 256 个 latent tokens。
3.  **语义理解 (Knowledge)**：
    *   使用 **CLIP-Large (ViT)**。
    *   **特点**：基于全局注意力 (Global Attention)，参数量 300M。
    *   **输入**：接收压缩后的 tokens，负责提取深层语义信息。

**设计哲学**：SAM 负责“看清”（Local details），CLIP 负责“看懂”（Semantic），中间通过卷积大幅压缩 Token 数量。

### 2.2 解码器：DeepSeek3B-MoE
*   使用 DeepSeek-VL 的 Decoder，参数量 3B，激活参数 570M (MoE)。
*   **作用**：将视觉 tokens 翻译回 Markdown 格式的文本。

### 2.3 分辨率策略：Gundam Mode (高达模式)
为了处理长文档（如报纸、论文），模型支持动态分辨率：
*   **Native Modes**：Tiny (512px), Small (640px), Base (1024px), Large (1280px)。
*   **Gundam Mode**：基于切片（Tile-based）。
    *   将大图切分为 $n$ 个 640x640 的局部视图（Local Views）。
    *   加上 1 个 1024x1024 的全局视图（Global View）。
    *   **Token 计算**：$N_{tokens} = n \times 100 + 256$。即便切 9 个图，总 Token 数也仅约 1156 个，远少于同类模型（如 MinerU 需要 6000+ tokens）。

## 3. 数据工程 (Data Engine)
高质量的 OCR 必须依赖高质量数据：
*   **OCR 1.0 (PDFs)**：收集 30M 页 PDF，覆盖 100 种语言。
*   **OCR 2.0 (Complex Parsing)**：针对图表（Charts）、公式（Formulas）、几何题（Geometry）生成的合成数据。
*   **General Vision**：引入 20% 的通用视觉数据（Caption/Detection），防止模型过拟合 OCR 任务，保持通用视觉能力。

## 4. 实验结论与意义
1.  **压缩极限**：证明了视觉通道可以将文本信息压缩 7-20 倍。
2.  **遗忘机制模拟**：随着压缩率增加（图片变模糊或 Token 变少），模型表现出类似人类记忆的“渐进式遗忘”（模糊记忆），这为 LLM 的记忆机制研究提供了新视角。
3.  **生产力工具**：单卡 A100 每天可生产 200k+ 页的高质量 Markdown 数据，成为 LLM 预训练数据的重要来源。

---

# 第二篇：DeepSeek-OCR 2: Visual Causal Flow
**论文核心**：推翻传统的“从左到右、从上到下”的机械式视觉扫描，引入**“视觉因果流” (Visual Causal Flow)**。
**核心假设**：人类阅读复杂文档（如报纸排版、图表）时，视线不是光栅扫描（Raster Scan），而是基于语义的因果跳转。VLM 的 Encoder 也应具备这种因果推理能力。

## 1. 核心痛点：光栅扫描的局限性
传统 VLM（如 Qwen-VL, InternVL）将图片切片后，按空间坐标（从左上到右下）展平成序列。
*   **问题**：对于多栏排版、插图环绕的文档，物理坐标相邻的 Token 在语义上可能毫无关系。强制 LLM 学习这种错误的序列会引入归纳偏差（Inductive Bias）。

## 2. 架构升级：DeepEncoder V2

DeepSeek-OCR 2 并没有升级 Decoder（依然是 3B MoE），而是彻底重构了 Encoder，使其具备“在看图时就进行推理”的能力。

### 2.1 架构变革：LLM as Vision Encoder
*   **移除 CLIP**：不再使用 CLIP 作为语义编码器。
*   **引入 Qwen2-0.5B**：直接使用一个小型的 LLM（Transformer Decoder）作为视觉编码器的后半部分。
*   **双流架构 (Dual-Stream)**：
    1.  **Visual Tokens (前缀)**：保留 SAM+Conv 提取的视觉特征，使用**双向注意力 (Bidirectional Attention)**。目的是保持全局感知（类似 ViT）。
    2.  **Causal Flow Queries (后缀)**：引入一组可学习的 Query Tokens，使用**因果注意力 (Causal Attention)**。

### 2.2 关键机制：注意力掩码 (Attention Mask)
这是实现“视觉因果流”的数学核心。Attention Mask $M$ 被设计为：

$$
M = \begin{bmatrix} 
\mathbf{1}_{m \times m} & \mathbf{0}_{m \times n} \\ 
\mathbf{1}_{n \times m} & \text{LowerTri}(n) 
\end{bmatrix}
$$

*   **左上 ($\mathbf{1}_{m \times m}$)**：Visual Tokens 之间完全可见（双向），确保看清整张图。
*   **左下 ($\mathbf{1}_{n \times m}$)**：Query Tokens 可以看到所有 Visual Tokens。
*   **右下 ($\text{LowerTri}(n)$)**：Query Tokens 之间是**因果可见**的（下三角矩阵）。即第 $i$ 个 Query 只能看到第 $1$ 到 $i-1$ 个 Query。

**原理解析**：
这迫使 Encoder 在生成第 $i$ 个 Query 时，必须基于之前的 Query 序列进行推理。模型学会了**“读完标题（Query 1），根据语义应该去读第一栏（Query 2），而不是读旁边无关的插图”**。这实际上是在 Encoder 阶段就完成了“阅读顺序”的重排。

### 2.3 训练策略
*   **三阶段训练**：
    1.  **Encoder Pretraining**：冻结 tokenizer，训练 Qwen2-Encoder，让它学会视觉重排序。
    2.  **Query Enhancement**：联合训练 Encoder 和 Decoder。
    3.  **Decoder Specialization**：冻结 Encoder，只训 Decoder，提高吞吐量。

## 3. 实验结果
1.  **OmniDocBench 性能**：DeepSeek-OCR 2 在使用 **更少 Vision Tokens** (1120 vs 1156) 的情况下，性能显著提升，尤其是在 **阅读顺序 (R-order)** 指标上，编辑距离从 0.085 降至 0.057，证明模型“看懂了”排版逻辑。
2.  **Deep Parsing**：模型具备了极强的结构化解析能力，能直接输出图表对应的 HTML 表格、几何图形的 Python 代码等。

## 4. 总结与意义
DeepSeek-OCR 2 的核心贡献在于**打破了 Vision Encoder 和 LLM Decoder 之间的界限**：
*   **传统架构**：Encoder 负责“拍照片”（静态特征），Decoder 负责“讲故事”（因果生成）。
*   **DeepSeek 架构**：Encoder 本身就是一个**因果推理者**。它不是在拍照片，而是在模拟人类的**眼动追踪 (Eye Movement)**，按逻辑顺序提取视觉信息。
*   **未来启示**：这种 **"LLM as Encoder"** 的范式可能成为未来原生多模态（Native Multimodality）的标准，让视觉编码本身具备推理能力。