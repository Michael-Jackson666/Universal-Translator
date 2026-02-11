# DeepSeek-OCR-2 Source Code

This folder contains the official source code for **DeepSeek-OCR-2** (second generation), an improved document OCR model by DeepSeek.

## Structure

```
DeepSeek-OCR2-master/
├── DeepSeek-OCR2-hf/         # HuggingFace Transformers implementation
│   └── run_dpsk_ocr2.py      # Simple inference script using AutoModel
└── DeepSeek-OCR2-vllm/       # vLLM-based implementation for high-throughput serving
    ├── config.py             # Model configuration
    ├── deepencoderv2/        # Vision encoder modules (v2)
    ├── deepseek_ocr2.py      # Main OCR-2 model
    ├── process/              # Image preprocessing utilities
    ├── run_dpsk_ocr2_eval_batch.py  # Batch evaluation
    ├── run_dpsk_ocr2_image.py       # Single image inference
    └── run_dpsk_ocr2_pdf.py         # PDF document processing
```

## Key Improvements over OCR-1

- **Model**: `deepseek-ai/DeepSeek-OCR-2` from HuggingFace Hub
- **Enhanced Vision Encoder**: `deepencoderv2` with improved architecture
- **Default Resolution**: 768 crop size (vs 640 in OCR-1) for better detail capture
- **Same API**: Compatible interface with OCR-1 for easy migration

## Quick Start (HuggingFace)

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'deepseek-ai/DeepSeek-OCR-2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\n<|grounding|>Convert the document to markdown."
res = model.infer(tokenizer, prompt=prompt, image_file='doc.jpg',
                  base_size=1024, image_size=768, crop_mode=True)
```

## Source

Official DeepSeek-AI repository for document understanding and OCR capabilities (second generation).
