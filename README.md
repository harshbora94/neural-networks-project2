# neural-networks-project2

## Neural Networks — Generative Project (Image Captioning)

### Overview
End-to-end image captioning pipeline using CLIP (ViT-B/32) for image embeddings
and GPT-2 for caption tokenization and generation.

### Dataset
- **Source**: COCO Captions 2017 (train split)
- **Subset**: 2,500 image–caption pairs
- **Preprocessing**: lowercased, cleaned, resized to 224×224 RGB

### Pipeline
```
Image → CLIP (ViT-B/32) → 512-dim embedding
Caption → GPT-2 Tokenizer → input_ids (max_length=32)
```

### Repo Structure
```
notebooks/    Colab notebooks for each milestone step
data/         Tokenizer stats and dataset metadata samples
outputs/      Sample image:caption test run figures
docs/         Project proposal and milestone reports
```

### Setup
1. Open any notebook in `notebooks/` in Google Colab
2. Mount Google Drive when prompted
3. Run cells top to bottom

### Tech Stack
- PyTorch
- Hugging Face Transformers (GPT-2)
- OpenAI CLIP (ViT-B/32)
- COCO Captions dataset

### Team
- Member 1 — Data Engineering
- Member 2 — Vision Pipeline (CLIP)
- Member 3 — Language Pipeline (GPT-2)
- Member 4 — Integration & Documentation
