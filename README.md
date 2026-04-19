# Image-to-Text Generation Using Vision Transformers
and Language Models

A multi-milestone deep learning project that builds an **automatic image captioning system** by bridging OpenAI's CLIP visual encoder with GPT-2's language generation capabilities. Trained and evaluated on the MS-COCO dataset across three progressive milestones.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Milestones](#milestones)
  - [Milestone 1 — Data Pipeline & CLIP Embeddings](#milestone-1--data-pipeline--clip-embeddings)
  - [Milestone 2 — Prefix Tuning](#milestone-2--prefix-tuning)
  - [Milestone 3 — LoRA & Full Fine-Tuning](#milestone-3--lora--full-fine-tuning)
  - [Demo — Unified Inference](#demo--unified-inference)
- [Model Architecture](#model-architecture)
- [Decoding Strategies](#decoding-strategies)
- [Setup & Usage](#setup--usage)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Project Overview

This project implements a **CLIP-conditioned image captioning pipeline** using a lightweight projection MLP that maps CLIP visual embeddings into GPT-2's token embedding space as "prefix tokens." Three training strategies are progressively explored:

| Strategy | GPT-2 Weights | Projection MLP | Prefix Length |
|---|---|---|---|
| **Prefix Tuning** (M2) | Frozen | Trainable | 10 |
| **LoRA Fine-Tuning** (M3) | LoRA adapters | Trainable | 20 |
| **Full Fine-Tuning** (M3) | Fully unfrozen | Trainable | 20 |

All experiments are run in **Google Colab** with GPU acceleration and data/checkpoints persisted to **Google Drive**.

---

## Repository Structure

```
├── NNDL_M1.ipynb          # Milestone 1: Data download, cleaning, CLIP embedding extraction
├── NNDL_M2.ipynb          # Milestone 2: Prefix tuning model — training & decoding
├── NNDL_M3.ipynb          # Milestone 3: LoRA + Full FT — training, evaluation & comparison
└── NNDL_Demo.ipynb        # Demo: Upload any image → captions from all 3 models
```

**Google Drive layout (auto-created at runtime):**
```
NeuralNetworks_Project2/
├── coco_subset/
│   ├── images/                    # 2,500 downloaded COCO images
│   ├── annotations/               # COCO captions JSON
│   ├── dataset_2500.json
│   ├── dataset_cleaned.json
│   ├── image_embeddings.npy       # CLIP embeddings (N × 512)
│   └── embeddings_meta.json
├── milestone2/
│   └── best_model_fixed.pt        # Prefix tuning checkpoint
└── milestone3/
    ├── lora_model.pt              # LoRA checkpoint
    └── full_ft_model.pt           # Full fine-tuning checkpoint
```

---

## Milestones

### Milestone 1 — Data Pipeline & CLIP Embeddings

**Notebook:** `NNDL_M1.ipynb`

Builds the full data foundation from scratch:

- Downloads **MS-COCO 2017** captions annotation file (~240 MB)
- Randomly samples **2,500 unique image–caption pairs** (seed = 42)
- Downloads the corresponding COCO images
- **Caption cleaning:** lowercasing, special-character removal, whitespace normalization, length filtering (3–50 words)
- **Image validation:** rejects corrupted files, resizes to 224×224
- Extracts **CLIP ViT-B/32 embeddings** for every image and saves them as `.npy`
- Saves tokenized captions (GPT-2 tokenizer, max length 32) as `.pt`

**Outputs saved to Drive:**
- `image_embeddings.npy` — CLIP feature matrix (N × 512)
- `embeddings_meta.json` — image IDs, filenames, captions
- `tokenized_captions.pt` — GPT-2 tokenized caption tensors

---

### Milestone 2 — Prefix Tuning

**Notebook:** `NNDL_M2.ipynb`

Trains a **frozen-GPT-2 prefix conditioning model**:

- Loads M1 embeddings and all 5 COCO captions per image (5× data expansion)
- Builds `ClipCaptionModel`: a 2-layer MLP projection (512 → 1024 → `prefix_len × 768`) prepended to GPT-2's token embedding stream
- **Only the MLP projection is trained** — GPT-2 weights are fully frozen
- Training: cross-entropy loss on caption tokens (shifted by prefix length), AdamW optimizer, learning rate scheduling, early stopping
- Saves best checkpoint: `best_model_fixed.pt` (`PREFIX_LEN=10`)

---

### Milestone 3 — LoRA & Full Fine-Tuning

**Notebook:** `NNDL_M3.ipynb`

Extends M2 with two stronger training regimes:

**LoRA Fine-Tuning:**
- Applies LoRA adapters (`r=16`, `alpha=32`, `dropout=0.1`) to GPT-2's attention layers (`c_attn`, `c_proj`) via the `peft` library
- Trains both the LoRA adapters and the projection MLP
- `PREFIX_LEN=20`, 15 epochs, LR=5e-5 with 200 warmup steps

**Full Fine-Tuning:**
- All GPT-2 weights are unfrozen and trained end-to-end alongside the projection
- `PREFIX_LEN=20`, same training configuration

**Evaluation metrics computed on the validation set:**
- BLEU-1 / BLEU-4
- METEOR
- ROUGE-L

All three models (Prefix Tuning, LoRA, Full FT) are compared side-by-side across all metrics and decoding strategies.

---

### Demo — Unified Inference

**Notebook:** `NNDL_Demo.ipynb`

A plug-and-play inference notebook:

- Loads all three trained model checkpoints from Drive
- Accepts **any image** — file path, URL, or PIL Image object
- Runs all 3 decoding strategies on all 3 models
- Displays a clean comparison grid with captions side-by-side

---

## Model Architecture

```
Image
  │
  ▼
CLIP ViT-B/32  ──→  512-dim embedding (normalized)
  │
  ▼
Projection MLP
  Linear(512 → 1024) → ReLU → Dropout(0.1) → Linear(1024 → prefix_len × 768)
  │
  ▼
Reshape → (batch, prefix_len, 768)   ← "visual prefix tokens"
  │
  ▼
Concatenate with GPT-2 caption token embeddings
  │
  ▼
GPT-2 (frozen / LoRA / fully trained)
  │
  ▼
Caption tokens  →  Decode  →  Caption string
```

---

## Decoding Strategies

All three strategies are implemented with a **repetition penalty** (`penalty=1.3`) to reduce degenerate output:

| Strategy | Description | Key Parameters |
|---|---|---|
| **Greedy** | Argmax at each step | `max_new=30` |
| **Beam Search** | Maintains top-k beam hypotheses | `num_beams=5` |
| **Nucleus Sampling** | Samples from top-p probability mass | `top_p=0.9`, `temp=0.8` |

> **Best overall:** LoRA model + Beam Search

---

## Setup & Usage

All notebooks are designed for **Google Colab**. No local setup is required.

### Step 1 — Open in Colab

Upload the `.ipynb` files to your Google Drive or open them directly in Colab.

### Step 2 — Enable GPU

```
Runtime → Change runtime type → T4 GPU
```

### Step 3 — Run in order

```
NNDL_M1.ipynb  →  NNDL_M2.ipynb  →  NNDL_M3.ipynb  →  NNDL_Demo.ipynb
```

Each notebook mounts your Drive and reads outputs from the previous milestone automatically.

### Demo — Quick Inference

Open `NNDL_Demo.ipynb`, run all cells, and upload any image when prompted:

```python
# From URL
run_inference("https://example.com/your-image.jpg")

# From local path (after upload)
run_inference("/content/your-image.jpg")
```

---

## Results

Evaluation is performed on the held-out validation split. The table below summarizes representative scores (exact values logged during M3 training):

| Model | Decoding | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|
| Prefix Tuning | Greedy | 35.86 | 5.10 | 35.84 | 28.15 | 1.94 |
| Prefix Tuning | Beam Search | 37.40 | 6.55 | 39.05 | 29.77 | 3.28 |
| Prefix Tuning | Nucleus | 32.85 | 3.42 | 33.14 | 25.94 | 1.00 |
| LoRA | Greedy | 40.62 | 7.00 | 39.38 | 29.09 | 3.38 |
| **LoRA** | **Beam Search** | **40.90** | **8.76** | **41.64** | **30.37** | **5.40** |
| LoRA | Nucleus | 36.25 | 4.70 | 35.49 | 26.54 | 1.58 |
| Full FT | Greedy | 37.39 | 6.28 | 37.71 | 27.49 | 2.87 |
| Full FT | Beam Search | 38.02 | 7.80 | 39.10 | 28.00 | 4.62 |
| Full FT | Nucleus | 33.74 | 4.33 | 32.79 | 25.38 | 1.72 |

> **Best overall: LoRA + Beam Search** — highest scores across all five metrics. Beam Search consistently outperforms Greedy and Nucleus across all models. LoRA strikes the best balance between parameter efficiency and caption quality, outperforming Full Fine-Tuning despite updating fewer weights.

---

## Dependencies

All dependencies are installed automatically inside each notebook via `!pip install`. No manual setup needed.

| Library | Purpose |
|---|---|
| `torch` | Model training & inference |
| `transformers` | GPT-2 model & tokenizer |
| `clip` (OpenAI) | CLIP ViT-B/32 visual encoder |
| `peft` | LoRA adapter support |
| `nltk` | BLEU & METEOR scoring |
| `rouge-score` | ROUGE-L scoring |
| `Pillow` | Image loading & preprocessing |
| `numpy` | Embedding storage & manipulation |
| `matplotlib` | Training curves & result visualization |

---

## Course

**IE 7615 — Neural Networks & Deep Learning**  
Northeastern University
