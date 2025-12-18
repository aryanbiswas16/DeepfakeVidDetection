# DeepfakeDetector_v3 — Model Build, Training, and Usage Guide

This document is a practical “handoff” guide for reproducing the **DeepfakeDetector_v3** workflow end-to-end:

- How to set up the environment
- How to generate **paired real/fake training frames** from videos
- How to train the model
- How to run inference on a video
- How checkpoints are saved/loaded
- Common failure modes and troubleshooting

The repository implements a **frame-based** deepfake detector with a **DINOv2-first encoder** (via `timm` if available) and a **ResNet50 fallback** (so the repo remains runnable even without DINOv2).

---

## 1) What this model is

### 1.1 Problem framing
The goal is to predict whether a video is **REAL** or **FAKE** by:

1. Sampling a fixed number of frames from the video
2. Preprocessing each frame (resize + normalize)
3. Encoding each frame into an embedding vector
4. Classifying each frame (real vs fake)
5. Aggregating frame probabilities into a single video score

### 1.2 Output definition
At inference time, the detector returns:

- `score`: mean probability of **FAKE** across sampled frames
- `label`: `FAKE` if `score > 0.5` else `REAL`
- `details`: per-frame probabilities and a simple instability estimate

---

## 2) Repository structure (what matters)

Minimal layout used by training and smoke tests:

```
DeepfakeDetector_v3/
  datasets/
    __init__.py
    paired_dataset.py
    paireddataset.py              # optional compatibility module
  models/
    __init__.py
    frame_encoder.py
    classifier_head.py
    detector.py
  scripts/
    extract_paired_frames.py
  training/
    train.py
  utils/
    preprocess.py
    video_io.py
    weights.py
  apps/
    streamlit_app.py
  smoke_test.py
  requirements.txt
  README.md
```

Key entrypoints:

- Smoke test: `smoke_test.py`
- Training: `training/train.py`
- Data extraction: `scripts/extract_paired_frames.py`

---

## 3) Environment setup

### 3.1 Prerequisites
- Python 3.10+ recommended (your environment uses Python 3.11)
- A working PyTorch install (CPU or CUDA)

### 3.2 Install dependencies
From the repo root:

```bash
pip install -r requirements.txt
```

### 3.3 Optional: enable DINOv2 via timm
If you want the **DINOv2-first** path to work (instead of ResNet50 fallback), install `timm`:

```bash
pip install timm
```

Notes:
- If `timm` is missing, the encoder will warn and fall back to ResNet50.
- DINOv2 model name availability depends on your `timm` version/build.

---

## 4) Data preparation: creating paired frames

### 4.1 The paired-frame concept
Training expects two folders of **images**:

- `data/train/real/`
- `data/train/fake/`

with **matching filenames**:

```
real/<name>.jpg  AND  fake/<name>.jpg
```

This “paired” requirement reduces shortcut learning from dataset-level artifacts because each pair should share nuisance factors (scene, identity, lighting, compression patterns) while differing primarily by manipulation.

### 4.2 Input video requirement
You typically start from:

- a directory of real videos (`REAL_VIDEOS_DIR`)
- a directory of fake videos (`FAKE_VIDEOS_DIR`)

Ideally the fake videos correspond to the real videos by basename or a deterministic mapping.

### 4.3 Extract paired frames
Run the extraction script:

```bash
python scripts/extract_paired_frames.py \
  --real_dir REAL_VIDEOS_DIR \
  --fake_dir FAKE_VIDEOS_DIR \
  --out_dir data/train \
  --num_frames 16
```

Expected output:

```
data/train/real/<something>_frame_0000.jpg
data/train/fake/<something>_frame_0000.jpg
...
```

### 4.4 Data sanity checklist
Before training:

- `data/train/real` exists
- `data/train/fake` exists
- There is **at least one filename** that appears in both folders
- Images open correctly with PIL

---

## 5) Preprocessing (what gets fed to the encoder)

The preprocessing pipeline is defined in `utils/preprocess.py`:

- Resize to `image_size x image_size` (default 224)
- Convert to tensor
- Normalize with ImageNet mean/std

This is compatible with standard vision backbones and works for both the ResNet fallback and common ViT-style models.

---

## 6) Model architecture

### 6.1 Frame encoder (`models/frame_encoder.py`)
The frame encoder produces an embedding vector per frame.

Behavior:

- Tries to build DINOv2 via `timm.create_model(..., num_classes=0)`
- If unavailable, falls back to torchvision ResNet50 with the classifier removed
- Freezes the backbone by default
- If a ViT-like DINOv2 is used, optionally unfreezes only LayerNorm parameters ("LayerNorm tuning")

Outputs:

- `embed_dim`: embedding dimension $D$
- `image_size`: expected input resolution (224)

### 6.2 Classifier head (`models/classifier_head.py`)
A small MLP that maps embeddings `[B, D]` → logits `[B, 2]`.

Interpretation:

- logit index 0 = REAL
- logit index 1 = FAKE

### 6.3 Detector wrapper (`models/detector.py`)
A convenience wrapper combining encoder + head.

Key APIs:

- `forward(images)` → `(logits, embeddings)`
- `predict_video(frames_tensor)` → `{score, label, details}`

The detector also exposes `det.preprocess` so apps/tests can reuse the correct resize/normalize settings.

---

## 7) Training procedure

Training is implemented in `training/train.py`.

### 7.1 Dataset
`datasets/paired_dataset.PairedDataset` loads paired frames and returns:

- `real_img_tensor`
- `fake_img_tensor`
- `filename`

### 7.2 Batch construction
Per iteration:

- concatenate real and fake tensors along batch dimension
- build labels: `0` for real, `1` for fake

### 7.3 Loss function
The training script uses a combined objective:

1. Cross entropy on logits vs labels
2. A simple embedding “metric loss” term based on cosine similarity

Total loss:

$$\mathcal{L} = \mathcal{L}_{CE} + 0.5 \cdot \mathcal{L}_{metric}$$

### 7.4 What parameters are trained
- Always trains `detector.head`
- Also trains any encoder parameters left unfrozen (intended to be LayerNorm-only in DINOv2 mode)

### 7.5 Mixed precision
If CUDA is available, the training script uses AMP (`autocast` + `GradScaler`) to reduce memory and speed up training.

---

## 8) Running the pipeline

### 8.1 Smoke test
The smoke test verifies that:

- the detector can be instantiated
- preprocessing works
- frame stacking works
- `predict_video` returns a result

Run:

```bash
python smoke_test.py
```

### 8.2 Train
Once `data/train/real` and `data/train/fake` exist:

```bash
python training/train.py
```

The script writes a checkpoint to:

```
training/weights/dinov2_best.pth
```

---

## 9) Checkpoint format and loading

### 9.1 Checkpoint structure
The training script saves a dictionary:

- `head`: `detector.head.state_dict()`
- optionally `encoder`: filtered encoder params (LayerNorm / LN / norm)

### 9.2 Loading checkpoints
Use `utils/weights.py` to load weights.

Security note:
- PyTorch checkpoints use Python pickle under the hood.
- Only load checkpoints you trust.

---

## 10) Inference on real videos

### 10.1 Frame sampling
Use `utils/video_io.py` to read and sample frames from a video.

Typical flow:

1. `frames = read_video_frames(path, num_frames=N)` (returns list of PIL images)
2. `preprocess = detector.preprocess`
3. `frames_tensor = stack_frames(frames, preprocess, device)`
4. `result = detector.predict_video(frames_tensor)`

### 10.2 Output interpretation
- Higher `score` means "more fake".
- `details.instability` gives a crude sense of variance across frames.

---

## 11) Troubleshooting

### 11.1 “No paired training data found”
Cause:
- `data/train/real` or `data/train/fake` missing

Fix:
- generate paired frames using the extraction script

### 11.2 “No paired files found…”
Cause:
- filenames don’t match between real/ and fake/

Fix:
- ensure both directories contain identically named images

### 11.3 DINOv2 warnings / fallback
If you see warnings about DINOv2 not being available:

- Install `timm` and re-run
- Verify your `timm` provides one of the candidate model names (see `models/frame_encoder.py`)

### 11.4 Torch/torchvision install issues
If import errors occur:

- reinstall dependencies
- ensure you’re using the intended Python interpreter

---

## 12) Reproducibility notes (practical)

For more deterministic runs:

- set random seeds for `torch`, `numpy`, and Python
- pin versions in `requirements.txt`
- ensure the same data generation procedure and frame sampling parameters

---

## 13) Quick copy/paste (minimal end-to-end)

```bash
pip install -r requirements.txt

python scripts/extract_paired_frames.py --real_dir REAL_VIDS --fake_dir FAKE_VIDS --out_dir data/train --num_frames 16

python smoke_test.py

python training/train.py
```
