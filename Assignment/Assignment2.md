# Assignment 2 — Semantic Segmentation with UNet++ (PyTorch Lightning) on Oxford‑IIIT Pet

**Deadline:** 2025-10-21 1:00 PM 


---

## 0) What you will do
- Explore the **Oxford‑IIIT Pet** dataset and its **trimap** segmentation masks using the provided script `oxpet_download_and_viz_fixed.py`.
- Implement a **PyTorch Lightning** training pipeline for **UNet++** semantic segmentation.
- Train, evaluate, and report **mIoU** and **Dice** (macro/per‑class).
- Produce qualitative results: show input, GT mask (colorized), and prediction overlays.
- Try at least **two improvements** and analyze their effect.

> You **must** use **PyTorch Lightning** for the training loop (trainer, callbacks, logging).

---

## 1) Dataset

### 1.1 Oxford‑IIIT Pet overview
- Images of cats & dogs; pixel‑wise **trimap** segmentation masks.  
- Trimap labels (original mask PNG values):
  - `1` — **pet (foreground)**
  - `2` — **background**
  - `3` — **border / ambiguous**


### 1.2 Folder layout (after download / extraction)
Typical structure under the dataset root (e.g., `~/data/oxford-iiit-pet/`):
```
oxford-iiit-pet/
  images/                         # All images (.jpg)
  annotations/
    trimaps/                      # PNG masks (values in {1,2,3})
    xmls/                         # VOC-style XML (not needed for this assignment)
    list.txt                      # Metadata list
    trainval.txt                  # Filenames (no extension) used for trainval split
    test.txt                      # Filenames (no extension) used for test split
```
**What each file/folder is for**
- `images/`: All RGB JPEG images, filenames match masks (same basename).
- `annotations/trimaps/`: **Segmentation masks**; you will convert to 3‑class or 2‑class IDs.
- `annotations/xmls/`: Detection/classification XMLs (**not used** for this assignment).
- `annotations/list.txt`: Global list with per‑image metadata.
- `annotations/trainval.txt` and `annotations/test.txt`: Official splits. You must further split **train/val** from `trainval` (e.g., 80/20).

### 1.3 Data exploration (required)
Use the provided script to visually verify images & masks before training:
```bash
python oxpet_download_and_viz_fixed.py \
  --root ~/data \
  --split trainval \
  --classes trimap \
  --n 12 \
  --resize 512
```
This creates:
```
samples/
  oxpet_viz_grid.png      # grid of (image, colored mask, overlay)
  overlay_*               # a few single-image overlays
```

---

## 2) Rules
- Work in **teams of two**.
- Use only the **trainval** split for model development; create your own **train/val** split (80/20 or 85/15). The official **test** split may be used for final reporting. I will give extra 1 points for winning team.
- You may consult public materials, but your **code must be your own**. Cite any external sources you drew ideas from.
- **PyTorch Lightning is required** for the training loop and logging.
- Keep total **GPU time ≤ 10 hours**. If you adjust crop size/epochs due to hardware, explain and justify it in the report.

---

## 3) Deliverables

1. **Code** (repo or zip)
   - Lightning modules (DataModule + LightningModule)
   - Training script (using `pl.Trainer` with AMP, checkpointing, logging)
   - Evaluation script/notebook (metrics + figures)
   - Inference/visualization utilities (produce overlays for at least 12 samples)
   - A short **README.md** with exact commands to reproduce your results

2. **Report (.pdf, ≤ 4 pages)**
   - Setup: task (3‑class), transforms, model/encoder, loss, optimizer, scheduler, batch size, epochs, AMP
   - **Metrics**: overall and per‑class **mIoU** and **Dice**
   - **Qualitative results**: at least 12 comparisons (input/GT/pred + overlays)
   - **Two improvements**: what you changed and how it affected results (include a small ablation table)
   - **Weight&Bias dashboard link**: Share you weights&bias dashboard link with me.
   - **Citations**: any external references

3. **Best weights** (`.ckpt` or `.pt`) saved by the Trainer (checkpoint with best val mIoU or Dice).

---

## 4) Baseline Configuration (fits in ≤10h on one RTS 5070 TI GPU)
- **Model**: UNet++ with lightweight encoder  
  `encoder_name = "timm-efficientnet-b0"` (or `"mobilenet_v2"`), `classes = 3` (trimap) or `2` (binary)
- **Input**: `RandomResizedCrop(512, 512, scale=(0.5, 2.0))` + HorizontalFlip; **NEAREST** for mask interpolation
- **Loss**:
  - Trimap (3‑class): CrossEntropy (optional class weights) **+ Dice** (e.g., 0.5/0.5)
- **Batch size**: 8 (reduce to 4/6 if OOM); **AMP** on
- **Epochs**: 60–100 (you may stop early via early‑stopping callback)

---

## 5) Project Structure
```
Assignment2/
  README.md
  requirements.txt
  oxpet_download_and_viz_fixed.py       # provided exploration script
  src/
    datamodule_oxpet.py                 # LightningDataModule (I suggest you use Lightning, but this is not required.)
    model_unetpp.py                     # LightningModule (UNet++)
    train.py                            # CLI to train/eval with pl.Trainer
    viz.py                              # utilities to colorize masks & make overlays
  outputs/
    checkpoints/
    logs/
    samples/
```

---

## 6) How to run
- Install dependencies, explore dataset with `oxpet_download_and_viz_fixed.py`, train with `src/train.py`, evaluate and visualize results.

---

## 7) Improvements
Please ask AI to pick any two (augmentation, loss, backbone, TTA, optimization).

---

## 8) Grading (100 pts)
| Component | Pts |
|-----------|-----|
| Data pipeline (correct masks & transforms) | 15 |
| Lightning training pipeline | 20 |
| Metrics & learning curves | 15 |
| Baseline performance | 15 |
| Two improvements | 20 |
| Code clarity | 10 |
| Report & citations | 5 |

---

## 10) Checklist
- Verified data with provided script  
- Implemented Lightning Module  
- Used NEAREST for mask interpolation  
- Saved best checkpoint by val mIoU/Dice  
- Report includes metrics, curves, visuals, two improvements  
- Included README and reproducibility instructions

---

