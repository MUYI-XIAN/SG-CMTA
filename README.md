# SG-CMTA
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## Overview

SGCMTA is a multi-modal survival prediction framework that integrates **whole slide pathology images (WSIs)** and **genomic data** through cross-modal transformer attention. Built upon the CMTA (Cross-Modal Transformer Attention) architecture, this work introduces a **Three-Module Collaborative Innovation Suite** to enhance the positional encoding within the pathomics transformer branch, replacing the standard multi-scale convolutional position encoding with a more expressive attention-driven mechanism.


## Requirements

### System Dependencies

[OpenSlide](https://openslide.org/) is required for reading `.svs` whole slide images:

```bash
# Ubuntu / Debian
sudo apt-get install openslide-tools

# macOS
brew install openslide

# Windows: Download from https://openslide.org/download/
```

### Python Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn scikit-survival \
    openslide-python opencv-python-headless matplotlib einops tqdm \
    tensorboardX Pillow
```

Full list of key dependencies:

- `torch >= 1.12`
- `torchvision`
- `openslide-python`
- `scikit-survival`
- `scikit-learn`
- `einops`
- `opencv-python-headless`
- `matplotlib`
- `tensorboardX`
- `pandas`, `numpy`, `tqdm`

## Data Preparation

### 1. Directory Structure

```
project_root/
├── SGCMTA.py                  # Main script (single-file)
├── csv/
│   ├── tcga_luad_all_clean.csv  # Metadata CSV
│   └── signatures.csv           # Genomic signatures (optional)
├── data/
│   └── svs/                     # WSI .svs files
├── features/                    # Auto-generated feature cache
└── results/                     # Training outputs & heatmaps
```

### 2. Metadata CSV Format

The metadata CSV file (e.g., `csv/tcga_luad_all_clean.csv`) should contain the following columns:

| Column | Description |
|--------|-------------|
| `case_id` | Patient / case identifier |
| `slide_id` | Slide identifier (matching `.svs` filenames) |
| `censorship` | Censorship indicator (0 = event occurred, 1 = censored) |
| `survival_months` | Survival time in months |
| *genomic columns* | Additional columns for genomic features (mutations, CNV, RNA-seq, etc.) |

### 3. Genomic Signatures (Optional)

If `csv/signatures.csv` exists, it is used to group genomic features into 6 omic signature groups. Each column in the signatures file lists gene names belonging to that pathway. Feature names are matched with suffixes `_mut`, `_cnv`, and `_rnaseq`.

## Usage

### Basic Training

```bash
python SGCMTA.py \
    --data_root_dir ./data/svs/ \
    --dataset tcga_luad \
    --num_epoch 20 \
    --lr 2e-4
```

### Full Argument Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root_dir` | `./data/svs/` | Path to WSI `.svs` files |
| `--feature_dir` | `./features/` | Directory for cached features |
| `--dataset` | `tcga_luad` | Cancer type identifier |
| `--seed` | `1` | Random seed |
| `--num_epoch` | `20` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--batch_size` | `1` | Batch size (1 recommended due to variable bag sizes) |
| `--optimizer` | `Adam` | Optimizer: `SGD`, `Adam`, `AdamW`, `RAdam`, `PlainRAdam`, `Lookahead` |
| `--scheduler` | `cosine` | LR scheduler: `None`, `exp`, `step`, `plateau`, `cosine` |
| `--loss` | `nll_surv_l1` | Loss function |
| `--fusion` | `concat` | Fusion strategy: `concat` or `bilinear` |
| `--model_size` | `small` | Model size: `small` or `large` |
| `--alpha` | `0.5` | Weight for similarity loss |
| `--patch_level` | `2` | SVS read level (0 = highest resolution) |
| `--patch_size` | `32` | Patch size for feature extraction |
| `--use_innovation` | `True` | Enable the SGAP + DSEP + ARFH modules |
| `--no_innovation` | — | Disable innovation modules (ablation study) |
| `--weighted_sample` | `True` | Enable class-balanced sampling |

### Ablation Study

To compare the standard positional encoding baseline against the proposed modules:

```bash
# With innovation modules (default)
python SGCMTA.py --use_innovation --dataset tcga_luad

# Without innovation modules (baseline)
python SGCMTA.py --no_innovation --dataset tcga_luad
```

## Pipeline

The script runs an end-to-end pipeline:

1. **Feature Extraction** — Automatically extracts and caches ResNet-50 features from WSI `.svs` files (with tissue detection via Otsu thresholding).
2. **5-Fold Stratified Cross-Validation** — Trains and validates using stratified K-fold splits based on discretized survival labels.
3. **Training** — Optimizes a combined loss of NLL survival loss and inter-modal similarity loss.
4. **Evaluation** — Reports per-fold and mean C-Index with standard deviation.
5. **Visualization** — Generates attention heatmaps overlaid on WSI thumbnails for the last fold's best model.

## Outputs

After training, results are organized as follows:

```
results/
└── tcga_luad/
    ├── [cmta]-fold_1-[timestamp]/
    │   ├── model_best_0.6821_15.pth.tar
    │   └── events.out.tfevents.*
    ├── [cmta]-fold_2-[timestamp]/
    │   └── ...
    └── [cmta]-CV_Summary/
        └── attention_heatmap_TCGA-XX-XXXX.png
```

- **Model checkpoints**: Best model per fold saved as `.pth.tar`
- **TensorBoard logs**: Training/validation loss and C-Index curves
- **Attention heatmaps**: Spatial attention visualization overlaid on tissue regions

### Viewing TensorBoard Logs

```bash
tensorboard --logdir ./results/tcga_luad/
```





## Acknowledgments

This project builds upon the [CMTA](https://github.com/is-the-biern/CMTA) framework and incorporates components from:

- [Nystrom Attention](https://github.com/mlpen/Nystromformer)
- [CLAM](https://github.com/mahmoodlab/CLAM) (for WSI processing paradigms)

## License

This project is released under the [MIT License](LICENSE).
