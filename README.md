# 3D Brain Tumor Segmentation from Multimodal MRI

This repository presents an end-to-end workflow for **multi-label brain tumor segmentation** from 3D multimodal MRI scans. The project targets segmentation of **glioma subregions** (tumor core, whole tumor, enhancing tumor) using a **3D SegResNet** model, trained with **Dice loss** and evaluated using **Mean Dice metrics**. The pipeline is implemented with **PyTorch** and **MONAI**, and experiment tracking, visualization, and artifact management are integrated through **Weights & Biases (W&B)**.

---

## Key Features

- 3D multi-label segmentation of gliomas
- Multimodal MRI input (FLAIR, T1, T1Gd, T2)
- **3D SegResNet** architecture with mixed-precision training
- Dice Loss optimization + Mean Dice evaluation
- Sliding window inference for volumetric prediction
- End-to-end W&B experiment tracking
- ROI-based patch sampling for computational efficiency
- Checkpointing, logging, and interactive visualization

---

## Dataset

**Source:** Medical Segmentation Decathlon – Task 01 Brain Tumor  
**Link:** http://medicaldecathlon.com/  
**Data format:** 750 multimodal 3D MRI volumes (484 train / 266 test)  
**Modalities:** FLAIR, T1w, T1gd, T2w

**Multi-Label Channel Mapping**

| Channel | Clinical Target Class            | Description                                 |
|---------|----------------------------------|---------------------------------------------|
| 0       | Tumor Core (TC)                  | Non-enhancing + necrotic core              |
| 1       | Whole Tumor (WT)                 | Edema + Core + Enhancing                   |
| 2       | Enhancing Tumor (ET)             | Actively enhancing tumor tissue            |

This follows the BraTS standard labeling convention (IEEE TMI).

---

## Methodology Overview

**Pipeline Steps**
1. Data download & preprocessing
2. One-hot label conversion to multi-channel segmentation
3. Spatial normalization (RAS orientation, voxel spacing)
4. ROI-based 3D patch extraction
5. SegResNet training with mixed precision
6. Sliding window inference for volume-level prediction
7. Visualization & evaluation in W&B

**Core Transforms**
- `Orientationd`, `Spacingd`
- `NormalizeIntensityd`
- `RandSpatialCropd`
- `RandFlipd`, `RandShiftIntensityd`, `RandScaleIntensityd`

---

## Model Architecture

**Network:** 3D SegResNet

```python
SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,
)
```

**Rationale**
- Residual encoder-decoder structure
- Effective multimodal feature extraction
- Optimized for 3D volumetric segmentation

---

## Training Configuration

| Parameter | Value |
|-----------|--------|
| Loss | Dice Loss |
| Optimizer | Adam |
| LR Scheduler | Cosine Annealing |
| Initial LR | 1e-4 |
| Epochs | 50 |
| Batch Size | 1 |
| ROI Crop | `[224, 224, 144]` |
| AMP | Enabled (Mixed Precision) |

**Dice Loss**
\[
L_{dice} = \frac{2 \sum p_{true} p_{pred}}{\sum p_{true}^2 + \sum p_{pred}^2 + \epsilon}
\]

---

## Results & Evaluation

**Primary Metric:** Mean Dice (validation)

| Class             | Dice Score (Val) |
|-------------------|------------------|
| Tumor Core (TC)   | TBD              |
| Whole Tumor (WT)  | TBD              |
| Enhancing Tumor   | TBD              |
| **Mean Dice**     | TBD              |

> Update when training is finalized.

---

## Visual Examples

Place your images/GIFs in the `assets/` directory and update paths below.

**Model Output Comparison**
```
![Overlay Example](assets/overlay_example.png)
```

**Volumetric MRI Slice**
```
![MRI Slice](assets/mri_slice_01.png)
```

**Inference Video Demo**
```
https://github.com/<username>/<repo>/assets/inference_demo.mp4
```

---

## Code Structure

```
📦 3D-brain-tumor-segmentation
├── 3D_Brain_tumor_segmentation.ipynb      # Main training notebook
├── checkpoints/                           # Model weights (ignored in version control)
├── assets/                                # Images, GIFs, videos for README
├── dataset/                               # Local MRI dataset (add to .gitignore)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Usage

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run Training (Notebook Execution)
```
Open and execute:
3D_Brain_tumor_segmentation.ipynb
```

### 3. Load Best Model for Inference
```python
model.load_state_dict(torch.load("checkpoints/model.pth"))
model.eval()
```

---

## Experiment Tracking (W&B)

This project logs:
- Training/validation metrics
- Model checkpoints as **versioned artifacts**
- Slice-by-slice overlay visualizations
- Full prediction comparison tables

Example workspace link:
```
https://wandb.ai/<your-username>/3D-brain-tumor-segmentation
```

---

## Challenges & Considerations

- High variability of tumor morphology
- Class imbalance across subregions
- Dependence on modality visibility
- Computational cost of 3D models

---

## Future Work

- SwinUNETR / UNETR backbone (transformer-based)
- MONAI Deploy inference pipeline
- External validation against BraTS data
- Uncertainty estimation & calibration
- Model pruning for deployment

---

## References

- MONAI: https://monai.io/
- Medical Decathlon: http://medicaldecathlon.com/
- BraTS Challenge Publications (IEEE TMI / MICCAI)
- "3D MRI brain tumor segmentation using autoencoder regularization"

---

## License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## Author

**Amr Amer**  
Medical Imaging & Deep Learning Research – 2024

---

