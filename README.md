# DispNet-OccNet: Unsupervised Light Field Depth Estimation

This is a PyTorch implementation of the paper "Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction" (IEEE TIP).

## Project Structure

```
DispNet-Occnet/
├── models/
│   ├── __init__.py
│   ├── dispnet.py      # DispNet: Disparity Estimation Network
│   └── occnet.py      # OccNet: Occlusion Prediction Network
├── utils/
│   ├── __init__.py
│   ├── losses.py      # Loss functions
│   └── config.py      # Configuration utilities
├── data/
│   └── lightfield_dataset.py  # Light field data loader
├── configs/
│   ├── dense_lf.yaml   # Dense light field configuration
│   └── sparse_lf.yaml  # Sparse light field configuration
├── train.py             # Training script
├── inference.py         # Inference script
├── evaluate.py         # Evaluation script
└── requirements.txt    # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Dense Light Field:
```bash
python train.py --data_dir ./data/hci --batch_size 4 --epochs 500
```

Sparse Light Field:
```bash
python train.py --data_dir ./data/slf --batch_size 4 --epochs 500
```

## Inference

```bash
python inference.py --model_path checkpoints/best_model.pth --data_dir ./data/hci/test
```

## Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir ./data/hci/test --gt_dir ./data/hci/gt
```

## Datasets

- **HCI Dataset**: http://lightfield-analysis.net/dataset/
- **DLF Dataset**: From the paper "A framework for learning depth from a flexible subset of dense and sparse light field views"
- **SLF Dataset**: Sparse light field dataset
- **Stanford Lytro Archive**: http://lightfields.stanford.edu/

## Network Architecture

### DispNet
- Feature Extractor with Residual Blocks + ASPP
- Variance-based feature matching for cost volume construction
- Coarse-to-fine disparity estimation
- Soft argmin for disparity regression

### OccNet
- U-Net structure with residual blocks
- Inputs: warped views + disparity map
- Output: confidence maps for occlusion handling

## Loss Functions

- Weighted Photometric Loss (ℓ_wpm)
- Reconstruction Loss (ℓ_rec)
- SSIM Loss (ℓ_SSIM)
- Smoothness Loss (ℓ_smd)
- Occlusion Smoothness Loss (ℓ_smo)

## Citation

If you use this code, please cite:

```
@article{zhang2023unsupervised,
  title={Unsupervised Light Field Depth Estimation via Multi-view Feature Matching with Occlusion Prediction},
  author={Zhang, Shansi and Meng, Nan and Lam, Edmund Y.},
  journal={IEEE Transactions on Image Processing},
  year={2023}
}
```
