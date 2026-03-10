# Thermal MOT - OATM+TGF Open Source Project

## Project Overview

This is the official open-source release of the **Thermal MOT (Multi-Object Tracking)** system using **OATM+TGF** (Optical-Aware Thermal Matching + Thermal Gradient Flow).

The system achieves state-of-the-art performance on thermal multi-object tracking tasks with:
- **MOTA**: 85.04%
- **MOTP**: 12.33%
- **IDF1**: 86.21%


### Core Scripts
- `scripts/generate_yolo_detections_v2.py` - YOLO detection generation
- `scripts/run_tracking_oatm_tgf_v2.py` - Main tracking pipeline
- `scripts/run_tracking_custom.py` - Custom tracking runner
- `scripts/interpolate_results.py` - Result interpolation with fixed optimal parameters
- `scripts/evaluate_submission.py` - Submission format validation
- `scripts/train_yolo_detector.py` - YOLO detector training
- `scripts/train_reid_model.py` - ReID model fine-tuning

### Configuration Files
- `configs/mot/bytetrack/bytetrack_reid_oatm_tgf_v2.py` - OATM+TGF configuration
- `configs/mot/bytetrack/bytetrack_reid_swin_l.py` - Base ByteTrack configuration

### Documentation
- `README.md` - This file
- `INSTALL.md` - Installation guide
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Installation

```bash
git clone <repo-url>
cd MOT_open
pip install -r requirements.txt
```

See [INSTALL.md](INSTALL.md) for detailed setup instructions.

### 2. Inference

Generate detections and run tracking:

```bash
# Step 1: Generate YOLO detections
python scripts/generate_yolo_detections_v2.py \
    --weights checkpoints/yolo_detector_best.pt \
    --source /path/to/images/val \
    --output-dir ./yolo_detections \
    --conf-thres 0.01

# Step 2: Run OATM+TGF v2 tracking
python scripts/run_tracking_oatm_tgf_v2.py \
    --detections-dir ./yolo_detections \
    --output-dir ./tracking_results \
    --reid-checkpoint checkpoints/reid_model_best.pth
```

### 3. Training (Optional)

Train your own YOLO detector:

```bash
python scripts/train_yolo_detector.py \
    --data /path/to/data.yaml \
    --weights yolov11s.pt \
    --epochs 1000 \
    --batch 16 \
    --device 0
```

Fine-tune ReID model:

```bash
python scripts/train_reid_model.py \
    --config configs/reid/resnet50_b32x8_VTMOT.py \
    --checkpoint /path/to/pretrained.pth \
    --work-dir ./work_dirs/reid
```

## Configuration

### OATM+TGF v2 Parameters

Key parameters in `configs/mot/bytetrack/bytetrack_reid_oatm_tgf_v2.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reid_alpha` | 0.45 | ReID feature weight |
| `oatm_alpha` | 0.22 | OATM matching weight |
| `tgf_alpha` | 0.12 | TGF feature weight |
| `center_ratio` | 0.6 | Center-based matching ratio |
| `gradient_bins` | 8 | Gradient bins for thermal features |
| `gaussian_sigma` | 1.0 | Gaussian smoothing sigma |
| `adaptive_tgf` | True | Enable adaptive TGF |


## Acknowledgments

- Built on [mmtracking](https://github.com/open-mmlab/mmtracking)
- Uses [YOLOv11](https://github.com/ultralytics/ultralytics)
- Inspired by [ByteTrack](https://github.com/ifzhang/ByteTrack)

## Pretrained checkpoint
Download: https://drive.google.com/drive/folders/1o60V-PumGyNP3yYFQmUthG4NjXPzG6Ce?usp=sharing

---

