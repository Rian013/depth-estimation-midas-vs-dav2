# Monocular Depth Estimation
### Comparing MiDaS and Depth Anything V2 with 3D Point Cloud Visualization

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?style=flat-square&logo=pytorch)
![Open3D](https://img.shields.io/badge/Open3D-0.17-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Ubuntu%2022.04-orange?style=flat-square&logo=ubuntu)
![CPU Only](https://img.shields.io/badge/Hardware-CPU%20Only-lightgrey?style=flat-square)

> CS 5330 Computer Vision | Northeastern University | Rian Archie Fernandes | April 2026

---

## Overview

This project implements and compares two state-of-the-art monocular depth estimation models, **MiDaS Small** and **Depth Anything V2 Small**, on real-world indoor and outdoor scenes. Monocular depth estimation predicts per-pixel depth from a single RGB image with no stereo camera or LiDAR required.

The pipeline is evaluated qualitatively across three scenes and quantitatively on **50 samples from the NYU Depth V2 benchmark**. Estimated depth maps are further projected into colored **3D point clouds using Open3D**, connecting the work to robotics and SLAM applications.

The entire pipeline runs on CPU with no GPU required.

---

## Key Results

| Model | AbsRel (lower = better) | RMSE (lower = better) |
|---|---|---|
| MiDaS Small | **0.55** | **1.82** |
| Depth Anything V2 Small | 0.65 | 2.19 |

**Key finding:** MiDaS wins on the NYU indoor benchmark due to domain-specific training. However, Depth Anything V2 produces **60x sharper depth maps** by Laplacian variance (96.43 vs 1.58), consistently outperforming MiDaS on visual quality across all scenes. This highlights the well-known gap between benchmark metrics and perceptual quality in depth estimation.

---

## Results Gallery

### Side by Side Depth Map Comparison
| Scene | Result |
|---|---|
| Outdoor bench | Original / MiDaS / Depth Anything V2 |
| Acorn Street, Boston | Original / MiDaS / Depth Anything V2 |
| Indoor living room | Original / MiDaS / Depth Anything V2 |

### 3D Point Cloud Visualization
Depth maps back-projected into colored 3D point clouds using Open3D pinhole camera model.

---

## Project Structure

```
depth_estimation_project/
├── src/
│   ├── depth_estimator.py      # MiDaS inference pipeline
│   ├── depth_anything.py       # Depth Anything V2 inference pipeline
│   ├── compare.py              # Side by side comparison figures
│   ├── pointcloud.py           # Open3D 3D point cloud generation
│   ├── benchmark.py            # NYU Depth V2 quantitative evaluation
│   ├── experiments.py          # Full experiment analysis and figures
│   └── download_nyu.py         # NYU Depth V2 dataset download utility
├── images/                     # Input images
├── outputs/                    # Generated depth maps, point clouds, figures
├── data/                       # NYU Depth V2 dataset (not tracked in git)
├── readme.txt                  # Gradescope submission readme
└── README.md                   # This file
```

---

## Setup

**Step 1 - Clone the repo**
```bash
git clone https://github.com/Rian013/depth-estimation-midas-vs-dav2.git
cd depth-estimation-midas-vs-dav2
```

**Step 2 - Create and activate virtual environment**
```bash
python3 -m venv depth_env
source depth_env/bin/activate
```

**Step 3 - Install dependencies**
```bash
pip install torch torchvision opencv-python open3d matplotlib numpy
pip install transformers huggingface_hub timm scipy h5py pillow
```

---

## How to Run

**Run MiDaS depth estimation**
```bash
cd src
python depth_estimator.py
```

**Run Depth Anything V2 depth estimation**
```bash
python depth_anything.py
```

**Generate side by side comparison figures**
```bash
python compare.py
```

**Generate 3D point clouds**
```bash
python pointcloud.py
```

**Download NYU Depth V2 dataset and run benchmark**
```bash
python download_nyu.py
python benchmark.py
```

**Run full experiment analysis (all 4 figures)**
```bash
python experiments.py
```

All outputs are saved to the `outputs/` folder.

---

## Pipeline

```
RGB Image
    |
    v
Preprocessing (OpenCV, BGR to RGB)
    |
    v
Depth Prediction (MiDaS via PyTorch Hub / DAV2 via Hugging Face)
    |
    v
Bicubic Upsampling to original resolution
    |
    v
Scale Alignment (least squares: s = p.g / p.p)
    |
    v
Evaluation (AbsRel, RMSE on NYU Depth V2)
    |
    v
3D Back-Projection (pinhole model: x=(u-cx)*z/fx, y=(v-cy)*z/fy)
    |
    v
Open3D Point Cloud (.ply)
```

---

## Models

**MiDaS Small** (Intel ISL, ICCV 2021)
- EfficientNet encoder-decoder backbone
- Scale and shift invariant loss for mixed dataset training
- Loaded via PyTorch Hub
- ~2-3 seconds per image on CPU

**Depth Anything V2 Small** (NeurIPS 2024)
- Vision Transformer (ViT) backbone
- Trained on massive labeled and pseudo-labeled data
- Loaded via Hugging Face Transformers
- ~3-5 seconds per image on CPU

---

## Experiments

Four structured experiment figures are generated by `experiments.py`:

- **Fig 1** - Full 3x3 comparison grid across all 3 scenes
- **Fig 2** - Indoor vs outdoor analysis with absolute difference maps
- **Fig 3** - Edge sharpness zoom analysis on Acorn Street, Boston
- **Fig 4** - Numerical bar charts comparing Laplacian variance and depth contrast

---

## Dependencies

```
Python 3.10
PyTorch
torchvision
opencv-python
open3d
matplotlib
numpy
transformers
huggingface_hub
timm
scipy
h5py
Pillow
```

---

## Hardware

- OS: Ubuntu 22.04
- CPU: Intel Core i5
- GPU: None (CPU only)
- RAM: 8GB

---

## Dataset

**NYU Depth V2** (Silberman et al., ECCV 2012)
- 1449 densely labeled RGB-D pairs from Microsoft Kinect
- 464 indoor scenes across 3 US cities
- Ground truth depth in meters
- Official source: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
- 50 validation samples used for benchmarking
- Dataset not tracked in this repo due to size (2.8GB)

---

## References

1. R. Ranftl, A. Bochkovskiy, and V. Koltun, "Vision Transformers for Dense Prediction," ICCV 2021.
2. L. Yang et al., "Depth Anything V2," NeurIPS 2024.
3. N. Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images," ECCV 2012.
4. D. Eigen, C. Puhrsch, and R. Fergus, "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network," NeurIPS 2014.

---

## Author

**Rian Archie Fernandes**
MS Robotics Engineering, Northeastern University
fernandes.ri@northeastern.edu

---

*CS 5330 Computer Vision | Khoury College of Computer Sciences | April 2026*
