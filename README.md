
# Deep Learning Enable Precision Authentication of Seasonal and Processing Signatures in Tieguanyin Tea

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
*(Note: Update the license badge based on your target journal's requirements)*

This repository contains the dataset and source code for the paper: **"Deep Learning Enable Precision Authentication of Seasonal and Processing Signatures in Tieguanyin Tea."** It provides a complete framework for transforming liquid chromatography–mass spectrometry (LC–MS) metabolomic data into image representations, enabling robust and highly accurate authentication of tea products even under severe chromatographic drift.

## 📖 Abstract
Authenticating specialty tea products remains a critical challenge in premium food markets, yet current analytical approaches are constrained by limited reproducibility and susceptibility to instrumental variation. Here, we present a deep learning framework that transforms LC–MS metabolomic data into image representations, enabling robust authentication of tea products under real-world analytical conditions. 

Profiling 274 Tieguanyin tea samples across seasonal harvests (spring and autumn) and processing methods (light-scented and strong-scented), our approach achieved **90.9%** (95% CI: 80.4%–96.0%) classification accuracy—substantially outperforming conventional multivariate and machine learning methods (sPLS-DA: 85.5%; random forest: 87.3%). Critically, when subjected to chromatographic drift—a pervasive source of analytical irreproducibility—our model maintained **78.2%** accuracy while traditional methods degraded to 69.1%. This framework addresses fundamental limitations in untargeted metabolomics, offering a generalizable solution for food authentication that extends beyond tea to broader applications in agricultural product verification and systems biology.

---

## 📂 Repository Structure

```text
├── code/                           # Scripts for modeling and data processing
│   ├── Data augmentation.R         # R script for generating simulated drift/shifts
│   ├── Deep learning.py            # Main deep learning framework and training script
│   ├── Random forest.py            # Baseline machine learning comparison
│   └── sPLS-DA.R                   # Baseline multivariate statistical comparison
├── image-part1.zip                 # Compressed metabolic image datasets
├── image-part2.zip                 # Compressed metabolic image datasets
└── README.md                       # Repository documentation

```


## 📊 Dataset Description (`image.zip`)
The `image.zip` archive contains all LC-MS metabolomic data transformed into 2D image representations. The dataset is organized by image resolution and simulated analytical drift conditions to test model robustness.

### Naming Convention
Folders are named using the format: `image_[resolution]_[condition]`
- **Resolution:** 224, 448, or 1024 pixels.
- **Condition:** `raw` (original data) or `shift` (data subjected to simulated chromatographic and mass spectrometry variations).
    - `rt`: Retention time drift parameters.
    - `mz`: Mass-to-charge ratio drift parameters.
    - `int`: Intensity variation parameters.
    
### Included Folders:
**224x224 Resolution:**
- `image_224_raw`
- `image_224_shift_rt5_1_mz5_1_int1_0.2`
- `image_224_shift_rt5_1_mz10_2_int0_0`
- `image_224_shift_rt10_2_mz5_1_int0_0`
- `image_224_shift_rt20_5_mz0_0_int0.5_0.1`
- `image_224_shift_rt20_5_mz10_2_int0_0`

**448x448 Resolution:**
- `image_448_raw`
- `image_448_shift_rt5_1_mz5_1_int1_0.2`
- `image_448_shift_rt5_1_mz10_2_int0_0`
- `image_448_shift_rt10_2_mz5_1_int0_0`
- `image_448_shift_rt20_5_mz0_0_int0.5_0.1`
- `image_448_shift_rt20_5_mz10_2_int0_0`

**1024x1024 Resolution:**
- `image_1024_raw`
- `image_1024_shift_rt5_1_mz5_1_int1_0.2`
- `image_1024_shift_rt5_1_mz10_2_int0_0`
- `image_1024_shift_rt10_2_mz5_1_int0_0`
- `image_1024_shift_rt20_5_mz0_0_int0.5_0.1`
- `image_1024_shift_rt20_5_mz10_2_int0_0`

**Testing Data:**
- `RT drift dataset for test`: A dedicated validation set containing simulated retention time drifts for final model evaluation.

---


## 💻 Code Description
The `code/` directory contains all necessary scripts to reproduce the study's findings:

1. **`Data augmentation.R`**: R script used to apply retention time, m/z, and intensity shifts to the raw LC-MS data, generating the augmented datasets used to train and test model robustness.
2. **`Deep learning.py`**: Python script containing the core deep learning architecture. Handles image loading, model training, validation, and performance evaluation on the metabolic images.
3. **`Random forest.py`**: Python script for training and evaluating the Random Forest classifier (baseline comparison model).
4. **`sPLS-DA.R`**: R script utilizing the sparse Partial Least Squares Discriminant Analysis (sPLS-DA) algorithm for baseline multivariate statistical comparison.
