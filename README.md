# Bias Analysis in Gender Classification using UTKFace Dataset

**Team A - Gender Classification**

A comprehensive analysis of potential bias in gender classification models using explainable AI techniques (LIME, SHAP, TCAV, ACE) on the UTKFace dataset.

## 📁 Directory Structure

```
├── README.md                    # This file
├── project_report.tex          # Main LaTeX report
├── train_config.yaml          # Training configuration
├── utils.py                   # Utility functions and helper classes
│
├── 📊 Analysis Notebooks
│   ├── data_analysis.ipynb        # Dataset balance and demographic analysis
│   ├── train_resnet.ipynb        # ResNet-18 training notebook
│   ├── lime_and_shap.ipynb       # LIME and SHAP explanations
│   └── tcav_and_ace.ipynb        # TCAV and ACE concept analysis
│
├── 🔧 Preprocessing
│   └── preprocess/
│       ├── crop_and_align.py     # Face detection and alignment using dlib
│       └── crop_and_align_util.py # Utility functions for preprocessing
│
├── 📂 Data Directories
│   ├── train_data/               # Raw and processed training data
│   │   ├── part1/, part2/, part3/     # Raw UTKFace images
│   │   └── part1_cropped/, etc.       # Dlib-aligned face crops (224×224)
│   ├── tcav_data/               # Concept data for TCAV analysis
│   │   └── concept/             # Human-defined concepts (beard, glasses, etc.)
│   └── cav/                     # Concept Activation Vectors storage
│
├── 📈 Results & Output
│   ├── train_output/            # Training results and model checkpoints
│   │   ├── best_resnet18_gender.pth   # Best trained model
│   │   ├── test_predictions.csv       # Test set predictions with metadata
│   │   └── test_results.txt          # Performance metrics by demographics
│   └── exp_output/              # Explainability analysis results
│       ├── lime_and_shap_explanations/ # LIME/SHAP visualizations by subgroup
│       ├── tcav_and_ace_plots/         # TCAV scores and ACE concept plots
│       └── ace_concept/               # Auto-discovered ACE concepts
│
└──  archive/       # Archived files
```

## 🚀 Quick Start

Before starting, download the UTKFace Dataset from [here](https://susanqq.github.io/UTKFace/). It has three folders. Save all those folders in a folder called train_data in the root directory.

Then create an environment using the given environment.yml file.

```bash
conda env create -f environment.yml
```

### 1. Preprocessing
Crop and align face images using dlib's 5-point landmark predictor:

```bash
# Download dlib landmark predictor
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
bunzip2 shape_predictor_5_face_landmarks.dat.bz2

# Run face alignment (modify paths in script as needed)
python preprocess/crop_and_align.py
```

### 2. Data Analysis
Analyze dataset balance and demographics:
```bash
data_analysis.ipynb
```

### 3. Model Training
Train ResNet-18 for gender classification:
```bash
# Configure training parameters in train_config.yaml
train_resnet.ipynb
```

### 4. Explainability Analysis
Generate explanations using multiple XAI techniques:

**LIME & SHAP:**
```bash
lime_and_shap.ipynb
```

**TCAV & ACE:**
```bash
tcav_and_ace.ipynb
```





