# Layout Estimation Network

A deep learning model for estimating 3D room layout from single RGB images. This project implements a Layout Estimation Network (LEN) as part of a larger 3D pose estimation system.

## Overview

The Layout Estimation Network (LEN) takes a single RGB image as input and predicts:
- 3D room layout parameters (centroid, size, orientation)
- Camera parameters (pitch and roll)

The model uses a ResNet backbone (ResNet18/34/50) with custom fully connected layers to predict the layout and camera parameters.

## Project Structure

```
layout_estimation/
├── configs/                # Configuration files
│   └── config.yaml         # Main config file
├── data/                   # Dataset handling
│   ├── __init__.py
│   └── dataset.py          # Pix3D dataset implementation
├── models/                 # Neural network models
│   ├── __init__.py
│   └── layout_estimation.py # Layout Estimation Network
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Visualization tools
├── train.py                # Training script
├── test.py                 # Testing script
├── demo.py                 # Demo script
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
└── README.md               # Project documentation
```

## Installation

### Option 1: Using conda

```bash
# Clone the repository
git clone https://github.com/your-username/layout_estimation.git
cd layout_estimation

# Create and activate conda environment
conda env create -f environment.yml
conda activate LayoutEstimation
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/layout_estimation.git
cd layout_estimation

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/config.yaml --device mps
```

### Testing

```bash
python test.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --device mps
```

### Demo

```bash
python demo.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.jpg --device mps
```

## Configuration

The `configs/config.yaml` file contains all the configuration parameters for the model, training, and testing. You can modify this file to customize the behavior of the model.

Key configuration sections:
- `model`: Model architecture settings
- `train`: Training parameters
- `dataset`: Dataset configuration
- `loss`: Loss function weights
- `test`: Testing and demonstration settings

## Dataset

This project uses the Pix3D dataset for training and evaluation. The dataset contains RGB images of indoor scenes with corresponding 3D layout annotations.

To set up the dataset:
1. Download the Pix3D dataset from the [official website](http://pix3d.csail.mit.edu/)
2. Extract the dataset to a directory
3. Update the `dataset.root_dir` in the config file to point to the dataset directory

## Model Architecture

The Layout Estimation Network (LEN) consists of:
1. A ResNet backbone (ResNet18/34/50) for feature extraction
2. A fully connected network for layout parameter estimation
3. A fully connected network for camera parameter estimation

The model outputs:
- Layout parameters (7 values):
  - Centroid coordinates (x, y, z)
  - Room size dimensions (width, height, depth)
  - Room orientation angle
- Camera parameters (2 values):
  - Camera pitch angle
  - Camera roll angle

## Evaluation Metrics

The model is evaluated using the following metrics:
- 3D IoU (Intersection over Union) for layout estimation
- Centroid error (Euclidean distance)
- Size error (Euclidean distance)
- Orientation error (angular difference)
- Camera parameter errors (pitch and roll)

## License

[MIT License](LICENSE)

## Acknowledgments

This project is based on the following papers:
- "Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image" (CVPR 2020)
- "Holistic 3D Scene Understanding from a Single Image with Implicit Representation" (CVPR 2021)