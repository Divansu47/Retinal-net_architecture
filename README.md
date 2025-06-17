Traffic Detection Model
This repository contains a PyTorch implementation of a simplified RetinaNet-like object detection model for detecting vehicles in traffic images. The model uses a ResNet18 backbone, focal loss for classification, and DIoU loss for bounding box regression. It is trained and tested on a traffic dataset with annotations in YOLO format (converted to Pascal VOC during processing).
Overview
The project implements a one-stage object detector inspired by RetinaNet, tailored for traffic scenes. It detects vehicles in images, classifies them into left and right lanes, and computes vehicle density. Key modifications from the original RetinaNet include a single feature map level (stride 32), a lighter ResNet18 backbone, and advanced loss functions to improve performance on the traffic dataset.
Features

Model Architecture: Simplified RetinaNet with ResNet18 backbone, single feature map level (stride 32), and anchor-based detection.
Loss Functions:
Focal loss to handle foreground-background class imbalance.
DIoU loss for improved bounding box regression.


Data Augmentation: Uses albumentations with transforms like horizontal flip, color jitter, and coarse dropout for robust training.
Training Optimizations:
Progressive unfreezing of the backbone to stabilize training.
Differential learning rates and cosine annealing scheduler for better convergence.


Inference Pipeline: Detects vehicles, classifies them into left/right lanes, computes density, and saves results to a file.

Prerequisites
Dependencies
The project requires the following Python libraries:

Python 3.8 or higher
PyTorch 1.9 or higher
torchvision
albumentations
numpy
tqdm
PIL (Pillow)

Install the dependencies using the provided requirements.txt:
pip install -r requirements.txt

Imports in traffic_detection.py
The script uses the following imports:
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou, nms, sigmoid_focal_loss
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import torchvision
import math

Dataset
The model is trained and tested on the traffic_wala_dataset, structured as follows:
traffic_wala_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── test/
│   ├── images/
│   │   ├── test_image1.jpg
│   │   ├── test_image2.jpg
│   │   └── ...


Images: JPEG images of traffic scenes.
Labels: Text files in YOLO format (class_id center_x center_y width height), with coordinates normalized between 0 and 1.

Downloading the Dataset
The dataset exceeds GitHub's file size limits and is hosted externally. Download it from one of the following sources:
Option 1: Kaggle (Recommended)

Dataset URL: https://www.kaggle.com/your-username/traffic-wala-dataset (Replace with your actual Kaggle dataset URL)
Using Kaggle API:
Install the Kaggle API:pip install kaggle


Set up your Kaggle API credentials by placing your kaggle.json file in ~/.kaggle/ (see Kaggle API documentation).
Download and unzip the dataset:kaggle datasets download -d your-username/traffic-wala-dataset -p ./traffic_wala_dataset
unzip traffic_wala_dataset/traffic-wala-dataset.zip -d traffic_wala_dataset





Option 2: Google Drive

Download Link: https://drive.google.com/drive/folders/your-folder-id (Replace with your actual Google Drive link)
Using gdown:
Install gdown:pip install gdown


Download and unzip the dataset:gdown https://drive.google.com/uc?id=your-folder-id -O traffic_wala_dataset.zip
unzip traffic_wala_dataset.zip -d traffic_wala_dataset





Update Dataset Paths
Update the paths in traffic_detection.py to match your local directory:
TRAIN_IMG_DIR = "traffic_wala_dataset/train/images"
TRAIN_LABEL_DIR = "traffic_wala_dataset/train/labels"
TEST_IMG_DIR = "traffic_wala_dataset/test/images"

Model Architecture
The model is a simplified version of RetinaNet, designed for efficiency on traffic scenes. Below is a diagram of the architecture:
Input Image (640x640x3)
        |
        v
+-----------------------+
| ResNet18 Backbone     |
| (Pretrained, Frozen   |
|  initially, Stride 32)|
| Output: 20x20x512     |
+-----------------------+
        |
        v
+-----------------------+
| Neck (1x1 Conv)       |
| 512 -> 256 channels   |
| Output: 20x20x256     |
+-----------------------+
        |-----------------+-----------------|
        v                 v                 v
+----------------+  +----------------+  +----------------+
| Classification  |  | Regression     |  | Anchor         |
| Head            |  | Head            |  | Generator      |
| - 3x3 Conv, ReLU|  | - 3x3 Conv, ReLU|  | - Stride: 32   |
| - 3x3 Conv, ReLU|  | - Dropout (0.5) |  | - Scales: 128, |
| - Dropout (0.5) |  | - 1x1 Conv      |  |   256, 512     |
| - 1x1 Conv      |  | Output: 3600x4  |  | - Ratios: 0.5, |
| Output: 3600x1  |  | (dx, dy, dw, dh)|  |   1.0, 2.0     |
+----------------+  +----------------+  +----------------+
        |                 |
        v                 v
+----------------+  +----------------+
| Focal Loss     |  | DIoU Loss      |
| (alpha=0.25,   |  | (Weighted 0.5) |
|  gamma=2.0)    |  |                |
+----------------+  +----------------+

Design Choices and Rationale

Single Feature Level (Stride 32):

Logic: Unlike the original RetinaNet, which uses a Feature Pyramid Network (FPN) with multiple levels (strides 8 to 128), this model uses a single feature map with stride 32. This simplifies the architecture, reducing computational cost, which is suitable for the traffic dataset where vehicles are typically medium to large in size.
Benefit: Faster training and inference, lower memory usage.


ResNet18 Backbone:

Logic: The original RetinaNet uses ResNet50. We opted for ResNet18 to make the model lightweight, enabling faster training on limited hardware while still leveraging pretrained features from ImageNet.
Benefit: Reduces computational overhead while maintaining good feature extraction.


Progressive Unfreezing:

Logic: The backbone is frozen for the first 10 epochs, allowing the classification and regression heads to adapt using pretrained features. It is unfrozen after epoch 10 for end-to-end fine-tuning, enabling adaptation to traffic scenes without catastrophic forgetting.
Benefit: Stabilizes training and improves convergence.


Dropout in Detection Heads:

Logic: nn.Dropout(0.5) is added to both heads to prevent overfitting, especially since the traffic dataset may be smaller than typical detection datasets like COCO.
Benefit: Encourages robust feature learning.


DIoU Loss for Regression:

Logic: Replaces the smooth L1 loss used in RetinaNet with DIoU loss, which considers both IoU and the distance between box centers, leading to better localization.
Benefit: Faster convergence and improved accuracy for bounding box predictions.


Custom Anchor Scales:

Logic: Anchor scales [128, 256, 512] and aspect ratios [0.5, 1.0, 2.0] are tailored for vehicles, which are typically larger objects in traffic scenes.
Benefit: Better matches the dataset’s object sizes, improving detection performance.



Workflow

Setup Environment:

Clone the repository and install dependencies.
Download the dataset and update paths in traffic_detection.py.


Train the Model:

Run traffic_detection.py to train the model for 20 epochs.
Checkpoints are saved after each epoch (e.g., checkpoint_epoch_20.pth).
Training progress (losses and positive anchors) is logged to the console.


Inference:

After training, the script automatically runs inference on the test dataset.
Results for each test image are saved to output.txt, including:
Total vehicle detections.
Vehicles in left lane (x < 320).
Vehicles in right lane (x ≥ 320).
Vehicle density (vehicles per pixel).




Optional: Run Inference Only:

Load a pretrained checkpoint and run inference on new test images by modifying the script.



Usage

Clone the Repository:
git clone https://github.com/your-username/traffic-detection.git
cd traffic-detection


Install Dependencies:
pip install -r requirements.txt


Prepare the Dataset:

Download the dataset as described above.
Update paths in traffic_detection.py if necessary.


Train and Run Inference:
python traffic_detection.py


Training runs for 20 epochs, saving checkpoints after each epoch.
Inference runs automatically on the test dataset, saving results to output.txt.

Example Output in output.txt:
Image: test_image1.jpg
Saved 29 vehicle detections to: output.txt
Vehicles in Left Lane: 17
Vehicles in Right Lane: 12
Vehicle Density: 0.000071 vehicles per pixel
--------------------------------------------------


Run Inference Only:To run inference with a pretrained checkpoint:
# Comment out the training loop in traffic_detection.py
start_epoch = load_checkpoint(model, optimizer, filename="checkpoint_epoch_20.pth")
inference(model, test_loader, device, output_file="output.txt")



Training Details

Batch Size: 4
Optimizer: Adam with weight decay (1e-5)
Learning Rates:
Backbone: 1e-5
Classification Head: 1e-3
Other Parameters: 1e-4


Scheduler: CosineAnnealingLR (T_max=20)
Anchor Configuration:
Stride: 32
Scales: [128, 256, 512]
Aspect Ratios: [0.5, 1.0, 2.0]


Loss:
Classification: Focal Loss (alpha=0.25, gamma=2.0)
Regression: DIoU Loss (weighted by 0.5)



Results
After training for 20 epochs, the model achieved:

Classification Loss: ~0.5
Regression Loss: ~0.5These results indicate good convergence for the traffic detection task.

Future Improvements

Add a validation loop to monitor performance (e.g., mAP).
Optimize anchor boxes using K-Means clustering.
Extend to multi-class detection (e.g., cars, trucks).
Incorporate a Feature Pyramid Network (FPN) for multi-scale detection.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Inspired by the RetinaNet paper: Focal Loss for Dense Object Detection (Lin et al., 2017).
Thanks to the PyTorch and albumentations communities for their excellent libraries.

