# Traffic Detection Model

This repository contains a PyTorch implementation of a RetinaNet-like object detection model for detecting traffic objects (e.g., vehicles) in images. The model uses a ResNet18 backbone, focal loss for classification, and DIoU loss for bounding box regression. It is trained on a traffic dataset with Pascal VOC format annotations.

## Features
- **Model Architecture**: Simplified RetinaNet with ResNet18 backbone, a single feature map level (stride 32), and anchor-based detection.
- **Loss Functions**:
  - Focal loss to handle foreground-background class imbalance.
  - DIoU loss for improved bounding box regression.
- **Data Augmentation**: Uses `albumentations` for robust training with transforms like horizontal flip, color jitter, and coarse dropout.
- **Training Optimizations**: Cosine annealing learning rate scheduler, differential learning rates, and backbone unfreezing for better convergence.

## Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- torchvision
- albumentations
- numpy
- tqdm
- PIL (Pillow)

You can install the dependencies using the following command:
```bash
pip install torch torchvision albumentations numpy tqdm pillow
```

## Dataset
The model is trained on a traffic dataset with the following structure:
```
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
```
- **Images**: JPEG images of traffic scenes.
- **Labels**: Text files in YOLO format (`class_id center_x center_y width height`), where coordinates are normalized between 0 and 1 relative to the image dimensions.

**Note**: The dataset path in the script points to a Kaggle dataset (`/kaggle/input/traffic-data/traffic_wala_dataset`). Update the `TRAIN_IMG_DIR` and `TRAIN_LABEL_DIR` variables in `traffic_detection.py` to match your local dataset path.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/traffic-detection.git
   cd traffic-detection
   ```

2. **Prepare the Dataset**:
   - Place your dataset in the appropriate directory.
   - Update the paths in `traffic_detection.py`:
     ```python
     TRAIN_IMG_DIR = "path/to/your/train/images"
     TRAIN_LABEL_DIR = "path/to/your/train/labels"
     ```

3. **Train the Model**:
   Run the training script:
   ```bash
   python traffic_detection.py
   ```
   - The script trains the model for 20 epochs.
   - The backbone (ResNet18) is unfrozen after 10 epochs to fine-tune features.
   - Training progress, including classification and regression losses, is printed per epoch.
   - The number of positive anchors per image is logged for debugging.

4. **Inference**:
   The current script focuses on training. To perform inference, you can extend the code by adding a post-processing step (e.g., NMS) and a prediction function. Refer to the `decode_boxes` function for decoding regression outputs.

## Training Details
- **Batch Size**: 4
- **Optimizer**: Adam with weight decay (1e-5)
- **Learning Rates**:
  - Backbone: 1e-5
  - Classification Head: 1e-3
  - Other Parameters: 1e-4
- **Scheduler**: CosineAnnealingLR over 20 epochs
- **Anchor Configuration**:
  - Stride: 32
  - Scales: [128, 256, 512]
  - Aspect Ratios: [0.5, 1.0, 2.0]
- **Loss**:
  - Classification: Focal Loss (alpha=0.25, gamma=2.0)
  - Regression: DIoU Loss (weighted by 0.5)

## Results
After training for 20 epochs, the classification loss was reduced to approximately 0.5, and the regression loss stabilized around 0.5, indicating good convergence for the traffic detection task.

## Future Improvements
- Add a validation loop to monitor performance on a validation set.
- Implement inference and evaluation scripts (e.g., mAP calculation).
- Optimize anchor boxes using K-Means clustering on the dataset.
- Extend to multi-class detection for different types of vehicles.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- This implementation is inspired by the RetinaNet paper: *Focal Loss for Dense Object Detection* (Lin et al., 2017).
- Thanks to the PyTorch and albumentations communities for their excellent libraries.
