# Defect Detection on Cast

This project implements an automated quality control system for detecting defects in cast industrial parts using the **YOLOv8** object detection framework. It provides a complete pipeline from model training on custom datasets to real-time inference on images and videos.

## 🚀 Features
- **Automated Defect Classification**: Detects and classifies specific defects such as holes and scratches.
- **Multi-Mode Inference**: 
  - **Image Detection**: Analyze static images for defects.
  - **Video Analysis**: Process video streams to track and count defects over time.
  - **Real-Time Detection**: Capabilities for live camera feed processing.
- **YOLOv8 Integration**: Leverages the state-of-the-art Ultralytics YOLOv8 architecture for high accuracy and speed.

## 🛠️ Project Structure
```text
.
├── Build_Model.py          # Script to train the YOLOv8 model
├── Detect_On_Image.py       # Inference script for single images
├── Detect_On_Video.py       # Inference script for video files with defect counting
├── Real_Time_Detection.py    # Script for real-time camera feed detection
├── data.yaml               # Dataset configuration file (paths and class names)
├── yolov8n.pt              # Pre-trained YOLOv8 nano weights
├── train/                  # Training dataset (images and labels)
├── val/                    # Validation dataset (images and labels)
└── runs/                   # Training logs and trained weights (.pt files)
```

## 📊 Dataset Overview
The model is trained on a custom dataset with the following classes:
- `Defective`: General defective part
- `cast`: The cast part itself
- `hole`: Specific hole defect
- `scratches`: Surface scratch defects

## ⚙️ Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python
```

## 📖 Usage

### Training the Model
To train the model from scratch or fine-tune using the provided dataset:
```bash
python Build_Model.py
```
The trained weights will be saved in `runs/detect/train/weights/best.pt`.

### Detecting Defects on an Image
To run inference on a specific image:
```bash
python Detect_On_Image.py
```

### Detecting Defects on a Video
To analyze a video and generate a defect report:
```bash
python Detect_On_Video.py
```

## 📈 Results
Training metrics (Confusion Matrix, PR Curve, etc.) can be found in the `runs/detect/train/` directory after running the training script.
