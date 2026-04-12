# Road Damage Detection Training Guide

This guide will walk you through training a YOLOv8 model on the RDD2022 dataset, adapted from the oracl4/RoadDamageDetection implementation.

## Overview

The training pipeline includes:
1. **Dataset Download**: Downloads the RDD2022 dataset (1.6GB)
2. **Data Conversion**: Converts Pascal VOC XML annotations to YOLO format
3. **Training**: Trains YOLOv8s model with optimized hyperparameters
4. **Evaluation**: Validates the trained model on test data

## Classes

The model detects 4 types of road damage:
- **Longitudinal Crack** (Class 0): Linear cracks along road direction
- **Transverse Crack** (Class 1): Linear cracks across road direction  
- **Alligator Crack** (Class 2): Network of interconnected cracks
- **Potholes** (Class 3): Holes and depressions in road surface

## Quick Start (Recommended)

For the simplest training experience:

```bash
# 1. Download and prepare dataset
python download_dataset.py --all

# 2. Convert to YOLO format
python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset

# 3. Train model (interactive)
python training_notebook.py
```

## Step-by-Step Guide

### Step 1: Download RDD2022 Dataset

```bash
# Download dataset only
python download_dataset.py --download

# Download and extract
python download_dataset.py --download --extract

# Download, extract, and download reference model
python download_dataset.py --all
```

This will:
- Download RDD2022.zip (1.6GB) from S3
- Extract to `RDD2022/` directory
- Download reference YOLOv8 model

### Step 2: Convert Dataset to YOLO Format

```bash
# Basic conversion (Japan + India countries)
python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset

# Custom configuration
python convert_dataset.py \
    --dataset-root RDD2022 \
    --output-dir yolo_dataset \
    --countries Japan India \
    --train-ratio 0.9 \
    --background-ratio 0.1
```

This will:
- Convert XML annotations to YOLO format
- Filter to main 4 damage classes
- Split into train/val sets (90/10)
- Create `yolo_dataset/rdd_dataset.yaml`

### Step 3: Train the Model

#### Option A: Interactive Training
```bash
python training_notebook.py
```

#### Option B: Advanced Training with CLI
```bash
# Basic training
python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml

# Custom training parameters
python train_model.py \
    --dataset-yaml yolo_dataset/rdd_dataset.yaml \
    --epochs 100 \
    --batch 32 \
    --device cuda \
    --name My_RDD_Model
```

Training parameters (same as reference implementation):
- **Base Model**: yolov8s.pt
- **Epochs**: 100 (with 5% warmup)
- **Batch Size**: 32
- **Image Size**: 640x640
- **Learning Rate**: Cosine annealing
- **Seed**: 1337

### Step 4: Evaluate the Model

```bash
python evaluate_model.py \
    --model runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt \
    --dataset-yaml yolo_dataset/rdd_dataset.yaml
```

## Expected Results

Based on the reference implementation:

### Dataset Statistics
- **Total Images**: ~18,200
- **Training Images**: ~16,400
- **Validation Images**: ~1,800
- **Classes**: 4 damage types

### Performance Metrics
- **mAP50**: 0.547 (54.7%)
- **mAP50-95**: 0.254 (25.4%)
- **Training Time**: 2-4 hours (GPU) / 8-12 hours (CPU)

### Per-Class Performance (Expected)
- **Longitudinal Crack**: ~50% mAP50
- **Transverse Crack**: ~45% mAP50
- **Alligator Crack**: ~71% mAP50
- **Potholes**: ~52% mAP50

## Using Your Trained Model

After training, use your model for inference:

```python
from road_damage_detector import RoadDamageDetector

# Load your trained model
detector = RoadDamageDetector("runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt")

# Analyze video
results = detector.analyze_video("road_video.mp4")
print(f"Detected {results['total_detections']} damage instances")

# Generate report
detector.generate_report(results, "damage_report.html")
```

Or use the command line:

```bash
python road_damage_detector.py \
    --model runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt \
    --source road_video.mp4 \
    --confidence 0.5
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch size
   python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml --batch 16
   ```

2. **No GPU Available**:
   ```bash
   # Force CPU usage
   python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml --device cpu
   ```

3. **Dataset Not Found**:
   ```bash
   # Make sure you've run the conversion step
   python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset
   ```

4. **Slow Training**:
   ```bash
   # Reduce workers or image size
   python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml --workers 1 --imgsz 416
   ```

### Validation Issues

If validation fails:
```bash
# Check dataset structure
ls -la yolo_dataset/
ls -la yolo_dataset/train/images/
ls -la yolo_dataset/val/labels/

# Validate dataset YAML
cat yolo_dataset/rdd_dataset.yaml
```

## Advanced Configuration

### Custom Training Parameters

You can modify training behavior:

```python
from train_model import RoadDamageTrainer

trainer = RoadDamageTrainer("yolov8m.pt")  # Use medium model

# Custom training
trainer.train_model(
    dataset_yaml="yolo_dataset/rdd_dataset.yaml",
    epochs=200,
    batch=16,
    device="cuda",
    cos_lr=False,  # Disable cosine LR
    mosaic=0.5,    # Reduce mosaic augmentation
)
```

### Resume Training

If training is interrupted:

```bash
# Resume from last checkpoint
python train_model.py \
    --dataset-yaml yolo_dataset/rdd_dataset.yaml \
    --model-path runs/RDD_Training/YOLOv8_RDD_Model/weights/last.pt
```

## File Structure After Training

```
road-damage-detection/
├── RDD2022/                                    # Downloaded dataset
│   └── RDD2022_all_countries/
│       ├── Japan/
│       └── India/
├── yolo_dataset/                              # Converted YOLO dataset
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── rdd_dataset.yaml
└── runs/                                      # Training results
    └── RDD_Training/
        └── YOLOv8_RDD_Model/
            ├── weights/
            │   ├── best.pt                    # Best model weights
            │   └── last.pt                    # Last checkpoint
            ├── train_batch*.jpg               # Training visualizations
            ├── val_batch*.jpg                 # Validation visualizations
            └── results.png                    # Training curves
```

## Next Steps

After successful training:

1. **Test your model** on new road videos
2. **Fine-tune** parameters for your specific use case
3. **Deploy** the model in your application
4. **Share** your results with the community

Happy training! 🚗🛣️
