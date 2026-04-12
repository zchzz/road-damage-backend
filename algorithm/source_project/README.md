# Road Damage Detection

An AI-powered system for detecting road damage in videos using YOLOv8. This system can analyze both YouTube videos and local video files to identify various types of road damage including potholes, cracks, patches, and other road surface issues.

## Features

- 🎥 **YouTube Video Analysis**: Download and analyze videos directly from YouTube URLs
- 📁 **Local Video Analysis**: Process local video files for damage detection
- 🤖 **YOLO Integration**: Uses YOLOv8 for accurate object detection
- 📊 **Detailed Reports**: Generate HTML reports with analysis results
- 🏋️ **Custom Training**: Train your own models using your video data
- 📈 **Visualization**: Annotated output videos showing detected damage

## Supported Damage Types

- Potholes
- Road cracks
- Patches/repairs
- Manhole covers
- Water damage
- General wear and tear
- Other road damage

## Installation

### Option 1: Using uv (Recommended)

1. Clone this repository:
```bash
git clone <repository-url>
cd road-damage-detection
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Run the automated setup:
```bash
# Linux/macOS
./setup_env.sh

# Windows
setup_env.bat

# Or manually
uv venv .venv
uv pip install -r requirements.txt
```

4. Run the setup and demo:
```bash
uv run python setup_and_demo.py
```

### Option 2: Traditional pip

1. Clone this repository:
```bash
git clone <repository-url>
cd road-damage-detection
```

2. Create virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Run the setup and demo:
```bash
python setup_and_demo.py
```

## Quick Start

### Using uv (Recommended)
```bash
# Analyze YouTube video (uses default channel)
uv run python setup_and_demo.py --youtube

# Analyze specific YouTube video
uv run python setup_and_demo.py --youtube "https://youtube.com/watch?v=VIDEO_ID"

# Analyze local video
uv run python setup_and_demo.py --local /path/to/video.mp4

# Or use dedicated scripts
uv run python analyze_youtube.py "https://youtube.com/watch?v=VIDEO_ID"
uv run python analyze_local.py /path/to/video.mp4
```

### Traditional Python
```bash
# Analyze YouTube video (uses default channel)  
python setup_and_demo.py --youtube

# Analyze specific YouTube video
python setup_and_demo.py --youtube "https://youtube.com/watch?v=VIDEO_ID"

# Analyze local video
python setup_and_demo.py --local /path/to/video.mp4
```

### Train Custom Model

The project now includes a complete training pipeline adapted from the oracl4/RoadDamageDetection repository. You can train on the RDD2022 dataset with proper class mapping.

#### Option 1: Quick Training (Interactive)
```bash
# With uv
uv run python training_notebook.py

# Traditional
python training_notebook.py
```

#### Option 2: Full Control Training
```bash
# 1. Download RDD2022 dataset
uv run python download_dataset.py --all

# 2. Convert XML annotations to YOLO format
uv run python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset

# 3. Train the model
uv run python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml --epochs 100

# 4. Evaluate the model
uv run python evaluate_model.py --model runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt --dataset-yaml yolo_dataset/rdd_dataset.yaml
```

## Usage Examples

### 1. Basic YouTube Analysis
```python
from road_damage_detector import RoadDamageDetector

detector = RoadDamageDetector()
results = detector.analyze_youtube_video(
    "https://youtube.com/watch?v=VIDEO_ID", 
    confidence=0.3
)
detector.generate_report(results, "report.html")
```

### 2. Local Video Analysis
```python
detector = RoadDamageDetector()
results = detector.analyze_video(
    "video.mp4", 
    output_path="analyzed_video.mp4",
    confidence=0.3
)
print(f"Found {results['total_detections']} damage instances")
```

### 3. Using Custom Trained Model
```python
detector = RoadDamageDetector("path/to/custom_model.pt")
results = detector.analyze_video("video.mp4")
```

## Command Line Options

### YouTube Analysis
```bash
python analyze_youtube.py [URL] [OPTIONS]

Options:
  --model MODEL         Path to custom YOLO model
  --confidence FLOAT    Detection confidence threshold (default: 0.3)
  --output DIR          Output directory (default: youtube_analysis)
```

### Local Video Analysis
```bash
python analyze_local.py [VIDEO_PATH] [OPTIONS]

Options:
  --model MODEL         Path to custom YOLO model
  --confidence FLOAT    Detection confidence threshold (default: 0.3)
  --output PATH         Output video path
  --skip-frames INT     Process every nth frame (default: 5)
```

## Training Your Own Model

The project now includes a complete training pipeline based on the RDD2022 dataset with proper class mapping from the reference implementation.

### Supported Classes (RDD2022 Dataset)
- **Longitudinal Crack**: Linear cracks running along the road direction
- **Transverse Crack**: Linear cracks running across the road direction  
- **Alligator Crack**: Network of interconnected cracks resembling alligator skin
- **Potholes**: Holes and depressions in the road surface

### Training Workflow

#### Method 1: Complete Pipeline (Recommended)

1. **Download and Prepare RDD2022 Dataset**:
```bash
# Download RDD2022 dataset (1.6GB)
python download_dataset.py

# Convert XML annotations to YOLO format and split data
python convert_dataset.py --dataset-root RDD2022 --output-dir yolo_dataset
```

2. **Train the Model**:
```bash
# Train with default settings (adapted from reference implementation)
python train_model.py --dataset-yaml yolo_dataset/rdd_dataset.yaml

# Or customize training parameters
python train_model.py \
    --dataset-yaml yolo_dataset/rdd_dataset.yaml \
    --epochs 100 \
    --batch 32 \
    --device cuda \
    --name Custom_RDD_Model
```

3. **Evaluate the Model**:
```bash
python evaluate_model.py \
    --model runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt \
    --dataset-yaml yolo_dataset/rdd_dataset.yaml
```

#### Method 2: Interactive Training
```bash
# Simple one-command training (checks dataset automatically)
python training_notebook.py
```

#### Method 3: Legacy Video-based Training
1. **Prepare Training Data**:
```bash
python train_model.py
```
This will extract frames from your video in `15-01-2024/C0007.MP4` and create a dataset structure.

2. **Annotate the Data**:
   - Install annotation tool: [LabelImg](https://github.com/tzutalin/labelImg) or use [Roboflow](https://roboflow.com/)
   - Annotate the extracted frames in `dataset/train/images/`
   - Save annotations to `dataset/train/labels/`

3. **Train the Model**:
The training will start automatically after data preparation, or you can customize it:

```python
from train_model import RoadDamageTrainer

trainer = RoadDamageTrainer()
trainer.train_model("dataset/dataset.yaml", epochs=100)
```

### Training Parameters

The training uses the same parameters as the reference implementation:

- **Base Model**: YOLOv8s (Small model, good balance of speed/accuracy)
- **Epochs**: 100 (with 5% warmup epochs)
- **Batch Size**: 32
- **Image Size**: 640x640
- **Learning Rate**: Cosine annealing
- **Seed**: 1337 (for reproducibility)
- **Augmentations**: Mosaic, MixUp, etc.

### Dataset Information

The RDD2022 dataset includes:
- **Japan**: ~10,500 images
- **India**: ~7,700 images  
- **Combined**: ~18,200 images with train/val split (90/10)
- **Classes**: 4 main damage types (filtered from original 9 classes)

### Expected Results

Based on the reference implementation, you can expect:
- **mAP50**: ~0.55-0.60
- **mAP50-95**: ~0.25-0.30
- **Training Time**: 2-4 hours (depending on hardware)

### Using Trained Models

Once training is complete, use your custom model:

```python
from road_damage_detector import RoadDamageDetector

# Use your trained model
detector = RoadDamageDetector("runs/RDD_Training/YOLOv8_RDD_Model/weights/best.pt")
results = detector.analyze_video("your_video.mp4")
```

## Output

The system generates several types of output:

1. **Annotated Videos**: Videos with bounding boxes around detected damage
2. **HTML Reports**: Detailed analysis reports with statistics and visualizations
3. **JSON Results**: Raw detection data for further processing
4. **Training Models**: Custom trained YOLO models

## File Structure

```
road-damage-detection/
├── road_damage_detector.py    # Main detection class
├── train_model.py            # Advanced training with CLI
├── training_notebook.py      # Simple interactive training
├── evaluate_model.py         # Model evaluation script
├── download_dataset.py       # RDD2022 dataset downloader
├── convert_dataset.py        # XML to YOLO converter
├── analyze_youtube.py        # YouTube analysis script
├── analyze_local.py          # Local video analysis script
├── setup_and_demo.py         # Setup and demonstration
├── requirements.txt          # Python dependencies
├── setup_env.sh/.bat         # Environment setup scripts
├── 15-01-2024/              # Sample video data
│   └── C0007.MP4            # Sample video
├── RDD2022/                 # Downloaded dataset (after download_dataset.py)
├── yolo_dataset/            # Converted YOLO dataset (after convert_dataset.py)
└── runs/                    # Training results and model weights
```

## Sample Results

After running analysis, you'll get:

- **Detection Statistics**: Total damages found, damage density per second
- **Damage Classification**: Breakdown by damage type (potholes, cracks, etc.)
- **Temporal Analysis**: When damage occurs in the video timeline
- **Visual Output**: Annotated video showing detected damage with bounding boxes

## Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
2. **Frame Skipping**: Use `--skip-frames` to process every nth frame for speed
3. **Confidence Tuning**: Adjust confidence threshold based on your needs
4. **Custom Models**: Train on your specific road conditions for better accuracy

## Troubleshooting

### Common Issues

1. **"Import could not be resolved"**: Run `pip install -r requirements.txt`
2. **Video download fails**: Check internet connection and YouTube URL
3. **Low detection accuracy**: Try lowering confidence threshold or training custom model
4. **Out of memory**: Reduce batch size or use CPU instead of GPU

### Dependencies Issues
If you encounter issues with dependencies, try:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Uses [OpenCV](https://opencv.org/) for video processing
- YouTube downloading via [pytube](https://github.com/pytube/pytube)

