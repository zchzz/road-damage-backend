# Road Damage Detection Configuration

# Detection Settings
DEFAULT_CONFIDENCE = 0.3
DEFAULT_SKIP_FRAMES = 5

# Model Settings
DEFAULT_MODEL = (
    "runs/RDD_Training/test_training/weights/best.pt"  # Use trained RDD model
)

# Damage Classes (RDD2022 dataset classes)
DAMAGE_CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes",
]

# Training Settings
TRAINING_EPOCHS = 100
TRAINING_BATCH_SIZE = 16
TRAINING_IMAGE_SIZE = 640
FRAME_EXTRACT_INTERVAL = 30  # Extract every 30th frame for training

# Output Settings
OUTPUT_VIDEO_CODEC = "mp4v"
GENERATE_HTML_REPORTS = True
SAVE_JSON_RESULTS = True

# YouTube Download Settings
YOUTUBE_QUALITY_PREFERENCE = "highest"  # or "720p", "480p", etc.

# Performance Settings
USE_GPU = True  # Set to False to force CPU usage
MAX_WORKERS = 4
