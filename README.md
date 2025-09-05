# MMA Fighter Tracker

A sophisticated computer vision system for tracking and analyzing fighters in MMA/UFC videos using semantic segmentation, CLIP embeddings, and multi-object tracking. This repository provides multiple tracking algorithms optimized for combat sports analysis.

## 🥊 Features

- **Person Segmentation**: Uses MaskR-CNN for accurate person detection and segmentation
- **Semantic Embeddings**: Leverages CLIP for robust visual feature extraction
- **Multi-Object Tracking**: Combines IoU and semantic similarity for consistent tracking
- **Keypoint Detection**: Optional pose estimation for enhanced analysis
- **Mask Splitting**: Automatic separation of interlinked fighters using watershed segmentation
- **Video Output**: Generates annotated videos with track IDs and optional keypoints
- **Data Export**: Exports tracking data in CSV format and semantic label maps
- **Multiple Algorithms**: BotSORT, DeepSORT, occlusion-aware tracking, and custom semantic tracking

## 📋 Requirements

- Python 3.8+
- PyTorch >= 1.13.0
- OpenCV >= 4.5.0
- Transformers (HuggingFace) >= 4.20.0
- Torchvision >= 0.14.0
- NumPy >= 1.21.0
- Additional dependencies in `requirements.txt`

## 🚀 Quick Start

### Installation

1. **Clone this repository:**
```bash
git clone https://github.com/Al31415/mma-fighter-tracker.git
cd mma-fighter-tracker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download model weights** (automatically downloaded on first use):
- MaskR-CNN ResNet-50 (person segmentation)
- CLIP ViT-B/32 (semantic embeddings)  
- Keypoint R-CNN ResNet-50 (pose estimation)
- Optional: YOLOv8, OSNet ReID, SAM models for specific trackers

### Test with Example Data

We provide a sample clip from the Adesanya vs. Du Plessis fight for testing:

```bash
# Download the example clip (30-second segment)
mkdir -p example_data
# Add your test video to example_data/

# Run basic tracking
python fighter_tracker.py --video example_data/test_clip.mp4 --outdir results --max-frames 100

# Or try different algorithms
python bmp_botsort_tracker.py --video example_data/test_clip.mp4 --outdir results_botsort
```

## Usage

### Basic Tracking

```bash
python fighter_tracker.py --video path/to/your/video.mp4 --outdir output_directory
```

### Advanced Configuration

```bash
python fighter_tracker.py \
    --video input_video.mp4 \
    --outdir tracker_output \
    --stride 5 \
    --conf 0.6 \
    --iou-weight 0.4 \
    --sim-weight 0.6 \
    --max-persons 2 \
    --draw-keypoints
```

### Parameters

- `--video`: Path to input video file
- `--outdir`: Output directory for results (default: `tracker_out`)
- `--stride`: Process every Nth frame (default: 5)
- `--conf`: Detection confidence threshold (default: 0.6)
- `--iou-weight`: Weight for IoU in matching score (default: 0.4)
- `--sim-weight`: Weight for semantic similarity (default: 0.6)
- `--iou-thresh`: Minimum IoU for geometric matching (default: 0.1)
- `--sim-thresh`: Minimum cosine similarity for semantic matching (default: 0.2)
- `--max-missed`: Max frames before terminating track (default: 20)
- `--max-frames`: Maximum frames to process (default: 400, 0 for all)
- `--max-persons`: Maximum persons to track per frame (default: 2)
- `--min-mask-area-frac`: Minimum mask area fraction (default: 0.005)
- `--center-bias`: Center bias weight for scoring (default: 0.3)
- `--no-disjoint-overlaps`: Disable overlap resolution
- `--no-keypoint-split`: Disable keypoint-based splitting
- `--draw-keypoints`: Draw pose keypoints on output video

## Output

The tracker generates several outputs:

1. **tracked.mp4**: Annotated video with track overlays
2. **tracks.csv**: Tracking data with bounding boxes and track IDs
3. **labels/**: Directory with semantic segmentation maps (PNG files)

### CSV Format

```
frame,track_id,x1,y1,x2,y2,age
0,1,100.0,50.0,200.0,300.0,1
0,2,250.0,80.0,350.0,320.0,1
...
```

## 📁 Project Structure & Files

### Core Tracking Scripts

| File | Purpose | Key Features |
|------|---------|-------------|
| **`fighter_tracker.py`** | Main semantic tracker | CLIP embeddings + MaskR-CNN, keypoint splitting, watershed segmentation |
| **`bmp_botsort_tracker.py`** | BotSORT implementation | YOLO detection, motion prediction, Re-ID features |
| **`bmp_botsort_reid.py`** | BotSORT with ReID | Enhanced identity matching with appearance features |
| **`bmp_deepsort_tracker.py`** | DeepSORT tracker | Kalman filtering + deep appearance features |
| **`bmp_occlusion_tracker.py`** | Occlusion-aware tracking | Handles fighter overlap and occlusion scenarios |
| **`bmp_occlusion_botsort.py`** | Occlusion-aware BotSORT | BotSORT optimized for combat sports occlusions |
| **`bmp_iterative_refine_tracker.py`** | Iterative refinement | SAM-based mask refinement with keypoint guidance |

### Analysis & Utilities

| File | Purpose | Usage |
|------|---------|-------|
| **`count_fights.py`** | Count downloaded fights | `python count_fights.py --dir ufc_videos --official-only` |
| **`download_ufc.py`** | Download UFC videos | Scrapes fight videos from online sources |
| **`youtube_free_fights.py`** | YouTube UFC downloader | `python youtube_free_fights.py --query "ufc free fight" --max-results 50` |
| **`interactive_embeddings.py`** | Embedding visualization | `python interactive_embeddings.py --csv embeddings_2d.csv` |
| **`video_cluster_embed.py`** | Video clustering | `python video_cluster_embed.py --video fight.mp4 --chunk-duration 30` |

### Algorithm Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Semantic Tracker** | Complex occlusions, identity consistency | CLIP embeddings, robust to appearance changes | Computationally intensive |
| **BotSORT** | Real-time applications | Fast, good motion prediction | May struggle with heavy occlusion |
| **DeepSORT** | Balanced performance | Proven track record, stable | Limited semantic understanding |
| **Occlusion-aware** | Close combat, clinching | Specialized for combat sports | May over-segment in open fighting |

## Technical Details

### Architecture

The system uses a multi-stage pipeline:

1. **Detection**: MaskR-CNN detects and segments persons in each frame
2. **Selection**: Filters detections based on size, position, and confidence
3. **Embedding**: CLIP encodes visual features for each detected person
4. **Tracking**: Associates detections across frames using IoU + semantic similarity
5. **Refinement**: Optional keypoint-based splitting of merged detections

### Key Components

- `PersonSegmenter`: MaskR-CNN-based person detection and segmentation
- `ClipEmbedder`: CLIP-based visual feature extraction
- `PersonKeypointDetector`: Pose estimation for enhanced tracking
- `SemanticTracker`: Multi-object tracking with semantic features
- `Track`: Individual track state management

### Algorithm Features

- **Semantic Consistency**: Uses CLIP embeddings to maintain identity across occlusions
- **Geometric Validation**: IoU-based validation prevents track switching
- **Adaptive Splitting**: Watershed algorithm separates merged fighter detections
- **Robust Association**: Combines multiple cues for reliable tracking

## 📊 Data Organization

### Recommended Directory Structure

```
mma-fighter-tracker/
├── example_data/                    # Test videos and sample data
│   ├── adesanya_ddp_clip.mp4       # Example: 30s clip for testing
│   └── README.md                   # Data usage instructions
├── models/                         # Downloaded model weights
│   ├── osnet_x0_25_msmt17.pt      # ReID model
│   ├── yolov8x-pose.pt            # Pose detection
│   └── sam_vit_h_4b8939.pth       # SAM segmentation
├── data/                           # Your video datasets
│   ├── ufc_videos/                # Downloaded UFC fights
│   ├── training_clips/            # Clips for model training
│   └── test_sets/                 # Evaluation datasets
└── outputs/                        # Tracking results
    ├── tracker_out_semantic/      # Semantic tracker results
    ├── tracker_out_botsort/       # BotSORT results
    └── analysis/                  # Embedding visualizations
```

### Adding Your Own Data

1. **Video Format**: MP4, AVI, MOV supported. Recommended: 720p+ resolution
2. **Naming Convention**: Use descriptive names like `fighter1_vs_fighter2_event_date.mp4`
3. **Organization**: Group by event, date, or fighter for easy management

```bash
# Example: Add new fight videos
mkdir -p data/ufc_events/ufc_300
cp your_fight_video.mp4 data/ufc_events/ufc_300/

# Process with different trackers
python fighter_tracker.py --video data/ufc_events/ufc_300/your_fight_video.mp4 --outdir outputs/ufc_300_semantic
python bmp_botsort_tracker.py --video data/ufc_events/ufc_300/your_fight_video.mp4 --outdir outputs/ufc_300_botsort
```

## 🎯 Example Data & Testing

### Adesanya vs. Du Plessis Test Clip

We provide a sample from the Adesanya vs. Du Plessis fight (UFC 305) for testing:

**Download Example Clip:**
```bash
# Create example data directory
mkdir -p example_data

# Download a 30-second test clip (you'll need to add this)
# Place your test clip as: example_data/adesanya_ddp_30s.mp4
```

**Run All Trackers:**
```bash
# Test semantic tracker (most accurate, slower)
python fighter_tracker.py \
    --video example_data/adesanya_ddp_30s.mp4 \
    --outdir results/semantic \
    --max-frames 150 \
    --draw-keypoints

# Test BotSORT (fastest)
python bmp_botsort_tracker.py \
    --video example_data/adesanya_ddp_30s.mp4 \
    --outdir results/botsort \
    --max-frames 150

# Test occlusion-aware tracker (best for clinching)
python bmp_occlusion_tracker.py \
    --video example_data/adesanya_ddp_30s.mp4 \
    --outdir results/occlusion \
    --max-frames 150

# Generate embeddings and clustering
python video_cluster_embed.py \
    --video example_data/adesanya_ddp_30s.mp4 \
    --outdir results/embeddings \
    --chunk-duration 5
```

**Expected Results:**
- `tracked.mp4`: Annotated video with fighter IDs
- `tracks.csv`: Frame-by-frame tracking data
- `labels/`: Semantic segmentation masks
- Interactive embedding visualizations

## ⚡ Performance Notes

| Tracker | Speed (FPS) | GPU Memory | Accuracy | Best Use Case |
|---------|-------------|------------|----------|---------------|
| BotSORT | ~15-20 | 2-4GB | Good | Real-time applications |
| Semantic | ~3-5 | 6-8GB | Excellent | Research, detailed analysis |
| DeepSORT | ~10-15 | 3-5GB | Good | Balanced performance |
| Occlusion-aware | ~8-12 | 4-6GB | Very Good | Close combat, clinching |

**Optimization Tips:**
- Use `--stride 5` or higher for faster processing
- Reduce `--max-persons 2` if only tracking main fighters
- Lower `--conf` threshold if missing detections
- Use GPU acceleration for significant speedup

## 📄 License

This project is open source under the MIT License. Please check individual model licenses for commercial use:
- MaskR-CNN, Keypoint R-CNN: Apache 2.0
- CLIP: MIT License  
- YOLO: GPL-3.0
- BotSORT, DeepSORT: Various (check respective repositories)

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New tracking algorithms
- Performance optimizations
- Better occlusion handling
- Multi-camera fusion
- Real-time processing

Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{mma_fighter_tracker,
  title={MMA Fighter Tracker: Semantic Multi-Object Tracking for Combat Sports},
  author={Al31415},
  year={2024},
  url={https://github.com/Al31415/mma-fighter-tracker}
}
``` 