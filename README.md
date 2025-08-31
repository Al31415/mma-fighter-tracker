# MMA Fighter Tracker

A sophisticated computer vision system for tracking and analyzing fighters in MMA/UFC videos using semantic segmentation, CLIP embeddings, and multi-object tracking.

## Features

- **Person Segmentation**: Uses MaskR-CNN for accurate person detection and segmentation
- **Semantic Embeddings**: Leverages CLIP for robust visual feature extraction
- **Multi-Object Tracking**: Combines IoU and semantic similarity for consistent tracking
- **Keypoint Detection**: Optional pose estimation for enhanced analysis
- **Mask Splitting**: Automatic separation of interlinked fighters using watershed segmentation
- **Video Output**: Generates annotated videos with track IDs and optional keypoints
- **Data Export**: Exports tracking data in CSV format and semantic label maps

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Transformers (HuggingFace)
- Torchvision
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mma-fighter-tracker.git
cd mma-fighter-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights (if not automatically downloaded):
- YOLOv8 pose model: `yolov8x-pose.pt`
- YOLOv8 segmentation model: `yolov8x-seg.pt`
- OSNet ReID model: `osnet_x0_25_msmt17.pt`
- SAM model: `sam_vit_h_4b8939.pth`

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

## Additional Scripts

- `count_fights.py`: Analyze fight statistics from tracking data
- `download_ufc.py`: Download UFC videos from online sources
- `interactive_embeddings.py`: Interactive visualization of video embeddings
- `video_cluster_embed.py`: Cluster video segments based on visual features
- Various tracker implementations with different algorithms

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

## Performance Notes

- Processing speed depends on video resolution and stride setting
- GPU acceleration strongly recommended for real-time performance
- Memory usage scales with number of simultaneous tracks

## License

This project is open source. Please check individual model licenses for commercial use.

## Contributing

Contributions welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mma_fighter_tracker,
  title={MMA Fighter Tracker: Semantic Multi-Object Tracking for Combat Sports},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mma-fighter-tracker}
}
``` 