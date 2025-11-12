# MMA Fighter Tracker

**Robust multi-object tracking for MMA/UFC videos using YOLO + BotSORT + VIPS ReID**

Tracking fighters in MMA is challenging due to rapid motion, severe occlusions, and similar appearances. This system combines state-of-the-art detection, tracking, and re-identification to maintain consistent fighter identities throughout combat sequences.

## Demo Video

See the tracker in action on a real UFC fight:

[![Demo Video](https://img.youtube.com/vi/TuAXbriYNxg/0.jpg)](https://youtu.be/TuAXbriYNxg)

*Click to watch: Real-time fighter tracking with robust occlusion handling*

## Key Features

- **Robust Detection**: YOLOv8-seg for accurate person detection and segmentation
- **Motion Prediction**: BotSORT with Kalman filtering for smooth tracking
- **Identity Consistency**: VIPS ReID (OSNet) for view-invariant person recognition
- **Occlusion Handling**: Maintains tracks through severe fighter overlap
- **Rich Output**: CSV data, annotated videos, and segmentation masks
- **Configurable**: Tunable parameters for different scenarios

## Quick Start

Get up and running in 5 minutes:

```bash
# Clone and setup
git clone https://github.com/Al31415/mma-fighter-tracker.git
cd mma-fighter-tracker
pip install -r requirements.txt

# Run on example video
python src/tracker.py --video example_data/UFC_20250803_Dricus_Du_Plessis_vs_Israel_Adesanya_FULL_FIGHT_UFC_319.f616\ -\ chunk_17\ \[510-540\]s.mp4 --outdir results

# View results
ls results/
# tracked.mp4  tracks.csv  labels/
```

## How It Works

### 3-Stage Pipeline

```
Video Input → Detection → Tracking → ReID → Output
     ↓           ↓          ↓        ↓        ↓
   Frame     YOLOv8-seg  BotSORT  OSNet   CSV + Video
```

1. **Detection**: YOLOv8-seg identifies and segments persons in each frame
2. **Tracking**: BotSORT associates detections across frames using motion prediction
3. **ReID**: OSNet ReID maintains identity consistency during occlusions

### Why This Approach?

- **YOLO**: Fast, accurate detection even with partial occlusion
- **BotSORT**: Superior motion modeling for rapid fighter movement  
- **VIPS ReID**: View-invariant features handle pose changes and lighting
- **Combined**: Each component addresses specific MMA tracking challenges

## Usage & Configuration

### Basic Usage

```bash
python src/tracker.py --video your_fight.mp4 --outdir results
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--conf` | 0.5 | Detection confidence threshold |
| `--max-frames` | 400 | Maximum frames to process (0 = all) |
| `--stride` | 5 | Process every Nth frame for speed |
| `--max-fighters` | 2 | Maximum fighters to track |
| `--device` | auto | Device for inference (auto/cpu/cuda) |

### Advanced Configuration

```bash
python src/tracker.py \
    --video fight.mp4 \
    --outdir results \
    --conf 0.4 \
    --max-frames 1000 \
    --stride 3 \
    --max-fighters 2 \
    --device cuda
```

### Output Format

**CSV Data** (`tracks.csv`):
```csv
frame,track_id,x1,y1,x2,y2
0,1,559.0,256.0,1420.0,1015.0
5,1,823.0,305.0,1404.0,1034.0
```

**Files Generated**:
- `tracked.mp4`: Annotated video with bounding boxes and IDs
- `tracks.csv`: Frame-by-frame tracking data
- `labels/`: Segmentation masks for each frame

## Results & Performance

### Example Results (Adesanya vs. Du Plessis)

| Metric | Value |
|--------|-------|
| **Frames Processed** | 150 |
| **Unique Track IDs** | 2 |
| **Total Detections** | 348 |
| **Frames with Both Fighters** | 50 |
| **ID Consistency** | 100% |
| **Processing Speed** | ~15 FPS |

### Comparison to Alternatives

| Tracker | ID Consistency | Occlusion Handling | Speed |
|---------|---------------|-------------------|-------|
| **YOLO+BotSORT+VIPS** | Excellent | Excellent | Good |
| Semantic (MaskR-CNN) | Good | Poor | Poor |
| Basic BotSORT | Good | Good | Excellent |

*See [docs/RESULTS.md](docs/RESULTS.md) for detailed performance analysis*

## Project Structure

```
mma-fighter-tracker/
├── src/
│   ├── tracker.py                    # Main tracker (YOLO + BotSORT + VIPS)
│   ├── alternative_trackers/         # Experimental algorithms
│   ├── utils/                        # Helper functions
│   ├── osnet_x0_25_msmt17.pt        # ReID model weights
│   └── yolov8x-seg.pt               # YOLO model weights
├── requirements.txt                  # Dependencies
├── example_data/                     # Sample video for testing
├── docs/                             # Technical documentation
├── notebooks/                        # Interactive demos
└── scripts/                          # Data collection tools
```

## Tuning Options

### Extending the System

1. **New Detection Models**: Replace YOLOv8-seg in `src/tracker.py`
2. **Alternative Trackers**: Add to `src/alternative_trackers/`
3. **Custom ReID**: Modify the OSNet implementation
4. **Post-processing**: Use utilities in `src/utils/`

### Alternative Trackers

- `src/alternative_trackers/semantic_tracker.py`: CLIP-based semantic tracking
- `src/alternative_trackers/botsort_basic.py`: Basic BotSORT implementation

*See [docs/TECHNICAL_DETAILS.md](docs/TECHNICAL_DETAILS.md) for implementation details*


## Documentation

- **[Technical Details](docs/TECHNICAL_DETAILS.md)**: Algorithm deep-dive and implementation
- **[Results Analysis](docs/RESULTS.md)**: Performance metrics and comparisons  
- **[Interactive Demo](notebooks/tracker_demo.ipynb)**: Jupyter notebook walkthrough
