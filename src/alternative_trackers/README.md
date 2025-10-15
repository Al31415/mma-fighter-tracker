# Alternative Tracking Algorithms

This directory contains experimental and alternative implementations of tracking algorithms for comparison and research purposes.

## Available Trackers

### 1. Semantic Tracker (`semantic_tracker.py`)
**Original**: `fighter_tracker.py`

**Approach**: MaskR-CNN + CLIP embeddings + semantic similarity
- **Detection**: MaskR-CNN for person segmentation
- **Features**: CLIP embeddings for semantic understanding
- **Tracking**: IoU + semantic similarity matching
- **Splitting**: Watershed algorithm with keypoint seeding

**Best For**: Research, detailed analysis, when semantic understanding is crucial
**Performance**: 3-5 FPS, 8-12GB GPU memory

### 2. Basic BotSORT (`botsort_basic.py`)
**Original**: `bmp_botsort_tracker.py`

**Approach**: YOLO detection + BotSORT tracking
- **Detection**: YOLOv8 for person detection
- **Tracking**: BotSORT with Kalman filtering
- **Features**: Basic appearance features

**Best For**: Real-time applications, when speed is critical
**Performance**: 20 FPS, 2-4GB GPU memory

## Comparison

| Tracker | Speed | Memory | ID Consistency | Occlusion Handling | Use Case |
|---------|-------|--------|----------------|-------------------|----------|
| **YOLO+BotSORT+VIPS** | Good | Good | Excellent | Excellent | **Production** |
| Semantic Tracker | Poor | Poor | Good | Poor | Research |
| Basic BotSORT | Excellent | Excellent | Good | Good | Real-time |

## Usage

```bash
# Run semantic tracker
python src/alternative_trackers/semantic_tracker.py --video your_video.mp4 --outdir results

# Run basic BotSORT
python src/alternative_trackers/botsort_basic.py --video your_video.mp4 --outdir results
```

## When to Use Each

- **YOLO+BotSORT+VIPS** (main tracker): Best overall performance, production use
- **Semantic Tracker**: When you need semantic understanding of fighter actions
- **Basic BotSORT**: When you need maximum speed and minimal resource usage

## Development

These trackers serve as:
1. **Baselines** for comparison
2. **Research platforms** for new algorithms
3. **Fallbacks** when main tracker fails
4. **Educational examples** of different approaches