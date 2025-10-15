# Troubleshooting Guide

## Common Issues & Solutions

### Installation Problems

#### 1. CUDA/GPU Issues

**Problem**: `CUDA out of memory` or `No CUDA devices found`

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use CPU if GPU not available
python tracker.py --device cpu --video your_video.mp4
```

**Memory Optimization**:
```bash
# Reduce batch size and max frames
python tracker.py --max-frames 100 --stride 10 --video your_video.mp4

# Use smaller YOLO model
# Edit tracker.py: change "yolov8x-seg.pt" to "yolov8n-seg.pt"
```

#### 2. Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'torchreid'`

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# If torchreid fails, install manually
pip install torchreid>=1.4.0
pip install tensorboard
```

**Problem**: `ModuleNotFoundError: No module named 'boxmot'`

**Solution**:
```bash
# Install boxmot for BotSORT
pip install boxmot>=10.0.0

# Alternative: install from source
git clone https://github.com/mikel-brostrom/yolo_tracking.git
cd yolo_tracking
pip install -e .
```

#### 3. Model Download Issues

**Problem**: Models not downloading automatically

**Solution**:
```bash
# Download models manually
mkdir -p models
cd models

# YOLOv8-seg (will download automatically on first use)
# OSNet ReID weights
wget https://github.com/KaiyangZhou/deep-person-reid/releases/download/v3.0/osnet_x0_25_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth
```

### Detection Issues

#### 1. No Detections

**Problem**: No fighters detected in video

**Solutions**:
```bash
# Lower confidence threshold
python tracker.py --conf 0.3 --video your_video.mp4

# Check if video is loading correctly
python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print('Frames:', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))"

# Try different stride
python tracker.py --stride 1 --max-frames 50 --video your_video.mp4
```

**Debug Steps**:
1. Verify video file exists and is readable
2. Check video format (MP4, AVI supported)
3. Ensure video contains visible persons
4. Try with example video first

#### 2. Too Many False Detections

**Problem**: Detecting referees, cornermen, crowd members

**Solutions**:
```bash
# Increase confidence threshold
python tracker.py --conf 0.6 --video your_video.mp4

# Filter by size (add to tracker.py)
# min_area = 0.01 * (frame_width * frame_height)
```

**Post-processing**:
```bash
# Use ID merging to consolidate tracks
python src/utils/merge_ids.py --tracks-csv results/tracks.csv --output-csv results/tracks_clean.csv
```

#### 3. Missing Detections During Occlusions

**Problem**: Fighters not detected when overlapping

**Solutions**:
```bash
# Lower confidence for partial detections
python tracker.py --conf 0.25 --video your_video.mp4

# Use smaller stride for more frames
python tracker.py --stride 3 --video your_video.mp4

# Enable debug mode (add --debug flag to tracker.py)
```

### Tracking Issues

#### 1. ID Switching

**Problem**: Same fighter gets different IDs across frames

**Solutions**:
```bash
# Use ID merging utility
python src/utils/merge_ids.py \
    --video your_video.mp4 \
    --tracks-csv results/tracks.csv \
    --output-csv results/tracks_merged.csv \
    --similarity-threshold 0.7

# Adjust ReID parameters in tracker.py
# Lower appearance_thresh for stricter matching
```

**Manual ID Assignment**:
```python
# Edit tracks.csv manually for critical frames
# Or use post-processing script to fix specific IDs
```

#### 2. Track Loss During Occlusions

**Problem**: Tracks disappear and don't recover

**Solutions**:
```bash
# Increase track buffer (edit tracker.py)
# tracker.track_buffer = 60  # Default: 30

# Lower min_hits requirement
# tracker.min_hits = 1  # Default: 3

# Increase max_age
# tracker.max_age = 60  # Default: 30
```

#### 3. Inconsistent Track IDs

**Problem**: Multiple IDs for same fighter

**Solutions**:
```bash
# Use smart merging with co-occurrence constraints
python src/utils/merge_ids.py \
    --tracks-csv results/tracks.csv \
    --output-csv results/tracks_smart.csv \
    --similarity-threshold 0.65

# Filter by box size to remove referee
python -c "
import pandas as pd
df = pd.read_csv('results/tracks.csv')
df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])
# Keep only large boxes (main fighters)
df_large = df[df['area'] > df['area'].quantile(0.7)]
df_large.to_csv('results/tracks_fighters_only.csv', index=False)
"
```

### Performance Issues

#### 1. Slow Processing

**Problem**: Processing takes too long

**Solutions**:
```bash
# Increase stride (process fewer frames)
python tracker.py --stride 10 --video your_video.mp4

# Reduce max frames for testing
python tracker.py --max-frames 100 --video your_video.mp4

# Use smaller YOLO model
# Edit tracker.py: "yolov8n-seg.pt" instead of "yolov8x-seg.pt"
```

**GPU Optimization**:
```bash
# Ensure GPU is being used
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Use mixed precision (add to tracker.py)
# model.half()  # Use FP16 for faster inference
```

#### 2. High Memory Usage

**Problem**: Out of memory errors

**Solutions**:
```bash
# Process video in chunks
python tracker.py --max-frames 50 --video your_video.mp4

# Use CPU instead of GPU
python tracker.py --device cpu --video your_video.mp4

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. Poor Quality Results

**Problem**: Low detection accuracy or tracking quality

**Solutions**:
```bash
# Use higher resolution model
# Edit tracker.py: "yolov8x-seg.pt" (largest model)

# Increase processing frequency
python tracker.py --stride 1 --max-frames 200 --video your_video.mp4

# Fine-tune parameters
python tracker.py --conf 0.4 --iou 0.6 --video your_video.mp4
```

### Output Issues

#### 1. No Output Files

**Problem**: No results generated

**Debug Steps**:
```bash
# Check if output directory exists
ls -la results/

# Verify video processing
python -c "
import cv2
cap = cv2.VideoCapture('your_video.mp4')
ret, frame = cap.read()
print('Frame read:', ret)
print('Frame shape:', frame.shape if ret else 'No frame')
"

# Check for errors in console output
python tracker.py --video your_video.mp4 --outdir results 2>&1 | tee log.txt
```

#### 2. Corrupted Video Output

**Problem**: Output video is corrupted or unplayable

**Solutions**:
```bash
# Try different codec
# Edit tracker.py: change fourcc to 'XVID' or 'MJPG'

# Check input video format
ffmpeg -i your_video.mp4 -f null -

# Re-encode input video
ffmpeg -i your_video.mp4 -c:v libx264 -crf 23 input_clean.mp4
```

#### 3. CSV Format Issues

**Problem**: CSV file is empty or malformed

**Solutions**:
```bash
# Check CSV content
head -10 results/tracks.csv

# Verify CSV format
python -c "
import pandas as pd
df = pd.read_csv('results/tracks.csv')
print('Columns:', df.columns.tolist())
print('Shape:', df.shape)
print('Sample:', df.head())
"
```

### Video-Specific Issues

#### 1. Low Resolution Videos

**Problem**: Poor detection on low-res videos

**Solutions**:
```bash
# Upscale video first
ffmpeg -i input.mp4 -vf scale=1920:1080 -c:v libx264 output.mp4

# Use lower confidence threshold
python tracker.py --conf 0.3 --video your_video.mp4
```

#### 2. High Motion Videos

**Problem**: Blurry frames affect detection

**Solutions**:
```bash
# Use motion deblurring (preprocessing)
# Or increase stride to skip blurry frames
python tracker.py --stride 5 --video your_video.mp4

# Use temporal smoothing (post-processing)
```

#### 3. Multiple People in Frame

**Problem**: Tracking non-fighters (referee, cornermen)

**Solutions**:
```bash
# Filter by position (center of frame)
# Filter by size (larger bounding boxes)
# Use post-processing to remove small tracks

python -c "
import pandas as pd
df = pd.read_csv('results/tracks.csv')
# Keep only tracks with >100 detections (main fighters)
track_counts = df['track_id'].value_counts()
main_tracks = track_counts[track_counts > 100].index
df_filtered = df[df['track_id'].isin(main_tracks)]
df_filtered.to_csv('results/tracks_main_fighters.csv', index=False)
"
```

## Getting Help

### Debug Mode

Enable detailed logging:
```bash
# Add debug prints to tracker.py
python tracker.py --video your_video.mp4 --outdir results --debug
```

### Performance Profiling

```bash
# Profile memory usage
python -m memory_profiler tracker.py --video your_video.mp4

# Profile execution time
python -m cProfile -s cumulative tracker.py --video your_video.mp4
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | GPU memory full | Use `--device cpu` or reduce `--max-frames` |
| `No module named 'torchreid'` | Missing dependency | `pip install torchreid` |
| `Failed to open video` | Invalid video file | Check file path and format |
| `No detections found` | Low confidence | Lower `--conf` threshold |
| `Track ID switching` | ReID threshold too low | Use ID merging utility |

### Contact & Support

- **Issues**: Open a GitHub issue with video details and error logs
- **Documentation**: Check [Technical Details](TECHNICAL_DETAILS.md)
- **Examples**: Use provided example video for testing

### Tips for Best Results

1. **Video Quality**: Use 720p+ resolution videos
2. **Lighting**: Ensure good lighting and contrast
3. **Camera Stability**: Minimize camera shake
4. **Fighter Visibility**: Avoid extreme close-ups
5. **Parameter Tuning**: Adjust confidence based on video content
6. **Post-processing**: Use ID merging for cleaner results
