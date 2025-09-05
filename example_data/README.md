# Example Data for MMA Fighter Tracker

This directory contains sample video clips for testing the various tracking algorithms.

## Test Clip: Adesanya vs. Du Plessis

**File:** `adesanya_ddp_30s.mp4`
**Source:** UFC 305 - Israel Adesanya vs. Dricus Du Plessis
**Duration:** 30 seconds (0:00-0:30 of the fight)
**Resolution:** 720p
**Content:** Opening moments of the fight, good for testing basic tracking

### Quick Test Commands

```bash
# Test the main semantic tracker
python fighter_tracker.py --video example_data/adesanya_ddp_30s.mp4 --outdir test_results/semantic --max-frames 150

# Test BotSORT (fastest)
python bmp_botsort_tracker.py --video example_data/adesanya_ddp_30s.mp4 --outdir test_results/botsort --max-frames 150

# Test with keypoints visualization
python fighter_tracker.py --video example_data/adesanya_ddp_30s.mp4 --outdir test_results/keypoints --max-frames 150 --draw-keypoints
```

### Expected Results

After running the trackers, you should see:
- **tracked.mp4**: Video with colored overlays showing fighter IDs
- **tracks.csv**: CSV file with bounding box coordinates and track IDs
- **labels/**: PNG files with semantic segmentation masks

### Performance Benchmarks

On a typical GPU setup (RTX 3080), expect:
- Semantic tracker: ~3-5 FPS, very accurate
- BotSORT: ~15-20 FPS, good accuracy
- Occlusion tracker: ~8-12 FPS, excellent for close combat

## Adding Your Own Test Data

1. Place MP4 files in this directory
2. Use descriptive names: `fighter1_vs_fighter2_duration.mp4`
3. Recommended: 30-60 second clips for quick testing
4. Resolution: 720p or higher for best results

## Troubleshooting

**Common Issues:**
- "CUDA out of memory": Reduce `--stride` or use CPU with smaller clips
- "No detections found": Lower `--conf` threshold (try 0.4-0.5)
- "Tracks switching": Increase `--sim-weight` for semantic tracker

**Performance Tips:**
- Use `--stride 5` for 5x faster processing
- Set `--max-frames 100` for quick tests
- Enable `--draw-keypoints` only for visualization, not analysis
