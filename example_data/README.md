# Example Data for MMA Fighter Tracker

This directory provides instructions and examples for testing the various tracking algorithms.

## Getting Test Data

Due to GitHub's file size limits, video files are not included in the repository. Here are several ways to get test data:

### Option 1: Use Your Own Videos
Place any MP4 video file in this directory and run the trackers:
```bash
# Add your video file
cp your_fight_video.mp4 example_data/

# Test with any tracker
python fighter_tracker.py --video example_data/your_fight_video.mp4 --outdir test_results
```

### Option 2: Download Sample Clips
Create small test clips from any MMA video source:
- **Recommended size**: 10-60 seconds for quick testing
- **Resolution**: 720p or higher for best results
- **Format**: MP4, AVI, MOV supported

### Option 3: Use YouTube Downloader
Use the included YouTube downloader to get UFC free fights:
```bash
python youtube_free_fights.py --query "ufc free fight" --max-results 5 --outdir example_data
```

## Quick Test Commands

Once you have a video file in this directory, use these commands to test different algorithms:

### For Standing Fights (open fighting)
```bash
# Test the main semantic tracker (best accuracy)
python fighter_tracker.py --video example_data/your_video.mp4 --outdir test_results/semantic --max-frames 150

# Test BotSORT (fastest)
python bmp_botsort_tracker.py --video example_data/your_video.mp4 --outdir test_results/botsort --max-frames 150

# Test with keypoints visualization
python fighter_tracker.py --video example_data/your_video.mp4 --outdir test_results/keypoints --max-frames 150 --draw-keypoints
```

### For Close Combat (clinch work, grappling)
```bash
# Test occlusion-aware tracker (best for close combat)
python bmp_occlusion_tracker.py --video example_data/your_video.mp4 --outdir test_results/occlusion --max-frames 200

# Test occlusion-aware BotSORT
python bmp_occlusion_botsort.py --video example_data/your_video.mp4 --outdir test_results/occlusion_botsort --max-frames 200

# Test iterative refinement (SAM-based)
python bmp_iterative_refine_tracker.py --video example_data/your_video.mp4 --outdir test_results/iterative --max-frames 200
```

### Quick Download and Test
```bash
# Download a UFC free fight clip
python youtube_free_fights.py --query "ufc free fight adesanya" --max-results 1 --outdir example_data

# Find the downloaded file and test it
ls example_data/*.mp4
python fighter_tracker.py --video example_data/UFC_*.mp4 --outdir test_results --max-frames 100
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
