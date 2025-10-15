"""
Analyze tracking results and merge fragmented IDs using ReID similarity
"""

import argparse
import os
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path


def load_reid_model(device, model_name="osnet_x1_0"):
	"""Load OSNet ReID model"""
	try:
		import torchreid
		from torchvision import transforms
		
		model = torchreid.models.build_model(
			name=model_name,
			num_classes=1000,
			loss='softmax',
			pretrained=True
		)
		model = model.to(device).eval()
		
		transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((256, 128)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
		print(f"Loaded ReID model: {model_name}")
		return model, transform
	except Exception as e:
		print(f"Failed to load ReID model: {e}")
		return None, None


def extract_reid_features(video_path, tracks_df, model, transform, device, max_samples=5):
	"""Extract ReID features for each track ID"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	
	# Group tracks by ID
	track_groups = tracks_df.groupby('track_id')
	
	# Sample frames for each track
	track_features = {}
	
	for track_id, group in track_groups:
		# Sample frames evenly
		frames = sorted(group['frame'].unique())
		step = max(1, len(frames) // max_samples)
		sampled_frames = frames[::step][:max_samples]
		
		features = []
		for frame_idx in sampled_frames:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
			ret, frame = cap.read()
			if not ret:
				continue
			
			# Get box for this track in this frame
			box_row = group[group['frame'] == frame_idx].iloc[0]
			x1, y1, x2, y2 = int(box_row['x1']), int(box_row['y1']), int(box_row['x2']), int(box_row['y2'])
			
			# Crop person
			crop = frame[y1:y2, x1:x2]
			if crop.size == 0:
				continue
			
			# Extract ReID features
			crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
			tensor = transform(crop_rgb).unsqueeze(0).to(device)
			with torch.no_grad():
				feat = model(tensor)
				feat = F.normalize(feat, dim=1)
			features.append(feat.cpu())
		
		if features:
			# Average features for this track
			track_features[track_id] = torch.cat(features).mean(dim=0)
			print(f"  Track {track_id}: {len(features)} samples")
	
	cap.release()
	return track_features


def merge_similar_ids(track_features, similarity_threshold=0.5):
	"""Merge track IDs with high ReID similarity"""
	track_ids = list(track_features.keys())
	
	# Compute pairwise similarities
	similarities = {}
	for i, id1 in enumerate(track_ids):
		for id2 in track_ids[i+1:]:
			sim = F.cosine_similarity(
				track_features[id1].unsqueeze(0),
				track_features[id2].unsqueeze(0)
			).item()
			similarities[(id1, id2)] = sim
	
	# Find merge candidates (high similarity pairs)
	print(f"\nPairwise similarities:")
	for (id1, id2), sim in sorted(similarities.items(), key=lambda x: -x[1]):
		print(f"  Track {id1} <-> Track {id2}: {sim:.3f}")
	
	# Build merge mapping using greedy clustering
	id_map = {tid: tid for tid in track_ids}  # Initially, each ID maps to itself
	
	# Sort by similarity (highest first)
	sorted_pairs = sorted(similarities.items(), key=lambda x: -x[1])
	
	for (id1, id2), sim in sorted_pairs:
		if sim >= similarity_threshold:
			# Get root IDs
			root1 = id_map[id1]
			root2 = id_map[id2]
			
			if root1 != root2:
				# Merge into smaller ID
				new_root = min(root1, root2)
				old_root = max(root1, root2)
				
				# Update all IDs pointing to old_root
				for tid in id_map:
					if id_map[tid] == old_root:
						id_map[tid] = new_root
				
				print(f"  Merging: {id1} and {id2} (sim={sim:.3f}) -> {new_root}")
	
	return id_map


def apply_id_mapping(tracks_df, id_map):
	"""Apply ID mapping to tracks CSV"""
	df = tracks_df.copy()
	df['original_track_id'] = df['track_id']
	df['track_id'] = df['track_id'].map(id_map)
	return df


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--video", type=str, required=True)
	parser.add_argument("--tracks-csv", type=str, required=True)
	parser.add_argument("--output-csv", type=str, required=True)
	parser.add_argument("--similarity-threshold", type=float, default=0.5)
	parser.add_argument("--reid-model", type=str, default="osnet_x1_0")
	args = parser.parse_args()
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	
	# Load tracks
	print(f"Loading tracks from {args.tracks_csv}...")
	tracks_df = pd.read_csv(args.tracks_csv)
	print(f"  {len(tracks_df)} detections, {tracks_df['track_id'].nunique()} unique IDs")
	
	# Load ReID model
	model, transform = load_reid_model(device, args.reid_model)
	if model is None:
		print("Failed to load ReID model, exiting.")
		return
	
	# Extract ReID features
	print("\nExtracting ReID features...")
	track_features = extract_reid_features(args.video, tracks_df, model, transform, device)
	
	# Merge similar IDs
	print(f"\nMerging IDs with similarity >= {args.similarity_threshold}...")
	id_map = merge_similar_ids(track_features, args.similarity_threshold)
	
	# Apply mapping
	print("\nApplying ID mapping...")
	merged_df = apply_id_mapping(tracks_df, id_map)
	
	# Save
	merged_df.to_csv(args.output_csv, index=False)
	print(f"\nSaved merged tracks to {args.output_csv}")
	
	# Report
	print(f"\nMerge Summary:")
	print(f"  Original IDs: {tracks_df['track_id'].nunique()}")
	print(f"  Merged IDs: {merged_df['track_id'].nunique()}")
	print(f"\nID mapping:")
	for old_id, new_id in sorted(id_map.items()):
		if old_id != new_id:
			print(f"  {old_id} -> {new_id}")
	
	print(f"\nMerged track distribution:")
	print(merged_df['track_id'].value_counts())


if __name__ == "__main__":
	main()

