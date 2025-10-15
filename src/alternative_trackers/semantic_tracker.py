import argparse
import os
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import cv2


# ------------------------------
# Utilities
# ------------------------------

def _read_video_meta(video_path: str) -> Tuple[float, int, int, int]:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	cap.release()
	return fps, n_frames, w, h


def _xyxy_to_xywh(box: np.ndarray) -> Tuple[float, float, float, float]:
	x1, y1, x2, y2 = box.tolist()
	return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
	x1 = max(box_a[0], box_b[0])
	y1 = max(box_a[1], box_b[1])
	x2 = min(box_a[2], box_b[2])
	y2 = min(box_a[3], box_b[3])
	inter_w = max(0.0, x2 - x1)
	inter_h = max(0.0, y2 - y1)
	inter = inter_w * inter_h
	area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
	area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
	union = area_a + area_b - inter + 1e-6
	return float(inter / union)


def _crop_with_padding(frame: np.ndarray, box: np.ndarray, pad: float = 0.15) -> np.ndarray:
	h, w = frame.shape[:2]
	x1, y1, x2, y2 = box
	bw = x2 - x1
	bh = y2 - y1
	px = bw * pad
	py = bh * pad
	x1p = int(max(0, math.floor(x1 - px)))
	y1p = int(max(0, math.floor(y1 - py)))
	x2p = int(min(w, math.ceil(x2 + px)))
	y2p = int(min(h, math.ceil(y2 + py)))
	crop = frame[y1p:y2p, x1p:x2p]
	if crop.size == 0:
		return frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
	return crop


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
	if mask_a.shape != mask_b.shape:
		return 0.0
	inter = np.logical_and(mask_a, mask_b).sum(dtype=np.float32)
	union = np.logical_or(mask_a, mask_b).sum(dtype=np.float32) + 1e-6
	return float(inter / union)


def _mask_to_box(mask: np.ndarray) -> np.ndarray:
	ys, xs = np.where(mask)
	if ys.size == 0 or xs.size == 0:
		return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	x1 = float(xs.min())
	y1 = float(ys.min())
	x2 = float(xs.max() + 1)
	y2 = float(ys.max() + 1)
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _mask_area(mask: np.ndarray) -> int:
	return int(mask.sum())


def _center_bias_score(mask: np.ndarray, h: int, w: int) -> float:
	# Score in [0,1], 1 when centered, 0 when far from center
	ys, xs = np.where(mask)
	if ys.size == 0:
		return 0.0
	cx = float(xs.mean())
	cy = float(ys.mean())
	cx0 = w * 0.5
	cy0 = h * 0.5
	d = math.hypot(cx - cx0, cy - cy0)
	d_max = math.hypot(cx0, cy0)
	if d_max <= 1e-6:
		return 1.0
	return float(max(0.0, 1.0 - d / d_max))


def _select_foreground_indices(masks: List[np.ndarray], scores: np.ndarray, h: int, w: int, max_persons: int, min_area_frac: float, center_bias: float) -> List[int]:
	areas = [float(_mask_area(m)) for m in masks]
	frame_area = float(h * w)
	cands = []
	for i, m in enumerate(masks):
		area_frac = areas[i] / max(1.0, frame_area)
		if area_frac < min_area_frac:
			continue
		center_score = _center_bias_score(m, h, w)
		# Combined score favors larger and more centered masks, weighted by detection score
		combined = ((1.0 - center_bias) * area_frac + center_bias * center_score) * float(scores[i])
		cands.append((combined, i))
	cands.sort(key=lambda x: x[0], reverse=True)
	return [i for _, i in cands[:max(0, max_persons)]]


def _disjointify_masks(masks: List[np.ndarray], probs: List[np.ndarray], scores: np.ndarray) -> List[np.ndarray]:
	# Assign overlapping pixels to the instance with highest probability*score
	if not masks:
		return []
	h, w = masks[0].shape
	n = len(masks)
	if n == 1:
		return [masks[0].copy()]
	stack = np.zeros((n, h, w), dtype=np.float32)
	for i in range(n):
		p = probs[i].astype(np.float32)
		stack[i] = p * float(scores[i]) * masks[i].astype(np.float32)
	argmax_idx = stack.argmax(axis=0)
	max_vals = stack.max(axis=0)
	assigned = max_vals > 0.0
	new_masks: List[np.ndarray] = []
	for i in range(n):
		new_mask = (argmax_idx == i) & assigned
		new_masks.append(new_mask)
	return new_masks


def _keypoints_centroid(kps: np.ndarray) -> Optional[Tuple[int, int]]:
	# kps: [17,3] -> (x,y,vis)
	if kps.ndim != 2 or kps.shape[1] < 2:
		return None
	vis = None
	if kps.shape[1] >= 3:
		vis = kps[:, 2] > 0.0
	pts = kps[:, :2]
	if vis is not None and vis.any():
		pts = pts[vis]
	if pts.size == 0:
		return None
	cx = int(np.clip(float(pts[:, 0].mean()), 0, 10**9))
	cy = int(np.clip(float(pts[:, 1].mean()), 0, 10**9))
	return (cx, cy)


def _watershed_split(mask: np.ndarray, seed_points: List[Tuple[int, int]], max_segments: int = 2) -> List[np.ndarray]:
	# Split a single binary mask into multiple regions using watershed seeded by seed_points
	if mask.sum() == 0 or len(seed_points) < 2:
		return [mask.copy()]
	h, w = mask.shape
	markers = np.zeros((h, w), dtype=np.int32)
	label = 1
	for (cx, cy) in seed_points[:max_segments]:
		if 0 <= cx < w and 0 <= cy < h and mask[cy, cx]:
			cv2.circle(markers, (cx, cy), 7, label, -1)
			label += 1
	if label <= 2:
		return [mask.copy()]
	# Use inverted distance transform as topography
	dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
	dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	inv = 255 - dist_norm
	img3 = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
	markers_ws = markers.copy()
	try:
		cv2.watershed(img3, markers_ws)
	except cv2.error:
		return [mask.copy()]
	parts: List[np.ndarray] = []
	for i in range(1, label):
		part = (markers_ws == i) & mask
		if part.sum() > 0:
			parts.append(part)
	if not parts:
		return [mask.copy()]
	return parts


# ------------------------------
# Models
# ------------------------------
class PersonSegmenter:
	def __init__(self, device: torch.device):
		from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
		weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
		self.model = maskrcnn_resnet50_fpn(weights=weights).to(device).eval()
		self.device = device

	@torch.no_grad()
	def segment(self, frame_bgr: np.ndarray, conf_thresh: float = 0.6) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]:
		# Convert to tensor
		img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		t = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
		outputs = self.model([t])[0]
		scores = outputs["scores"].detach().cpu().numpy()  # [N]
		labels = outputs["labels"].detach().cpu().numpy()  # [N]
		masks_t = outputs["masks"].detach().cpu().squeeze(1)  # [N,H,W]
		# Keep person class == 1
		keep = (labels == 1) & (scores >= conf_thresh)
		masks: List[np.ndarray] = []
		probs: List[np.ndarray] = []
		for m in masks_t[keep]:
			m_np = m.numpy()
			m_bin = (m_np >= 0.5)
			masks.append(m_bin)
			probs.append(m_np)
		return masks, scores[keep], probs


class ClipEmbedder:
	def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-base-patch32"):
		from transformers import CLIPProcessor, CLIPModel
		self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
		self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
		self.device = device

	@torch.no_grad()
	def embed_batch(self, crops_rgb: List[np.ndarray]) -> torch.Tensor:
		if not crops_rgb:
			return torch.empty(0, 512)
		# Ensure contiguous arrays to avoid negative stride issues with fast processor
		crops_rgb = [crop.copy() for crop in crops_rgb]
		inputs = self.processor(images=crops_rgb, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		feat = self.model.get_image_features(**inputs)
		feat = F.normalize(feat, dim=1)
		return feat.detach().cpu()


class PersonKeypointDetector:
	def __init__(self, device: torch.device):
		from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
		weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
		self.model = keypointrcnn_resnet50_fpn(weights=weights).to(device).eval()
		self.device = device

	@torch.no_grad()
	def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		t = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
		outputs = self.model([t])[0]
		scores = outputs["scores"].detach().cpu().numpy()
		boxes = outputs["boxes"].detach().cpu().numpy()
		keypoints = outputs["keypoints"].detach().cpu().numpy()  # [N,17,3]
		keep = scores >= conf_thresh
		return boxes[keep], scores[keep], keypoints[keep]


# ------------------------------
# Tracking
# ------------------------------
class Track:
	def __init__(self, track_id: int, mask: np.ndarray, emb: torch.Tensor, frame_idx: int):
		self.track_id = track_id
		self.last_mask = mask.astype(bool)
		self.last_box = _mask_to_box(self.last_mask)
		self.last_emb = emb.clone()  # [D]
		self.last_frame_idx = frame_idx
		self.missed = 0
		self.age = 1

	def update(self, mask: np.ndarray, emb: torch.Tensor, frame_idx: int, emb_momentum: float = 0.7):
		self.last_mask = mask.astype(bool)
		self.last_box = _mask_to_box(self.last_mask)
		# EMA update on embedding
		self.last_emb = F.normalize(emb_momentum * self.last_emb + (1.0 - emb_momentum) * emb, dim=0)
		self.last_frame_idx = frame_idx
		self.missed = 0
		self.age += 1


class SemanticTracker:
	def __init__(self, iou_weight: float = 0.4, sim_weight: float = 0.6, iou_thresh: float = 0.1, sim_thresh: float = 0.2, max_missed: int = 20):
		self.iou_weight = iou_weight
		self.sim_weight = sim_weight
		self.iou_thresh = iou_thresh
		self.sim_thresh = sim_thresh
		self.max_missed = max_missed
		self.next_id = 1
		self.tracks: List[Track] = []

	def _score(self, mask: np.ndarray, emb: torch.Tensor, track: Track) -> float:
		iou = _mask_iou(mask, track.last_mask)
		sim = float(F.cosine_similarity(emb.unsqueeze(0), track.last_emb.unsqueeze(0), dim=1).item())
		return self.iou_weight * iou + self.sim_weight * max(0.0, sim)

	def step(self, masks: List[np.ndarray], embs: torch.Tensor, frame_idx: int) -> List[Tuple[int, np.ndarray, float]]:
		# masks: List[H,W] bool, embs: [M,D]
		assignments: Dict[int, List[int]] = {}  # track_idx -> list of det_idx
		det_assigned: Dict[int, int] = {}  # det_idx -> track_idx
		results: List[Tuple[int, np.ndarray, float]] = []

		# Greedy matching by best score
		for det_idx in range(len(masks)):
			best_score = -1.0
			best_track = -1
			for t_idx, track in enumerate(self.tracks):
				score = self._score(masks[det_idx], embs[det_idx], track)
				if score > best_score:
					best_score = score
					best_track = t_idx
			# Thresholds: require either reasonable IoU or high semantic sim
			if best_track >= 0:
				track = self.tracks[best_track]
				iou_val = _mask_iou(masks[det_idx], track.last_mask)
				sim_val = float(F.cosine_similarity(embs[det_idx].unsqueeze(0), track.last_emb.unsqueeze(0), dim=1).item())
				if (iou_val >= self.iou_thresh) or (sim_val >= self.sim_thresh):
					assignments.setdefault(best_track, []).append(det_idx)
					det_assigned[det_idx] = best_track

		# Update matched tracks (union masks, average embeddings)
		for t_idx, det_indices in assignments.items():
			if not det_indices:
				continue
			union_mask = np.zeros_like(self.tracks[t_idx].last_mask, dtype=bool)
			emb_accum = []
			for di in det_indices:
				union_mask |= masks[di]
				emb_accum.append(embs[di].unsqueeze(0))
			emb_avg = torch.mean(torch.cat(emb_accum, dim=0), dim=0)
			self.tracks[t_idx].update(union_mask, emb_avg, frame_idx)

		# Create new tracks for unmatched detections
		for det_idx in range(len(masks)):
			if det_idx in det_assigned:
				continue
			new_track = Track(self.next_id, masks[det_idx], embs[det_idx], frame_idx)
			self.tracks.append(new_track)
			self.next_id += 1

		# Age and prune missing tracks
		alive_tracks: List[Track] = []
		for tr in self.tracks:
			if tr.last_frame_idx < frame_idx:
				tr.missed += 1
			if tr.missed <= self.max_missed:
				alive_tracks.append(tr)
		self.tracks = alive_tracks

		# Prepare results for visualization: return current last_mask for all tracks updated this frame
		for tr in self.tracks:
			if tr.last_frame_idx == frame_idx:
				results.append((tr.track_id, tr.last_mask.copy(), float(tr.age)))
		return results


# ------------------------------
# Main pipeline
# ------------------------------

def run(video_path: str, outdir: str, stride: int, conf_thresh: float, iou_weight: float, sim_weight: float, iou_thresh: float, sim_thresh: float, max_missed: int, max_frames: int, max_persons: int, min_mask_area_frac: float, center_bias: float, disjoint_overlaps: bool, split_with_keypoints: bool, draw_keypoints: bool = False) -> None:
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video_path)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	segmenter = PersonSegmenter(device)
	embedder = ClipEmbedder(device)
	tracker = SemanticTracker(iou_weight=iou_weight, sim_weight=sim_weight, iou_thresh=iou_thresh, sim_thresh=sim_thresh, max_missed=max_missed)
	kp_detector = PersonKeypointDetector(device) if (split_with_keypoints or draw_keypoints) else None

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

	# Output video (process-only frames)
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	video_out_path = os.path.join(outdir, "tracked.mp4")
	out = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / max(1, stride)), (w, h))
	if not out.isOpened():
		# Fallback to avi
		video_out_path = os.path.splitext(video_out_path)[0] + ".avi"
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		out = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / max(1, stride)), (w, h))

	csv_path = os.path.join(outdir, "tracks.csv")
	csv_f = open(csv_path, "w", encoding="utf-8")
	csv_f.write("frame,track_id,x1,y1,x2,y2,age\n")

	labels_dir = os.path.join(outdir, "labels")
	os.makedirs(labels_dir, exist_ok=True)

	palette = {}
	def _color_for(track_id: int) -> Tuple[int, int, int]:
		if track_id not in palette:
			# pseudo-random stable color
			np.random.seed(track_id * 7919)
			palette[track_id] = tuple(int(x) for x in np.random.randint(0, 255, size=3))
		return palette[track_id]

	frame_idx = 0
	processed = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if frame_idx % stride != 0:
			frame_idx += 1
			continue

		masks_all, scores_all, probs_all = segmenter.segment(frame, conf_thresh=conf_thresh)
		# Select foreground masks (e.g., two closest fighters)
		selected_idx = _select_foreground_indices(masks_all, scores_all, h, w, max_persons=max_persons, min_area_frac=min_mask_area_frac, center_bias=center_bias)
		masks = [masks_all[i] for i in selected_idx]
		probs = [probs_all[i] for i in selected_idx]
		scores_sel = np.array([scores_all[i] for i in selected_idx], dtype=np.float32)
		# Resolve overlaps between selected masks to produce disjoint per-pixel assignments
		if disjoint_overlaps and len(masks) > 1:
			masks = _disjointify_masks(masks, probs, scores_sel)

		# If still fewer than desired persons and keypoint splitting is enabled, try to split an interlinked mask
		if split_with_keypoints and len(masks) < max_persons and len(masks) >= 1 and (kp_detector is not None):
			try:
				_, _, kps = kp_detector.detect(frame, conf_thresh=max(0.4, conf_thresh - 0.1))
				seeds: List[Tuple[int, int]] = []
				for k in kps:
					c = _keypoints_centroid(k)
					if c is not None:
						seeds.append(c)
				# Use the largest mask as the candidate to split
				if len(seeds) >= 2:
					areas = [m.sum() for m in masks]
					idx_largest = int(np.argmax(areas))
					mask_big = masks[idx_largest]
					# Keep only seeds that fall inside the big mask
					seeds_in = [(x, y) for (x, y) in seeds if 0 <= x < w and 0 <= y < h and mask_big[y, x]]
					if len(seeds_in) >= 2:
						parts = _watershed_split(mask_big, seeds_in, max_segments=max_persons)
						# Replace the largest mask with split parts, keep others
						new_masks: List[np.ndarray] = []
						for i, m in enumerate(masks):
							if i == idx_largest:
								continue
							new_masks.append(m)
						new_masks.extend(parts[: max_persons - (len(new_masks))])
						masks = new_masks
			except Exception:
				pass
		# Build crops around each mask's bounding box for embeddings
		boxes_for_emb = [_mask_to_box(m) for m in masks]
		crops = [_crop_with_padding(frame, box)[:, :, ::-1] for box in boxes_for_emb]  # to RGB
		embs = embedder.embed_batch(crops)  # [M,D]
		results = tracker.step(masks, embs, frame_idx)

		# Optional: detect keypoints for drawing
		kp_boxes = kp_scores = kp_points = None
		if draw_keypoints and (kp_detector is not None):
			try:
				kp_boxes, kp_scores, kp_points = kp_detector.detect(frame, conf_thresh=max(0.4, conf_thresh - 0.1))
			except Exception:
				kp_boxes = kp_scores = kp_points = None

		# Draw
		label_map = np.zeros((h, w), dtype=np.uint16)
		for tid, mask, age in results:
			color = _color_for(tid)
			# Overlay mask with transparency
			overlay = np.zeros_like(frame, dtype=np.uint8)
			overlay[mask] = color
			cv2.addWeighted(overlay, 0.5, frame, 0.5, 0.0, frame)
			# Draw contours for each connected component to emphasize separated parts
			comp_count, comp_ids = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
			for comp_id in range(1, comp_count):
				comp_mask = (comp_ids == comp_id)
				contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				for cnt in contours:
					cv2.drawContours(frame, [cnt], -1, color, 2)
			# Put label near mask bbox
			bbox = _mask_to_box(mask)
			x1, y1, x2, y2 = [int(v) for v in bbox]
			label = f"ID {tid} | age {int(age)}"
			cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
			# Draw keypoints if available: associate nearest keypoint detection by IoU
			if draw_keypoints and (kp_boxes is not None) and (kp_points is not None) and len(kp_boxes) > 0:
				# find best kp det by IoU between track bbox and kp bbox
				best_iou = 0.0
				best_kp = None
				for i in range(len(kp_boxes)):
					kb = kp_boxes[i]
					# compute IoU with track bbox
					xA = max(bbox[0], kb[0]); yA = max(bbox[1], kb[1])
					xB = min(bbox[2], kb[2]); yB = min(bbox[3], kb[3])
					inter = max(0.0, xB - xA) * max(0.0, yB - yA)
					area_a = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
					area_b = max(0.0, kb[2] - kb[0]) * max(0.0, kb[3] - kb[1])
					iou_val = inter / (area_a + area_b - inter + 1e-6)
					if iou_val > best_iou:
						best_iou = iou_val
						best_kp = kp_points[i]
				if best_kp is not None and best_iou >= 0.1:
					# Draw keypoints and skeleton
					kps_int = best_kp.astype(int)
					# COCO skeleton pairs
					skel = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),(5,1),(6,2),(1,3),(2,4),(3,4)]
					# Draw joints
					for j in range(kps_int.shape[0]):
						xj, yj = int(kps_int[j,0]), int(kps_int[j,1])
						cv2.circle(frame, (xj, yj), 3, color, -1, lineType=cv2.LINE_AA)
					# Draw limbs
					for (a,b) in skel:
						if a-1 < kps_int.shape[0] and b-1 < kps_int.shape[0]:
							pa = (int(kps_int[a-1,0]), int(kps_int[a-1,1]))
							pb = (int(kps_int[b-1,0]), int(kps_int[b-1,1]))
							cv2.line(frame, pa, pb, color, 2, lineType=cv2.LINE_AA)
			# Update label map (semantic annotation). Later overlaps are allowed to overwrite earlier ones.
			label_map[mask] = tid

		out.write(frame)

		# Write CSV lines
		for tid, mask, age in results:
			bbox = _mask_to_box(mask)
			x1, y1, x2, y2 = bbox.tolist()
			csv_f.write(f"{frame_idx},{tid},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{int(age)}\n")

		# Save semantic label map as 16-bit PNG (track IDs). 0 is background.
		label_path = os.path.join(labels_dir, f"{frame_idx:06d}.png")
		cv2.imwrite(label_path, label_map)

		processed += 1
		if processed % 50 == 0:
			print(f"Processed {processed} frames (every {stride}th) ...")
		if (max_frames > 0) and (processed >= max_frames):
			break

		frame_idx += 1

	cap.release()
	out.release()
	csv_f.close()
	print(f"Saved tracked video: {video_out_path}")
	print(f"Saved track CSV: {csv_path}")
	print(f"Saved semantic labels to folder: {labels_dir}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Semantic fighter tracking: detect persons, embed with CLIP, and associate across frames.")
	p.add_argument("--video", required=True, help="Path to input video")
	p.add_argument("--outdir", default="tracker_out", help="Directory to save outputs")
	p.add_argument("--stride", type=int, default=5, help="Process every Nth frame to speed up (default: 5)")
	p.add_argument("--conf", type=float, default=0.6, help="Detection confidence threshold (default: 0.6)")
	p.add_argument("--iou-weight", type=float, default=0.4, help="Weight for IoU in matching score (default: 0.4)")
	p.add_argument("--sim-weight", type=float, default=0.6, help="Weight for cosine similarity in matching score (default: 0.6)")
	p.add_argument("--iou-thresh", type=float, default=0.1, help="Minimum IoU to consider geometric match (default: 0.1)")
	p.add_argument("--sim-thresh", type=float, default=0.2, help="Minimum cosine similarity to consider semantic match (default: 0.2)")
	p.add_argument("--max-missed", type=int, default=20, help="Max missed steps before terminating a track (default: 20)")
	p.add_argument("--max-frames", type=int, default=400, help="Process at most this many processed frames (after stride). 0 for all (default: 400)")
	p.add_argument("--max-persons", type=int, default=2, help="Max number of foreground persons to keep per frame (default: 2)")
	p.add_argument("--min-mask-area-frac", type=float, default=0.005, help="Minimum mask area as fraction of frame to keep (default: 0.005)")
	p.add_argument("--center-bias", type=float, default=0.3, help="Center bias weight in [0,1] for foreground scoring (default: 0.3)")
	p.add_argument("--no-disjoint-overlaps", action="store_true", help="Disable per-pixel disjoint assignment between overlapping masks")
	p.add_argument("--no-keypoint-split", action="store_true", help="Disable keypoint-based splitting of interlinked persons")
	p.add_argument("--draw-keypoints", action="store_true", help="Draw person keypoints over tracks in the output video")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		video_path=args.video,
		outdir=args.outdir,
		stride=args.stride,
		conf_thresh=args.conf,
		iou_weight=args.iou_weight,
		sim_weight=args.sim_weight,
		iou_thresh=args.iou_thresh,
		sim_thresh=args.sim_thresh,
		max_missed=args.max_missed,
		max_frames=args.max_frames,
		max_persons=args.max_persons,
		min_mask_area_frac=args.min_mask_area_frac,
		center_bias=args.center_bias,
		disjoint_overlaps=(not args.no_disjoint_overlaps),
		split_with_keypoints=(not args.no_keypoint_split),
		draw_keypoints=args.draw_keypoints,
	) 