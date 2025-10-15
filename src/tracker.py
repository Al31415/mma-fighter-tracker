"""
YOLO + BotSORT + VIPS ReID Tracker
Best occlusion handling: YOLO detection + BotSORT motion prediction + OSNet ReID
"""

import argparse
import os
import math
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F


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


def _mask_to_box(mask: np.ndarray) -> np.ndarray:
	ys, xs = np.where(mask)
	if ys.size == 0 or xs.size == 0:
		return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
	x1 = float(xs.min())
	y1 = float(ys.min())
	x2 = float(xs.max() + 1)
	y2 = float(ys.max() + 1)
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _center_bias_score(mask: np.ndarray, h: int, w: int) -> float:
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


def _select_person_indices(result, h: int, w: int, max_persons: int, min_area_frac: float, center_bias: float) -> List[int]:
	"""Select top N persons by score"""
	idxs: List[int] = []
	if result.masks is None or result.boxes is None:
		return idxs
	masks = result.masks.data.cpu().numpy().astype(bool)
	classes = result.boxes.cls.cpu().numpy()
	scores = result.boxes.conf.cpu().numpy()
	frame_area = float(h * w)
	cands = []
	for i in range(masks.shape[0]):
		if int(classes[i]) != 0:  # person class
			continue
		area_frac = float(masks[i].sum()) / max(1.0, frame_area)
		if area_frac < min_area_frac:
			continue
		cb = _center_bias_score(masks[i], h, w)
		score = ((1.0 - center_bias) * area_frac + center_bias * cb) * float(scores[i])
		cands.append((score, i))
	cands.sort(key=lambda x: x[0], reverse=True)
	return [i for _, i in cands[:max(0, max_persons)]]


def _find_botsort_cfg() -> str:
	"""Locate BotSORT config file"""
	try:
		import boxmot
		root = os.path.dirname(boxmot.__file__)
		candidates = [
			os.path.join(root, "configs", "botsort.yaml"),
			os.path.join(root, "configs", "trackers", "botsort.yaml"),
			os.path.join(root, "cfg", "tracker", "botsort.yaml"),
			os.path.join(root, "cfg", "trackers", "botsort.yaml"),
		]
		for p in candidates:
			if os.path.isfile(p):
				return p
	except Exception:
		pass
	return "botsort.yaml"


class VIPSReIDEmbedder:
	"""OSNet person ReID for enhanced identity features"""
	def __init__(self, device: torch.device, model_name: str = "osnet_x1_0"):
		print(f"[VIPS-ReID] Loading {model_name}...")
		try:
			import torchreid
			self.torchreid = torchreid
			
			self.model = torchreid.models.build_model(
				name=model_name,
				num_classes=1000,
				loss='softmax',
				pretrained=True
			)
			self.model = self.model.to(device).eval()
			self.device = device
			
			from torchvision import transforms
			self.transform = transforms.Compose([
				transforms.ToPILImage(),
				transforms.Resize((256, 128)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])
			print(f"[VIPS-ReID] OSNet loaded successfully!")
			self.use_reid = True
			
		except Exception as e:
			print(f"[VIPS-ReID] WARNING: OSNet not available, ReID features disabled")
			print(f"[VIPS-ReID] Error: {e}")
			self.use_reid = False
			self.torchreid = None

	@torch.no_grad()
	def embed_batch(self, crops_rgb: List[np.ndarray]) -> torch.Tensor:
		if not self.use_reid or not crops_rgb:
			return torch.empty((len(crops_rgb), 512))
		
		batch = []
		for crop in crops_rgb:
			if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
				continue
			tensor = self.transform(crop)
			batch.append(tensor)
		
		if not batch:
			return torch.empty((0, 512))
		
		batch_tensor = torch.stack(batch).to(self.device)
		features = self.model(batch_tensor)
		features = F.normalize(features, dim=1)
		return features.detach().cpu()


def crop_from_box(frame_bgr: np.ndarray, box_xyxy: np.ndarray, pad: float = 0.1) -> np.ndarray:
	"""Crop person from frame with padding"""
	h, w = frame_bgr.shape[:2]
	x1, y1, x2, y2 = box_xyxy
	bw = x2 - x1
	bh = y2 - y1
	px = bw * pad
	py = bh * pad
	x1p = int(max(0, math.floor(x1 - px)))
	y1p = int(max(0, math.floor(y1 - py)))
	x2p = int(min(w, math.ceil(x2 + px)))
	y2p = int(min(h, math.ceil(y2 + py)))
	return frame_bgr[y1p:y2p, x1p:x2p]


def run(video: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, center_bias: float, reid_model: str, use_reid_features: bool) -> None:
	
	os.makedirs(outdir, exist_ok=True)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	
	fps, n_frames, w, h = _read_video_meta(video)
	print(f"Video: {w}x{h} @ {fps:.2f}fps, {n_frames} frames")
	
	# YOLO for detection (handles occlusions better)
	from ultralytics import YOLO
	print("Loading YOLOv8-seg...")
	seg_model = YOLO("yolov8x-seg.pt")
	
	# BotSORT for tracking (motion prediction + occlusion handling)
	from boxmot.tracker_zoo import create_tracker
	tracker_cfg = _find_botsort_cfg()
	reid_w = Path("osnet_x0_25_msmt17.pt")
	print(f"Loading BotSORT tracker...")
	tracker = create_tracker(
		tracker_type="botsort",
		tracker_config=tracker_cfg,
		reid_weights=reid_w,
		device="0" if torch.cuda.is_available() else "cpu",
		half=False
	)
	
	# VIPS ReID embedder for enhanced identity
	reid_embedder = None
	if use_reid_features:
		reid_embedder = VIPSReIDEmbedder(device, model_name=reid_model)
	
	print(f"Config: conf={conf}, iou={iou}, max_persons={max_persons}")
	print(f"VIPS ReID: {'ENABLED' if use_reid_features else 'DISABLED (using BotSORT default ReID)'}")
	
	cap = cv2.VideoCapture(video)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video}")

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	video_out_path = os.path.join(outdir, "tracked.mp4")
	out = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / max(1, stride)), (w, h))
	if not out.isOpened():
		video_out_path = os.path.splitext(video_out_path)[0] + ".avi"
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		out = cv2.VideoWriter(video_out_path, fourcc, max(1.0, fps / max(1, stride)), (w, h))

	csv_path = os.path.join(outdir, "tracks.csv")
	csv_f = open(csv_path, "w", encoding="utf-8")
	csv_f.write("frame,track_id,x1,y1,x2,y2\n")

	labels_dir = os.path.join(outdir, "labels")
	os.makedirs(labels_dir, exist_ok=True)

	palette: Dict[int, Tuple[int, int, int]] = {}
	def _color_for(tid: int) -> Tuple[int, int, int]:
		if tid not in palette:
			np.random.seed(tid * 7919)
			palette[tid] = tuple(int(x) for x in np.random.randint(0, 255, size=3))
		return palette[tid]
	
	# VIPS ReID identity storage
	id_track_map: Dict[int, str] = {}
	id_protos: Dict[str, torch.Tensor] = {}
	identity_names = ["FighterA", "FighterB", "FighterC"]
	
	def assign_identity(tid: int, emb: torch.Tensor) -> str:
		"""Assign fighter identity using VIPS ReID features"""
		if not id_protos:
			name = identity_names[0]
			id_protos[name] = emb.clone()
			return name
		
		# Find best match
		best_name = None
		best_sim = -1.0
		for name, proto in id_protos.items():
			sim = float(F.cosine_similarity(emb.unsqueeze(0), proto.unsqueeze(0), dim=1).item())
			if sim > best_sim:
				best_sim = sim
				best_name = name
		
		# Create new identity if similarity too low
		if best_sim < 0.4 and len(id_protos) < len(identity_names):
			name = identity_names[len(id_protos)]
			id_protos[name] = emb.clone()
			return name
		
		# Update prototype
		if best_name:
			id_protos[best_name] = F.normalize(0.7 * id_protos[best_name] + 0.3 * emb, dim=0)
			return best_name
		
		return identity_names[0]

	frame_idx = 0
	processed = 0
	print("\nProcessing...")
	
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if frame_idx % stride != 0:
			frame_idx += 1
			continue

		# YOLO detection
		res = seg_model.predict(source=frame, conf=float(conf), iou=float(iou), verbose=False, device=0 if torch.cuda.is_available() else None)
		result = res[0]
		
		# Select persons
		sel = _select_person_indices(result, h, w, max_persons=max_persons, min_area_frac=min_mask_area_frac, center_bias=center_bias)
		
		# Prepare detections for BotSORT
		boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
		scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,), dtype=np.float32)
		classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else np.zeros((0,), dtype=np.float32)
		
		dets = []
		for i in sel:
			if int(classes[i]) != 0:  # person class
				continue
			dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i]), 0])
		dets_np = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)
		
		# BotSORT tracking (handles occlusions!)
		tracks = tracker.update(dets_np, frame)
		
		# Get masks for visualization
		label_map = np.zeros((h, w), dtype=np.uint16)
		sel_boxes = []
		sel_masks = []
		if result.boxes is not None and result.masks is not None and len(sel) > 0:
			all_xyxy = result.boxes.xyxy.cpu().numpy()
			masks_np = result.masks.data.cpu().numpy().astype(bool)
			for i in sel:
				sel_boxes.append(all_xyxy[i])
				sel_masks.append(masks_np[i])
		
		# Process tracks with VIPS ReID
		for tr in tracks:
			x1, y1, x2, y2, tid = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3]), int(tr[4])
			box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
			
			# Find matching mask
			best_iou = 0.0
			best_mask = None
			for j, db in enumerate(sel_boxes):
				xA = max(box_arr[0], db[0])
				yA = max(box_arr[1], db[1])
				xB = min(box_arr[2], db[2])
				yB = min(box_arr[3], db[3])
				inter = max(0.0, xB - xA) * max(0.0, yB - yA)
				area_bx = max(0.0, box_arr[2] - box_arr[0]) * max(0.0, box_arr[3] - box_arr[1])
				area_db = max(0.0, db[2] - db[0]) * max(0.0, db[3] - db[1])
				iou_td = inter / (area_bx + area_db - inter + 1e-6)
				if iou_td > best_iou:
					best_iou = iou_td
					best_mask = sel_masks[j]
			
			# VIPS ReID identity assignment
			fighter_name = f"Fighter {tid}"
			if use_reid_features and reid_embedder and reid_embedder.use_reid:
				if tid not in id_track_map:
					crop = crop_from_box(frame, box_arr, pad=0.1)
					crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
					embs = reid_embedder.embed_batch([crop_rgb])
					if embs.shape[0] > 0:
						fighter_name = assign_identity(tid, embs[0])
						id_track_map[tid] = fighter_name
				else:
					fighter_name = id_track_map[tid]
			
			color = _color_for(tid)
			
			# Draw mask if available
			if best_mask is not None and best_iou >= 0.1:
				mask_uint8 = (best_mask.astype(np.uint8) * 255)
				mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST) > 0
				over = np.zeros_like(frame, dtype=np.uint8)
				over[mask_resized] = color
				cv2.addWeighted(over, 0.4, frame, 0.6, 0.0, frame)
				label_map[mask_resized] = tid
			else:
				# Draw bbox if no mask
				label_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = tid
			
			# Draw bounding box and label
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
			label = f"{fighter_name} | ID {tid}"
			cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
			
			csv_f.write(f"{frame_idx},{tid},{float(x1)},{float(y1)},{float(x2)},{float(y2)}\n")
		
		out.write(frame)
		cv2.imwrite(os.path.join(labels_dir, f"frame_{frame_idx:06d}.png"), label_map)
		
		processed += 1
		if processed % 20 == 0:
			active_tracks = len(tracks) if isinstance(tracks, (list, np.ndarray)) else 0
			print(f"  Frame {frame_idx}: {active_tracks} active tracks")
		
		frame_idx += 1
		if max_frames > 0 and processed >= max_frames:
			break

	cap.release()
	out.release()
	csv_f.close()

	print(f"\nDone! Processed {processed} frames")
	print(f"  Video: {video_out_path}")
	print(f"  CSV: {csv_path}")
	print(f"  Labels: {labels_dir}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="YOLO + BotSORT + VIPS ReID - Best Occlusion Handling")
	p.add_argument("--video", type=str, required=True)
	p.add_argument("--outdir", type=str, default="yolo_botsort_vips_out")
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--conf", type=float, default=0.4, help="YOLO confidence (lower for occlusions)")
	p.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU")
	p.add_argument("--max-frames", type=int, default=400)
	p.add_argument("--max-persons", type=int, default=3, help="Max persons to track")
	p.add_argument("--min-mask-area-frac", type=float, default=0.002, help="Very low for occluded fighters")
	p.add_argument("--center-bias", type=float, default=0.15, help="Low for edge fighters")
	p.add_argument("--reid-model", type=str, default="osnet_x1_0")
	p.add_argument("--no-vips-reid", action="store_true", help="Disable VIPS ReID (use BotSORT default)")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		video=args.video,
		outdir=args.outdir,
		stride=args.stride,
		conf=args.conf,
		iou=args.iou,
		max_frames=args.max_frames,
		max_persons=args.max_persons,
		min_mask_area_frac=args.min_mask_area_frac,
		center_bias=args.center_bias,
		reid_model=args.reid_model,
		use_reid_features=(not args.no_vips_reid),
	)

