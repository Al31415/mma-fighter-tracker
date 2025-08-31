import argparse
import os
import math
from typing import List, Tuple, Dict
from pathlib import Path

import numpy as np
import cv2

_ultra = None
_create_tracker = None

# COCO skeleton connections (17 keypoints)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def _lazy_imports() -> None:
	global _ultra, _create_tracker
	if _ultra is None:
		from ultralytics import YOLO  # type: ignore
		_ultra = YOLO
	if _create_tracker is None:
		from boxmot.tracker_zoo import create_tracker  # type: ignore
		_create_tracker = create_tracker


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
	x1 = float(xs.min()); y1 = float(ys.min())
	x2 = float(xs.max() + 1); y2 = float(ys.max() + 1)
	return np.array([x1, y1, x2, y2], dtype=np.float32)


def _xyxy_to_ltwh(box: np.ndarray) -> Tuple[float, float, float, float]:
	x1, y1, x2, y2 = box.tolist()
	return (float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1)))


def _center_bias_score(mask: np.ndarray, h: int, w: int) -> float:
	ys, xs = np.where(mask)
	if ys.size == 0:
		return 0.0
	cx = float(xs.mean()); cy = float(ys.mean())
	cx0 = w * 0.5; cy0 = h * 0.5
	d = math.hypot(cx - cx0, cy - cy0)
	d_max = math.hypot(cx0, cy0)
	if d_max <= 1e-6:
		return 1.0
	return float(max(0.0, 1.0 - d / d_max))


def _disjointify(masks: List[np.ndarray], probs: List[np.ndarray], scores: np.ndarray) -> List[np.ndarray]:
	if not masks:
		return []
	h, w = masks[0].shape
	n = len(masks)
	stack = np.zeros((n, h, w), dtype=np.float32)
	for i in range(n):
		stack[i] = probs[i].astype(np.float32) * float(scores[i]) * masks[i].astype(np.float32)
	argmax_idx = stack.argmax(axis=0)
	max_vals = stack.max(axis=0)
	assigned = max_vals > 0.0
	out: List[np.ndarray] = []
	for i in range(n):
		out.append((argmax_idx == i) & assigned)
	return out


def _draw_keypoints_and_skeleton(frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], conf_threshold: float = 0.5) -> None:
	"""Draw keypoints and skeleton connections on frame"""
	if keypoints.shape[0] < 17:  # Need 17 COCO keypoints
		return
	
	# Draw skeleton connections
	for connection in COCO_SKELETON:
		kpt1_idx, kpt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed
		if kpt1_idx >= len(keypoints) or kpt2_idx >= len(keypoints):
			continue
		
		kpt1, kpt2 = keypoints[kpt1_idx], keypoints[kpt2_idx]
		x1, y1, c1 = int(kpt1[0]), int(kpt1[1]), kpt1[2]
		x2, y2, c2 = int(kpt2[0]), int(kpt2[1]), kpt2[2]
		
		if c1 > conf_threshold and c2 > conf_threshold:
			cv2.line(frame, (x1, y1), (x2, y2), color, 2)
	
	# Draw keypoints
	for i, kpt in enumerate(keypoints):
		x, y, conf = int(kpt[0]), int(kpt[1]), kpt[2]
		if conf > conf_threshold:
			cv2.circle(frame, (x, y), 3, color, -1)
			cv2.circle(frame, (x, y), 3, (0, 0, 0), 1)


def _find_botsort_cfg() -> str:
	try:
		import boxmot  # type: ignore
		root = os.path.dirname(boxmot.__file__)
		cands = [
			os.path.join(root, "configs", "trackers", "botsort.yaml"),
			os.path.join(root, "cfg", "tracker", "botsort.yaml"),
			os.path.join(root, "cfg", "trackers", "botsort.yaml"),
		]
		for p in cands:
			if os.path.isfile(p):
				return p
	except Exception:
		pass
	return "botsort.yaml"


def run(video: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, min_bbox_area_frac: float, center_bias: float, morph_ks: int) -> None:
	_lazy_imports()
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}", flush=True)

	seg_model = _ultra("yolov8x-seg.pt")
	pose_model = _ultra("yolov8x-pose.pt")  # Add pose model
	tracker_cfg = _find_botsort_cfg()
	# Disable ReID to reduce memory and avoid CLIP load
	tracker = _create_tracker(
		tracker_type="botsort",
		tracker_config=tracker_cfg,
		reid_weights=Path("osnet_x0_25_msmt17.pt"),  # Default OSNet model as Path
		device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
		half=False
	)

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
			palette[tid] = tuple(int(x) for x in np.random.randint(0,255,size=3))
		return palette[tid]

	last_boxes: Dict[int, np.ndarray] = {}

	frame_idx = 0
	processed = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if frame_idx % stride != 0:
			frame_idx += 1; continue

		res = seg_model.predict(source=frame, conf=float(conf), iou=float(iou), verbose=False)[0]
		masks_raw = res.masks; boxes_raw = res.boxes
		if (masks_raw is None) or (boxes_raw is None):
			frame_idx += 1; continue
		masks_np = masks_raw.data.cpu().numpy().astype(bool)
		probs_np = masks_raw.data.cpu().numpy().astype(np.float32)
		boxes = boxes_raw.xyxy.cpu().numpy(); scores = boxes_raw.conf.cpu().numpy(); classes = boxes_raw.cls.cpu().numpy()
		# Resize masks to frame size
		masks_rs: List[np.ndarray] = []; probs_rs: List[np.ndarray] = []
		for i in range(masks_np.shape[0]):
			m = masks_np[i]; p = probs_np[i]
			m_r = cv2.resize((m.astype(np.uint8)*255), (w, h), interpolation=cv2.INTER_NEAREST) > 0
			p_r = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
			masks_rs.append(m_r); probs_rs.append(p_r)
		masks = masks_rs; probs = probs_rs

		# Candidate scoring (area, bbox area, center-bias, temporal prior)
		frame_area = float(w*h)
		cands: List[Tuple[float,int]] = []
		for i in range(len(masks)):
			if int(classes[i]) != 0: continue
			area_frac = float(masks[i].sum())/max(1.0, frame_area)
			bx = boxes[i]
			bbox_area_frac = float(max(0.0,bx[2]-bx[0])*max(0.0,bx[3]-bx[1]))/max(1.0, frame_area)
			if area_frac < min_mask_area_frac: continue
			if bbox_area_frac < min_bbox_area_frac: continue
			cb = _center_bias_score(masks[i], h, w)
			prior_boost = 1.0
			for tb in last_boxes.values():
				xA = max(tb[0], bx[0]); yA = max(tb[1], bx[1]); xB = min(tb[2], bx[2]); yB = min(tb[3], bx[3])
				inter = max(0.0, xB-xA)*max(0.0, yB-yA)
				area_tb = max(0.0,tb[2]-tb[0])*max(0.0,tb[3]-tb[1])
				area_bx = max(0.0,bx[2]-bx[0])*max(0.0,bx[3]-bx[1])
				ioup = inter/(area_tb+area_bx-inter+1e-6)
				prior_boost = max(prior_boost, 1.0 + 0.5*ioup)
			s = ((1.0-center_bias)*area_frac + center_bias*cb) * float(scores[i]) * prior_boost
			cands.append((s, i))
		cands.sort(key=lambda x: x[0], reverse=True)
		keep_idx = [i for _, i in cands[:max(0, max_persons)]]
		masks_sel = [masks[i] for i in keep_idx]
		probs_sel = [probs[i] for i in keep_idx]
		scores_sel = np.array([scores[i] for i in keep_idx], dtype=np.float32)
		boxes_sel = [boxes[i] for i in keep_idx]

		if len(masks_sel) > 1:
			masks_sel = _disjointify(masks_sel, probs_sel, scores_sel)

		if morph_ks > 1:
			k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
			masks_sel = [cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) for m in masks_sel]

		# Build BoT-SORT detections and update
		dets = []
		for bx, sc in zip(boxes_sel, [0.9]*len(boxes_sel)):
			dets.append([bx[0], bx[1], bx[2], bx[3], sc, 0])
		dets_np = np.array(dets, dtype=np.float32) if dets else np.zeros((0,6), dtype=np.float32)
		tracks = tracker.update(dets_np, frame)

		# Run pose detection on the frame
		pose_results = pose_model.predict(source=frame, conf=float(conf), verbose=False)[0]
		pose_keypoints = []
		pose_boxes = []
		if pose_results.keypoints is not None and pose_results.boxes is not None:
			pose_keypoints = pose_results.keypoints.data.cpu().numpy()  # Shape: (N, 17, 3)
			pose_boxes = pose_results.boxes.xyxy.cpu().numpy()  # Shape: (N, 4)

		label_map = np.zeros((h, w), dtype=np.uint16)
		for tr in tracks:
			x1, y1, x2, y2, tid = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3]), int(tr[4])
			# best mask for label map
			best_iou = 0.0; best_m = None
			for m, bx in zip(masks_sel, boxes_sel):
				xA = max(x1, bx[0]); yA = max(y1, bx[1]); xB = min(x2, bx[2]); yB = min(y2, bx[3])
				inter = max(0.0, xB-xA)*max(0.0, yB-yA)
				area_t = max(0.0,x2-x1)*max(0.0,y2-y1)
				area_b = max(0.0,bx[2]-bx[0])*max(0.0,bx[3]-bx[1])
				iou = inter/(area_t+area_b-inter+1e-6)
				if iou > best_iou:
					best_iou = iou; best_m = m
			# draw thick semantic outline
			color = _color_for(tid)
			if best_m is not None and best_iou >= 0.1:
				contours, _ = cv2.findContours(best_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				for cnt in contours:
					cv2.drawContours(frame, [cnt], -1, (0,0,0), 8)
					cv2.drawContours(frame, [cnt], -1, color, 4)
				label_map[best_m] = tid
			else:
				cnt = np.array([[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.int32).reshape(-1,1,2)
				cv2.drawContours(frame, [cnt], -1, (0,0,0), 8)
				cv2.drawContours(frame, [cnt], -1, color, 4)
				label_map[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = tid
			
			# Find best matching pose for this track
			best_pose_iou = 0.0
			best_pose_kpts = None
			for pose_box, pose_kpts in zip(pose_boxes, pose_keypoints):
				# Calculate IoU between track box and pose box
				px1, py1, px2, py2 = pose_box
				xA = max(x1, px1); yA = max(y1, py1); xB = min(x2, px2); yB = min(y2, py2)
				inter = max(0.0, xB-xA) * max(0.0, yB-yA)
				area_track = max(0.0, x2-x1) * max(0.0, y2-y1)
				area_pose = max(0.0, px2-px1) * max(0.0, py2-py1)
				pose_iou = inter / (area_track + area_pose - inter + 1e-6)
				if pose_iou > best_pose_iou:
					best_pose_iou = pose_iou
					best_pose_kpts = pose_kpts
			
			# Draw keypoints and skeleton if we found a good match
			if best_pose_kpts is not None and best_pose_iou > 0.3:
				_draw_keypoints_and_skeleton(frame, best_pose_kpts, color)
			
			# ID banner
			label = f"ID {tid}"
			size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
			bx2 = x1 + size[0] + 10; by1 = max(0, y1 - size[1] - 10)
			cv2.rectangle(frame, (x1, by1), (bx2, y1), (0,0,0), -1)
			cv2.putText(frame, label, (x1 + 5, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
			csv_f.write(f"{frame_idx},{tid},{float(x1)},{float(y1)},{float(x2)},{float(y2)}\n")

		out.write(frame)
		cv2.imwrite(os.path.join(labels_dir, f"{frame_idx:06d}.png"), label_map)

		# update priors
		last_boxes = {}
		for tr in tracks:
			bb = np.array([tr[0], tr[1], tr[2], tr[3]], dtype=np.float32)
			last_boxes[int(tr[4])] = bb

		processed += 1
		if processed % 50 == 0:
			print(f"Processed {processed} frames (every {stride}th) ...", flush=True)
		if (max_frames>0) and (processed>=max_frames):
			break
		frame_idx += 1

	cap.release(); out.release(); csv_f.close()
	print(f"Saved tracked video: {video_out_path}", flush=True)
	print(f"Saved track CSV: {csv_path}", flush=True)
	print(f"Saved semantic labels to folder: {labels_dir}", flush=True)


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="BoT-SORT BMP tracker with occlusion handling and semantic outlines")
	p.add_argument("--video", required=True)
	p.add_argument("--outdir", default="tracker_out_bmp_botsort")
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--conf", type=float, default=0.5)
	p.add_argument("--iou", type=float, default=0.45)
	p.add_argument("--max-frames", type=int, default=400)
	p.add_argument("--max-persons", type=int, default=3)
	p.add_argument("--min-mask-area-frac", type=float, default=0.02)
	p.add_argument("--min-bbox-area-frac", type=float, default=0.01)
	p.add_argument("--center-bias", type=float, default=0.7)
	p.add_argument("--morph-ks", type=int, default=3)
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
		min_bbox_area_frac=args.min_bbox_area_frac,
		center_bias=args.center_bias,
		morph_ks=args.morph_ks,
	) 