import argparse
import os
import math
from typing import List, Tuple, Dict

import numpy as np
import cv2

_ultra = None
_deepsort = None


def _lazy_imports() -> None:
	global _ultra, _deepsort
	if _ultra is None:
		from ultralytics import YOLO  # type: ignore
		_ultra = YOLO
	if _deepsort is None:
		from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
		_deepsort = DeepSort


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


def run(video: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, min_bbox_area_frac: float, center_bias: float, morph_ks: int) -> None:
	_lazy_imports()
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}")

	seg_model = _ultra("yolov8x-seg.pt")
	tracker = _deepsort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, embedder="mobilenet", polygon=False)

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

	# Track prior: last boxes for area-prior selection
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
		masks_raw = res.masks
		boxes_raw = res.boxes
		if (masks_raw is None) or (boxes_raw is None):
			frame_idx += 1
			continue
		masks_np = masks_raw.data.cpu().numpy().astype(bool)  # [N,Hm,Wm]
		probs_np = masks_raw.data.cpu().numpy().astype(np.float32)  # [N,Hm,Wm] as soft
		boxes = boxes_raw.xyxy.cpu().numpy(); scores = boxes_raw.conf.cpu().numpy(); classes = boxes_raw.cls.cpu().numpy()
		# Resize masks to frame size
		masks_rs: List[np.ndarray] = []
		probs_rs: List[np.ndarray] = []
		for i in range(masks_np.shape[0]):
			m = masks_np[i]; p = probs_np[i]
			m_r = cv2.resize((m.astype(np.uint8)*255), (w, h), interpolation=cv2.INTER_NEAREST) > 0
			p_r = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
			masks_rs.append(m_r); probs_rs.append(p_r)
		masks = masks_rs; probs = probs_rs

		# Filter by person class, area, bbox size, and center bias
		frame_area = float(w*h)
		cands: List[Tuple[float,int]] = []
		for i in range(len(masks)):
			if int(classes[i]) != 0:  # person class
				continue
			area_frac = float(masks[i].sum())/max(1.0, frame_area)
			bx = boxes[i]
			bbox_area_frac = float(max(0.0, bx[2]-bx[0])*max(0.0, bx[3]-bx[1]))/max(1.0, frame_area)
			if area_frac < min_mask_area_frac: continue
			if bbox_area_frac < min_bbox_area_frac: continue
			cb = _center_bias_score(masks[i], h, w)
			# temporal prior: boost candidates near last boxes (if any)
			prior_boost = 1.0
			for tb in last_boxes.values():
				xA = max(tb[0], bx[0]); yA = max(tb[1], bx[1]); xB = min(tb[2], bx[2]); yB = min(tb[3], bx[3])
				inter = max(0.0, xB-xA)*max(0.0, yB-yA)
				area_tb = max(0.0,tb[2]-tb[0])*max(0.0,tb[3]-tb[1])
				area_bx = max(0.0,bx[2]-bx[0])*max(0.0,bx[3]-bx[1])
				ioup = inter/(area_tb+area_bx-inter+1e-6)
				prior_boost = max(prior_boost, 1.0 + 0.5*ioup)
			score = ((1.0-center_bias)*area_frac + center_bias*cb) * float(scores[i]) * prior_boost
			cands.append((score, i))
		cands.sort(key=lambda x: x[0], reverse=True)
		keep_idx = [i for _, i in cands[:max(0, max_persons)]]
		masks_sel = [masks[i] for i in keep_idx]
		probs_sel = [probs[i] for i in keep_idx]
		scores_sel = np.array([scores[i] for i in keep_idx], dtype=np.float32)
		boxes_sel = [boxes[i] for i in keep_idx]

		# Disjoint per-pixel assignment for overlap
		if len(masks_sel) > 1:
			masks_sel = _disjointify(masks_sel, probs_sel, scores_sel)

		# Morphological smoothing
		if morph_ks > 1:
			k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
			masks_sel = [cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool) for m in masks_sel]

		# Build DeepSORT bboxes and update
		bbs = []
		for bx in boxes_sel:
			ltwh = _xyxy_to_ltwh(bx)
			bbs.append((list(ltwh), 0.9, 0))
		tracks = tracker.update_tracks(bbs, frame=frame)

		# Visualization + outputs
		label_map = np.zeros((h, w), dtype=np.uint16)
		# match masks to tracks by IoU (only for label map), keep video clean with thick borders + ID
		for t in tracks:
			if not t.is_confirmed():
				continue
			tid = int(t.track_id)
			ltrb = t.to_ltrb(orig=True, orig_strict=True)
			if ltrb is None:
				ltrb = t.to_ltrb()
			x1,y1,x2,y2 = [int(v) for v in ltrb]
			# For label map, optionally assign best mask if available; else fill bbox
			best_iou = 0.0; best_m = None
			for m, bx in zip(masks_sel, boxes_sel):
				xA = max(ltrb[0], bx[0]); yA = max(ltrb[1], bx[1])
				xB = min(ltrb[2], bx[2]); yB = min(ltrb[3], bx[3])
				inter = max(0.0, xB-xA)*max(0.0, yB-yA)
				area_t = max(0.0,ltrb[2]-ltrb[0])*max(0.0,ltrb[3]-ltrb[1])
				area_b = max(0.0,bx[2]-bx[0])*max(0.0,bx[3]-bx[1])
				iou = inter/(area_t+area_b-inter+1e-6)
				if iou > best_iou:
					best_iou = iou; best_m = m
			if best_m is not None and best_iou >= 0.1:
				label_map[best_m] = tid
			else:
				label_map[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = tid
			# Draw thick semantic outline (mask contours) and ID label
			color = _color_for(tid)
			if best_m is not None and best_iou >= 0.1:
				contours, _ = cv2.findContours(best_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				for cnt in contours:
					cv2.drawContours(frame, [cnt], -1, (0,0,0), 8)
					cv2.drawContours(frame, [cnt], -1, color, 4)
				bb = _mask_to_box(best_m); tx1, ty1, tx2, ty2 = [int(v) for v in bb]
			else:
				# Fallback: draw bbox contour
				cnt = np.array([[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.int32).reshape(-1,1,2)
				cv2.drawContours(frame, [cnt], -1, (0,0,0), 8)
				cv2.drawContours(frame, [cnt], -1, color, 4)
				tx1, ty1, tx2, ty2 = x1, y1, x2, y2
			label = f"ID {tid}"
			size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
			bx2 = tx1 + size[0] + 10; by1 = max(0, ty1 - size[1] - 10)
			cv2.rectangle(frame, (tx1, by1), (bx2, ty1), (0,0,0), -1)
			cv2.putText(frame, label, (tx1 + 5, max(0, ty1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
			# CSV line
			csv_f.write(f"{frame_idx},{tid},{float(x1)},{float(y1)},{float(x2)},{float(y2)}\n")

		out.write(frame)
		cv2.imwrite(os.path.join(labels_dir, f"{frame_idx:06d}.png"), label_map)

		# Update priors
		last_boxes = {}
		for t in tracks:
			if t.is_confirmed():
				bb = t.to_ltrb()
				last_boxes[int(t.track_id)] = np.array(bb, dtype=np.float32)

		processed += 1
		if processed % 50 == 0:
			print(f"Processed {processed} frames (every {stride}th) ...")
		if (max_frames>0) and (processed>=max_frames):
			break
		frame_idx += 1

	cap.release(); out.release(); csv_f.close()
	print(f"Saved tracked video: {video_out_path}")
	print(f"Saved track CSV: {csv_path}")
	print(f"Saved semantic labels to folder: {labels_dir}")


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Robust BMP tracker with occlusion handling and better selection")
	p.add_argument("--video", required=True)
	p.add_argument("--outdir", default="tracker_out_bmp_occl")
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--conf", type=float, default=0.5)
	p.add_argument("--iou", type=float, default=0.45)
	p.add_argument("--max-frames", type=int, default=400)
	p.add_argument("--max-persons", type=int, default=3)
	p.add_argument("--min-mask-area-frac", type=float, default=0.02)
	p.add_argument("--min-bbox-area-frac", type=float, default=0.01)
	p.add_argument("--center-bias", type=float, default=0.7)
	p.add_argument("--morph-ks", type=int, default=3, help="Morphological kernel size for mask smoothing")
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