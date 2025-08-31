import argparse
import os
import math
from typing import List, Tuple
from pathlib import Path

import numpy as np
import cv2

_ultra = None
_botsort = None


def _lazy_imports():
	global _ultra, _botsort
	if _ultra is None:
		from ultralytics import YOLO  # type: ignore
		_ultra = YOLO
	if _botsort is None:
		from boxmot.tracker_zoo import create_tracker  # type: ignore
		_botsort = create_tracker


def _find_botsort_cfg() -> str:
	# Try common locations inside boxmot package
	try:
		import boxmot  # type: ignore
		root = os.path.dirname(boxmot.__file__)
		candidates = [
			os.path.join(root, "configs", "trackers", "botsort.yaml"),
			os.path.join(root, "cfg", "tracker", "botsort.yaml"),
			os.path.join(root, "cfg", "trackers", "botsort.yaml"),
		]
		for p in candidates:
			if os.path.isfile(p):
				return p
	except Exception:
		pass
	# Fallback: rely on working directory relative path if user provides
	return "botsort.yaml"


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
	idxs: List[int] = []
	if result.masks is None or result.boxes is None:
		return idxs
	masks = result.masks.data.cpu().numpy().astype(bool)  # [N,Hm,Wm]
	classes = result.boxes.cls.cpu().numpy()
	scores = result.boxes.conf.cpu().numpy()
	frame_area = float(h * w)
	cands = []
	for i in range(masks.shape[0]):
		if int(classes[i]) != 0:
			continue
		area_frac = float(masks[i].sum()) / max(1.0, frame_area)
		if area_frac < min_area_frac:
			continue
		cb = _center_bias_score(masks[i], h, w)
		score = ((1.0 - center_bias) * area_frac + center_bias * cb) * float(scores[i])
		cands.append((score, i))
	cands.sort(key=lambda x: x[0], reverse=True)
	return [i for _, i in cands[:max(0, max_persons)]]


def run(video_path: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, center_bias: float) -> None:
	_lazy_imports()
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video_path)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}")

	seg_model = _ultra("yolov8x-seg.pt")
	tracker_cfg = _find_botsort_cfg()
	# Use a commonly available ReID weight; boxmot will download if missing
	reid_w = "osnet_x0_25_msmt17.pt"
	tracker = _botsort(tracker_type="botsort", tracker_config=tracker_cfg, reid_weights=Path(reid_w), device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu", half=False)

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

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

	palette = {}
	def _color_for(track_id: int) -> Tuple[int, int, int]:
		if track_id not in palette:
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

		seg_res = seg_model.predict(source=frame, conf=conf, iou=iou, verbose=False, device=0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else None)
		res = seg_res[0]

		# Select two persons
		sel = _select_person_indices(res, h, w, max_persons=max_persons, min_area_frac=min_mask_area_frac, center_bias=center_bias)

		boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
		scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)
		classes = res.boxes.cls.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)

		# Build detections (xyxy, conf, cls) for BoT-SORT
		dets = []
		for i in sel:
			if int(classes[i]) != 0:
				continue
			dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i]), 0])
		dets_np = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)

		# Track
		tracks = tracker.update(dets_np, frame)

		# Visualization + outputs
		label_map = np.zeros((h, w), dtype=np.uint16)
		for tr in tracks:
			# BoT-SORT returns [x1,y1,x2,y2,track_id,cls,conf]
			x1, y1, x2, y2, tid = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3]), int(tr[4])
			color = _color_for(tid)
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(frame, f"ID {tid}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
			label_map[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = tid
			csv_f.write(f"{frame_idx},{tid},{float(x1)},{float(y1)},{float(x2)},{float(y2)}\n")

		out.write(frame)
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
	p = argparse.ArgumentParser(description="YOLOv8-seg + BoT-SORT tracking")
	p.add_argument("--video", required=True, help="Path to input video")
	p.add_argument("--outdir", default="tracker_out_botsort", help="Directory to save outputs")
	p.add_argument("--stride", type=int, default=5, help="Process every Nth frame (default: 5)")
	p.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold (default: 0.5)")
	p.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold (default: 0.45)")
	p.add_argument("--max-frames", type=int, default=400, help="Max processed frames after stride; 0 for all (default: 400)")
	p.add_argument("--max-persons", type=int, default=2, help="Max number of persons per frame (default: 2)")
	p.add_argument("--min-mask-area-frac", type=float, default=0.005, help="Minimum mask area fraction (default: 0.005)")
	p.add_argument("--center-bias", type=float, default=0.3, help="Center bias weight in [0,1] (default: 0.3)")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run(
		video_path=args.video,
		outdir=args.outdir,
		stride=args.stride,
		conf=args.conf,
		iou=args.iou,
		max_frames=args.max_frames,
		max_persons=args.max_persons,
		min_mask_area_frac=args.min_mask_area_frac,
		center_bias=args.center_bias,
	) 