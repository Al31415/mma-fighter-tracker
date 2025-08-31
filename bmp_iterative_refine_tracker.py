import argparse
import os
import math
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Lazy deps
_ultra = None
_deepsort = None
_sam_predictor = None


def _lazy_imports():
	global _ultra, _deepsort, _sam_predictor
	if _ultra is None:
		from ultralytics import YOLO  # type: ignore
		_ultra = YOLO
	if _deepsort is None:
		from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
		_deepsort = DeepSort
	if _sam_predictor is None:
		try:
			from segment_anything import sam_model_registry, SamPredictor  # type: ignore
			_sam_predictor = (sam_model_registry, SamPredictor)
		except Exception:
			_sam_predictor = None


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


def _select_person_indices(result, h: int, w: int, max_persons: int, min_area_frac: float, center_bias: float) -> List[int]:
	idxs: List[int] = []
	if result.masks is None or result.boxes is None:
		return idxs
	masks = result.masks.data.cpu().numpy().astype(bool)
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


def _run_pose_on_masked(frame_bgr: np.ndarray, mask: np.ndarray, pose_model) -> Optional[np.ndarray]:
	# Soft mask the frame and run top-down pose; returns [17,3] keypoints or None
	if pose_model is None:
		return None
	H, W = frame_bgr.shape[:2]
	# Ensure mask matches frame size
	mask_rsz = cv2.resize((mask.astype(np.uint8) * 255), (W, H), interpolation=cv2.INTER_NEAREST) > 0
	m3 = np.repeat(mask_rsz[..., None], 3, axis=2)
	soft = (frame_bgr * (m3 > 0)).astype(np.uint8)
	res = pose_model.predict(source=soft, conf=0.3, verbose=False)
	if not res or res[0].keypoints is None or res[0].keypoints.xy is None:
		return None
	kps = res[0].keypoints.xy.cpu().numpy()  # [N,17,2]
	if kps.shape[0] == 0:
		return None
	# pick the one overlapping mask best
	best = None; best_count = -1
	for i in range(kps.shape[0]):
		pts = kps[i]
		cnt = 0
		for (x,y) in pts:
			xi = int(np.clip(x, 0, W-1)); yi = int(np.clip(y, 0, H-1))
			cnt += int(mask_rsz[yi, xi])
		if cnt > best_count:
			best_count = cnt; best = pts
	if best is None:
		return None
	vis = np.ones((best.shape[0],1), dtype=np.float32)
	return np.concatenate([best, vis], axis=1)  # [17,3]


def _refine_with_sam(frame_bgr: np.ndarray, init_mask: np.ndarray, keypoints_xy: Optional[np.ndarray], sam_predictor_obj) -> np.ndarray:
	# If SAM available, prompt with some visible keypoints; else return init_mask
	if sam_predictor_obj is None:
		return init_mask
	try:
		sam_model_registry, SamPredictor = sam_predictor_obj
		# Heuristic: choose SAM ViT-H default if env var SAM_MODEL is not set
		import os as _os
		ckpt = _os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
		if not _os.path.isfile(ckpt):
			return init_mask
		sam = sam_model_registry["vit_h"](checkpoint=ckpt)
		pred = SamPredictor(sam)
		pred.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
		points = None
		labels = None
		if keypoints_xy is not None:
			pts = keypoints_xy[:, :2]
			points = pts.astype(np.float32)
			labels = np.ones((points.shape[0],), dtype=np.int32)
		masks, scores, _ = pred.predict(point_coords=points, point_labels=labels, multimask_output=True)
		# pick best IoU with init_mask
		best_iou = -1.0; best_m = None
		for m in masks:
			m_bin = m.astype(bool)
			inter = np.logical_and(m_bin, init_mask).sum(dtype=np.float32)
			uni = np.logical_or(m_bin, init_mask).sum(dtype=np.float32)+1e-6
			iou = float(inter/uni)
			if iou > best_iou:
				best_iou = iou; best_m = m_bin
		return best_m if best_m is not None else init_mask
	except Exception:
		return init_mask


def run(video: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, center_bias: float, sam_enable: bool) -> None:
	_lazy_imports()
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}")

	seg_model = _ultra("yolov8x-seg.pt")
	pose_model = _ultra("yolov8x-pose.pt")
	sam_obj = _sam_predictor if sam_enable else None

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

	palette = {}
	def _color_for(tid: int) -> Tuple[int, int, int]:
		if tid not in palette:
			np.random.seed(tid * 7919)
			palette[tid] = tuple(int(x) for x in np.random.randint(0,255,size=3))
		return palette[tid]

	frame_idx = 0
	processed = 0
	while True:
		ok, frame = cap.read()
		if not ok:
			break
		if frame_idx % stride != 0:
			frame_idx += 1
			continue

		# Initial seg
		res0 = seg_model.predict(source=frame, conf=conf, iou=iou, verbose=False)
		r = res0[0]
		masks_all = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0,h,w),bool)
		# Resize masks to frame size if needed
		if masks_all.size > 0 and (masks_all.shape[1] != h or masks_all.shape[2] != w):
			masks_resized: List[np.ndarray] = []
			for m in masks_all:
				mr = cv2.resize((m.astype(np.uint8) * 255), (w, h), interpolation=cv2.INTER_NEAREST) > 0
				masks_resized.append(mr)
			masks_all = np.stack(masks_resized, axis=0) if masks_resized else np.zeros((0,h,w), bool)
		boxes_all = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
		classes = r.boxes.cls.cpu().numpy() if r.boxes is not None else np.zeros((0,))
		scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
		keep_idx = _select_person_indices(r, h, w, max_persons, min_mask_area_frac, center_bias)
		masks_sel = [masks_all[i] for i in keep_idx if int(classes[i])==0]

		# Two-pass refine
		for _pass in range(2):
			# Pose on masked, SAM refine
			refined_masks: List[np.ndarray] = []
			for m in masks_sel:
				kps = _run_pose_on_masked(frame, m, pose_model)
				m_ref = _refine_with_sam(frame, m, kps, sam_obj)
				refined_masks.append(m_ref)
			masks_sel = refined_masks
			# Re-detect with masked-out foreground to recover misses
			fg = np.zeros_like(frame)
			if masks_sel:
				fg_mask = np.zeros((h,w), dtype=bool)
				for m in masks_sel:
					fg_mask |= m
				masked_frame = frame.copy()
				masked_frame[fg_mask] = 0
				res_rec = seg_model.predict(source=masked_frame, conf=max(0.3, conf-0.1), iou=iou, verbose=False)
				rr = res_rec[0]
				if rr.masks is not None and rr.boxes is not None:
					m_new = rr.masks.data.cpu().numpy().astype(bool)
					# Resize recovered masks
					if m_new.size > 0 and (m_new.shape[1] != h or m_new.shape[2] != w):
						m_new_res: List[np.ndarray] = []
						for mn in m_new:
							mnr = cv2.resize((mn.astype(np.uint8) * 255), (w, h), interpolation=cv2.INTER_NEAREST) > 0
							m_new_res.append(mnr)
						m_new = np.stack(m_new_res, axis=0) if m_new_res else np.zeros((0,h,w), bool)
					cls_new = rr.boxes.cls.cpu().numpy()
					for i in range(m_new.shape[0]):
						if int(cls_new[i]) != 0:
							continue
						mn = m_new[i]
						# Skip if overlaps existing a lot
						over = False
						for m in masks_sel:
							inter = np.logical_and(m, mn).sum(dtype=np.float32)
							uni = np.logical_or(m, mn).sum(dtype=np.float32)+1e-6
							if float(inter/uni) > 0.3:
								over = True; break
						if not over:
							masks_sel.append(mn)
			# Early stop if nothing added/refined couldn't improve
			# (simple heuristic omitted for brevity)

		# Tracking with DeepSORT using bboxes
		bbs = []
		for m in masks_sel:
			box = _mask_to_box(m)
			ltwh = _xyxy_to_ltwh(box)
			bbs.append((list(ltwh), 0.9, 0))
		tracks = tracker.update_tracks(bbs, frame=frame)

		# Visualization + outputs
		label_map = np.zeros((h, w), dtype=np.uint16)
		for t in tracks:
			if not t.is_confirmed():
				continue
			tid = int(t.track_id)
			ltrb = t.to_ltrb(orig=True, orig_strict=True)
			if ltrb is None:
				ltrb = t.to_ltrb()
			x1,y1,x2,y2 = [int(v) for v in ltrb]
			# Find best mask by IoU for overlay
			best_iou = 0.0; best_m = None
			for m in masks_sel:
				mb = _mask_to_box(m)
				xA = max(ltrb[0], mb[0]); yA = max(ltrb[1], mb[1])
				xB = min(ltrb[2], mb[2]); yB = min(ltrb[3], mb[3])
				inter = max(0.0, xB-xA)*max(0.0, yB-yA)
				area_t = max(0.0,ltrb[2]-ltrb[0])*max(0.0,ltrb[3]-ltrb[1])
				area_m = max(0.0,mb[2]-mb[0])*max(0.0,mb[3]-mb[1])
				iou_tm = inter/(area_t+area_m-inter+1e-6)
				if iou_tm>best_iou:
					best_iou=iou_tm; best_m=m
			color = _color_for(tid)
			if best_m is not None and best_iou>=0.1:
				over = np.zeros_like(frame, dtype=np.uint8)
				over[best_m] = color
				cv2.addWeighted(over, 0.5, frame, 0.5, 0.0, frame)
				label_map[best_m] = tid
			else:
				label_map[max(0,y1):max(0,y2), max(0,x1):max(0,x2)] = tid
			cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
			cv2.putText(frame, f"ID {tid}", (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

		out.write(frame)
		cv2.imwrite(os.path.join(labels_dir, f"{frame_idx:06d}.png"), label_map)

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
	p = argparse.ArgumentParser(description="Iterative BMP refine (seg->pose->SAM->re-detect) + DeepSORT")
	p.add_argument("--video", required=True)
	p.add_argument("--outdir", default="tracker_out_bmp_iter")
	p.add_argument("--stride", type=int, default=5)
	p.add_argument("--conf", type=float, default=0.5)
	p.add_argument("--iou", type=float, default=0.45)
	p.add_argument("--max-frames", type=int, default=400)
	p.add_argument("--max-persons", type=int, default=2)
	p.add_argument("--min-mask-area-frac", type=float, default=0.005)
	p.add_argument("--center-bias", type=float, default=0.3)
	p.add_argument("--sam", action="store_true", help="Enable SAM refinement (requires SAM checkpoint path in SAM_CHECKPOINT)")
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
		sam_enable=args.sam,
	) 