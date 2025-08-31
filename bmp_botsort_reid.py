import argparse
import os
import math
from typing import List, Tuple
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

_ultra = None
_create_tracker = None
_clip_proc = None
_clip_model = None


def _lazy_imports() -> None:
	global _ultra, _create_tracker, _clip_proc, _clip_model
	if _ultra is None:
		from ultralytics import YOLO  # type: ignore
		_ultra = YOLO
	if _create_tracker is None:
		from boxmot.tracker_zoo import create_tracker  # type: ignore
		_create_tracker = create_tracker
	if (_clip_proc is None) or (_clip_model is None):
		try:
			from transformers import CLIPProcessor, CLIPModel  # type: ignore
			_clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
			_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
		except Exception:
			_clip_proc, _clip_model = None, None


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


def _find_botsort_cfg() -> str:
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
	return "botsort.yaml"


def run(video_path: str, outdir: str, stride: int, conf: float, iou: float, max_frames: int, max_persons: int, min_mask_area_frac: float, center_bias: float) -> None:
	_lazy_imports()
	os.makedirs(outdir, exist_ok=True)
	fps, n_frames, w, h = _read_video_meta(video_path)
	print(f"Video meta: {w}x{h} @ {fps:.2f}fps, frames={n_frames}")

	seg_model = _ultra("yolov8x-seg.pt")
	tracker_cfg = _find_botsort_cfg()
	reid_w = Path("osnet_x0_25_msmt17.pt")
	tracker = _create_tracker(tracker_type="botsort", tracker_config=tracker_cfg, reid_weights=reid_w, device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu", half=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if _clip_model is not None:
		_clip_model.to(device).eval()

	def embed_crops(crops_bgr: List[np.ndarray]) -> torch.Tensor:
		if _clip_proc is None or _clip_model is None or not crops_bgr:
			return torch.empty(0, 512)
		imgs_rgb = [c[:, :, ::-1].copy() for c in crops_bgr]
		inputs = _clip_proc(images=imgs_rgb, return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		with torch.no_grad():
			feat = _clip_model.get_image_features(**inputs)
		feat = F.normalize(feat, dim=1)
		return feat.detach().cpu()

	identity_names = ["FighterA", "FighterB"]
	identity_colors = {"FighterA": (0, 0, 255), "FighterB": (255, 0, 0)}
	id_protos: dict[str, torch.Tensor] = {}
	id_track_map: dict[int, str] = {}

	def crop_from_box(frame_bgr: np.ndarray, box_xyxy: np.ndarray, pad: float = 0.15) -> np.ndarray:
		x1, y1, x2, y2 = box_xyxy
		bw = x2 - x1; bh = y2 - y1
		px = bw * pad; py = bh * pad
		x1p = int(max(0, math.floor(x1 - px)))
		y1p = int(max(0, math.floor(y1 - py)))
		x2p = int(min(w, math.ceil(x2 + px)))
		y2p = int(min(h, math.ceil(y2 + py)))
		return frame_bgr[y1p:y2p, x1p:x2p]

	def assign_identity(emb: torch.Tensor) -> str:
		if not id_protos:
			return identity_names[0]
		best_name = identity_names[0]
		best_sim = -1.0
		for name, proto in id_protos.items():
			sim = float(F.cosine_similarity(emb.unsqueeze(0), proto.unsqueeze(0), dim=1).item())
			if sim > best_sim:
				best_sim = sim; best_name = name
		return best_name

	def update_proto(name: str, emb: torch.Tensor, momentum: float = 0.7) -> None:
		if name in id_protos:
			id_protos[name] = F.normalize(momentum * id_protos[name] + (1.0 - momentum) * emb, dim=0)
		else:
			id_protos[name] = emb.clone()

	prev_hist = None
	def is_scene_cut(frame_bgr: np.ndarray, thresh: float = 0.6) -> bool:
		nonlocal prev_hist
		hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
		hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
		hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
		cut = False
		if prev_hist is not None:
			corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
			cut = (corr < thresh)
		prev_hist = hist
		return cut

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

		sel = _select_person_indices(res, h, w, max_persons=max_persons, min_area_frac=min_mask_area_frac, center_bias=center_bias)

		boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
		scores = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)
		classes = res.boxes.cls.cpu().numpy() if res.boxes is not None else np.zeros((0,), dtype=np.float32)

		dets = []
		for i in sel:
			if int(classes[i]) != 0:
				continue
			dets.append([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], float(scores[i]), 0])
		dets_np = np.array(dets, dtype=np.float32) if dets else np.zeros((0, 6), dtype=np.float32)

		tracks = tracker.update(dets_np, frame)

		label_map = np.zeros((h, w), dtype=np.uint16)
		sel_boxes = []
		sel_masks = []
		if res.boxes is not None and res.masks is not None and len(sel) > 0:
			all_xyxy = res.boxes.xyxy.cpu().numpy()
			masks_np = res.masks.data.cpu().numpy().astype(bool)
			for i in sel:
				sel_boxes.append(all_xyxy[i])
				sel_masks.append(masks_np[i])

		cut_now = is_scene_cut(frame)

		for tr in tracks:
			x1, y1, x2, y2, tid = int(tr[0]), int(tr[1]), int(tr[2]), int(tr[3]), int(tr[4])
			box_arr = np.array([x1, y1, x2, y2], dtype=np.float32)
			best_iou = 0.0; best_mask = None
			for j, db in enumerate(sel_boxes):
				xA = max(box_arr[0], db[0]); yA = max(box_arr[1], db[1])
				xB = min(box_arr[2], db[2]); yB = min(box_arr[3], db[3])
				inter = max(0.0, xB - xA) * max(0.0, yB - yA)
				area_bx = max(0.0, box_arr[2] - box_arr[0]) * max(0.0, box_arr[3] - box_arr[1])
				area_db = max(0.0, db[2] - db[0]) * max(0.0, db[3] - db[1])
				iou_td = inter / (area_bx + area_db - inter + 1e-6)
				if iou_td > best_iou:
					best_iou = iou_td; best_mask = sel_masks[j]

			name = id_track_map.get(tid)
			if cut_now or (name is None):
				crop = crop_from_box(frame, box_arr, pad=0.15)
				embs = embed_crops([crop])
				if embs.shape[0] > 0:
					name = assign_identity(embs[0]) if id_protos else (identity_names[0] if len(id_protos) == 0 else identity_names[min(1, len(id_protos))])
					id_track_map[tid] = name
					update_proto(name, embs[0])
			color = identity_colors.get(name or identity_names[0], (0, 255, 0))

			if best_mask is not None and best_iou >= 0.1:
				mask_uint8 = (best_mask.astype(np.uint8) * 255)
				mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST) > 0
				over = np.zeros_like(frame, dtype=np.uint8)
				over[mask_resized] = color
				cv2.addWeighted(over, 0.5, frame, 0.5, 0.0, frame)
				label_map[mask_resized] = tid
			else:
				label_map[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = tid
				over = np.zeros_like(frame, dtype=np.uint8)
				over[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = color
				cv2.addWeighted(over, 0.35, frame, 0.65, 0.0, frame)

			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
			cv2.putText(frame, f"{name or 'Fighter'} | ID {tid}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
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
	p = argparse.ArgumentParser(description="YOLOv8-seg + BoT-SORT + CLIP ReID with scene-cut handling")
	p.add_argument("--video", required=True, help="Path to input video")
	p.add_argument("--outdir", default="tracker_out_botsort_reid", help="Directory to save outputs")
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