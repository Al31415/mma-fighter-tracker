import argparse
import os
import math
from typing import List, Tuple, Optional

import numpy as np

# Optional backends
try:
	from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor
	_HAS_HF = True
except Exception:
	_HAS_HF = False

import torch
import torch.nn as nn

import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------------
# Video utils
# ------------------------------

def _read_video_meta(video_path: str) -> Tuple[float, int]:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	cap.release()
	return float(fps), n_frames


def _sample_frames_uniform(video_path: str, start_s: float, end_s: float, num_frames: int) -> List[np.ndarray]:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	start_f = max(0, int(start_s * fps))
	end_f = max(start_f + 1, int(end_s * fps))
	idxs = np.linspace(start_f, end_f - 1, num=num_frames, dtype=np.int64)
	frames: List[np.ndarray] = []
	for idx in idxs:
		cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
		ok, frame_bgr = cap.read()
		if not ok:
			break
		frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
		frames.append(frame_rgb)
	cap.release()
	return frames


def _save_clip(video_path: str, start_s: float, end_s: float, dst_path: str) -> None:
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	start_f = max(0, int(start_s * fps))
	end_f = max(start_f + 1, int(end_s * fps))
	# Try mp4v; if open fails at first write, we fallback to XVID avi
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
	if not out.isOpened():
		# Fallback to avi
		avi_path = os.path.splitext(dst_path)[0] + ".avi"
		fourcc = cv2.VideoWriter_fourcc(*"XVID")
		out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
		dst_path = avi_path
	cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
	cur = start_f
	while cur < end_f:
		ok, frame_bgr = cap.read()
		if not ok:
			break
		out.write(frame_bgr)
		cur += 1
	out.release()
	cap.release()


def _sanitize_label(label: str) -> str:
	# Remove or replace characters illegal on Windows filenames
	illegal = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
	res = label
	for ch in illegal:
		res = res.replace(ch, '-')
	# Collapse multiple spaces/dashes
	res = ' '.join(res.split())
	return res


def _build_label(video_path: str, idx: int, s: int, e: int) -> str:
	root = os.path.splitext(os.path.basename(video_path))[0]
	return f"{root} - chunk_{idx} [{s}-{e}]s"


# ------------------------------
# Model backends
# ------------------------------
class VideoEmbedder(nn.Module):
	def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
		raise NotImplementedError


class CLIPFrameEmbedder(VideoEmbedder):
	def __init__(self, model_name: str = "openai/clip-vit-large-patch14", batch_size: int = 8):
		super().__init__()
		self.processor = CLIPProcessor.from_pretrained(model_name)
		self.model = CLIPModel.from_pretrained(model_name)
		self.model.eval()
		self.batch_size = batch_size
		for p in self.model.parameters():
			p.requires_grad = False

	@torch.no_grad()
	def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
		device = next(self.model.parameters()).device
		embs: List[torch.Tensor] = []
		# Process in batches to control memory
		for i in range(0, len(frames), self.batch_size):
			batch = frames[i:i + self.batch_size]
			inputs = self.processor(images=batch, return_tensors="pt")
			inputs = {k: v.to(device) for k, v in inputs.items()}
			feat = self.model.get_image_features(**inputs)  # [B, D]
			# Normalize per-frame
			feat = torch.nn.functional.normalize(feat, dim=1)
			embs.append(feat.cpu())
		if not embs:
			raise RuntimeError("CLIP produced no embeddings")
		all_embs = torch.cat(embs, dim=0)  # [T, D]
		clip_emb = all_embs.mean(dim=0)  # [D]
		clip_emb = torch.nn.functional.normalize(clip_emb, dim=0)
		return clip_emb


class HFVideoMAEEmbedder(VideoEmbedder):
	def __init__(self, model_name: str = "MCG-NJU/videomae-base", target_frames: int = 16):
		super().__init__()
		self.processor = AutoProcessor.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.model.eval()
		self.target_frames = target_frames
		for p in self.model.parameters():
			p.requires_grad = False

	def _resample(self, frames: List[np.ndarray]) -> List[np.ndarray]:
		if len(frames) == self.target_frames:
			return frames
		idxs = np.linspace(0, max(0, len(frames) - 1), num=self.target_frames, dtype=np.int64)
		return [frames[i] for i in idxs]

	@torch.no_grad()
	def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
		device = next(self.model.parameters()).device
		frames = self._resample(frames)
		inputs = self.processor(videos=[frames], return_tensors="pt")
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = self.model(**inputs)
		last = outputs.last_hidden_state  # [B=1, N, D]
		emb = last.mean(dim=1).squeeze(0).cpu()  # [D]
		return emb


class HFVideoSwinEmbedder(VideoEmbedder):
	def __init__(self, model_name: str = "microsoft/videoswin-base-patch244-kinetics400"):
		super().__init__()
		self.processor = AutoProcessor.from_pretrained(model_name)
		self.model = AutoModel.from_pretrained(model_name)
		self.model.eval()
		for p in self.model.parameters():
			p.requires_grad = False

	@torch.no_grad()
	def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
		# Processor expects a list of PIL.Images or numpy arrays
		inputs = self.processor(videos=[frames], return_tensors="pt")
		# Move to model device
		device = next(self.model.parameters()).device
		inputs = {k: v.to(device) for k, v in inputs.items()}
		outputs = self.model(**inputs)
		# Try to use pooled output; if not present, average last_hidden_state
		if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
			emb = outputs.pooler_output  # [1, D]
		else:
			last = outputs.last_hidden_state  # [1, T, D] or [1, N, D]
			emb = last.mean(dim=1)
		return emb.squeeze(0).cpu()


class TorchvisionMViTEmbedder(VideoEmbedder):
	def __init__(self):
		super().__init__()
		from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
		weights = MViT_V2_S_Weights.DEFAULT
		model = mvit_v2_s(weights=weights)
		# Replace classifier head with identity to get penultimate features
		if hasattr(model, "classifier"):
			model.classifier = nn.Identity()
		elif hasattr(model, "heads"):
			model.heads = nn.Identity()
		self.model = model.eval()
		for p in self.model.parameters():
			p.requires_grad = False
		self.num_frames = 16

	def _preprocess(self, frames: List[np.ndarray], size: int = 224) -> torch.Tensor:
		# Convert list of RGB frames (H,W,C) uint8 to tensor [1,3,T,H,W]
		processed = []
		for f in frames:
			resized = cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
			t = torch.from_numpy(resized).float() / 255.0  # [H,W,C]
			t = t.permute(2, 0, 1)  # [C,H,W]
			processed.append(t)
		vid = torch.stack(processed, dim=1)  # [C,T,H,W]
		mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
		std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
		vid = (vid - mean) / std
		return vid.unsqueeze(0)  # [1,3,T,H,W]

	def _resample_frames(self, frames: List[np.ndarray], target_T: int) -> List[np.ndarray]:
		if len(frames) == target_T:
			return frames
		if len(frames) <= 1:
			# Duplicate the single frame
			return frames * target_T
		idxs = np.linspace(0, len(frames) - 1, num=target_T, dtype=np.int64)
		return [frames[i] for i in idxs]

	@torch.no_grad()
	def forward(self, frames: List[np.ndarray]) -> torch.Tensor:
		# Ensure fixed T for MViT
		frames = self._resample_frames(frames, self.num_frames)
		inp = self._preprocess(frames)
		device = next(self.model.parameters()).device
		inp = inp.to(device)
		feat = self.model(inp)  # [1, D]
		return feat.squeeze(0).cpu()


def get_embedder(backend: str = "clip", prefer_hf: bool = True) -> VideoEmbedder:
	backend = backend.lower()
	if backend == "clip":
		return CLIPFrameEmbedder()
	if backend == "videomae" and _HAS_HF:
		try:
			return HFVideoMAEEmbedder()
		except Exception:
			pass
	if backend == "videoswin" and _HAS_HF:
		try:
			return HFVideoSwinEmbedder()
		except Exception:
			pass
	if backend == "mvit":
		return TorchvisionMViTEmbedder()
	# Fallback preference
	if _HAS_HF:
		try:
			return HFVideoMAEEmbedder()
		except Exception:
			try:
				return HFVideoSwinEmbedder()
			except Exception:
				return TorchvisionMViTEmbedder()
	return TorchvisionMViTEmbedder()


# ------------------------------
# Pipeline
# ------------------------------

def compute_chunks(duration_s: float, chunk_s: int) -> List[Tuple[float, float]]:
	n_chunks = int(math.ceil(duration_s / chunk_s))
	chunks: List[Tuple[float, float]] = []
	for i in range(n_chunks):
		start = i * chunk_s
		end = min((i + 1) * chunk_s, duration_s)
		if end - start > 0.5:
			chunks.append((start, end))
	return chunks


def run(video_path: str, outdir: str, chunk_seconds: int, frames_per_chunk: int, k_clusters: int, prefer_hf: bool, save_clips: bool, clips_dir: Optional[str], backend: str) -> None:
	os.makedirs(outdir, exist_ok=True)
	if save_clips:
		clips_dir = clips_dir or os.path.join(outdir, "clips")
		os.makedirs(clips_dir, exist_ok=True)
	fps, n_frames = _read_video_meta(video_path)
	duration_s = n_frames / max(fps, 1e-6)
	chunks = compute_chunks(duration_s, chunk_seconds)
	print(f"Video meta: fps={fps:.2f}, frames={n_frames}, duration={duration_s:.2f}s, chunks={len(chunks)}")

	embedder = get_embedder(backend=backend, prefer_hf=prefer_hf)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	embedder.to(device)

	embeddings: List[np.ndarray] = []
	labels: List[str] = []
	for idx, (s, e) in enumerate(chunks):
		frames = _sample_frames_uniform(video_path, s, e, frames_per_chunk)
		if len(frames) < max(2, frames_per_chunk // 2):
			continue
		emb = embedder(frames)
		embeddings.append(emb.numpy())
		label = _build_label(video_path, idx, int(s), int(e))
		label = _sanitize_label(label)
		labels.append(label)
		if save_clips:
			dst_path = os.path.join(clips_dir or outdir, f"{label}.mp4")
			try:
				_save_clip(video_path, s, e, dst_path)
			except Exception as ce:
				print(f"Clip save failed for {label}: {ce}")
		if (idx + 1) % 10 == 0:
			print(f"Processed {idx + 1}/{len(chunks)} chunks")

	if not embeddings:
		raise RuntimeError("No embeddings extracted.")

	X = np.stack(embeddings, axis=0)  # [N, D]

	# Cluster
	kmeans = KMeans(n_clusters=min(k_clusters, len(X)), random_state=42, n_init="auto")
	cluster_ids = kmeans.fit_predict(X)

	# 2D projection via PCA
	pca = PCA(n_components=2, random_state=42)
	XY = pca.fit_transform(X)

	# Save CSV
	csv_path = os.path.join(outdir, "embeddings_2d.csv")
	with open(csv_path, "w", encoding="utf-8") as f:
		f.write("label,start_sec,end_sec,cluster,x,y\n")
		for (lab, (s, e), c, (x, y)) in zip(labels, chunks[:len(labels)], cluster_ids, XY):
			f.write(f"{lab},{int(s)},{int(e)},{c},{x:.6f},{y:.6f}\n")
	print(f"Saved CSV: {csv_path}")

	# Plot 1: Large clustered scatter with centroids
	plt.figure(figsize=(20, 14))
	sc = plt.scatter(XY[:, 0], XY[:, 1], c=cluster_ids, cmap="tab10", s=120, alpha=0.9, edgecolors="white", linewidths=0.5)
	unique_clusters = sorted(set(cluster_ids.tolist()))
	centroids = {}
	for c in unique_clusters:
		mask = cluster_ids == c
		centroid = XY[mask].mean(axis=0)
		centroids[c] = centroid
		plt.scatter([centroid[0]], [centroid[1]], c="black", s=350, marker="x", linewidths=2)
		plt.annotate(f"Cluster {c} (n={mask.sum()})", (centroid[0], centroid[1]), fontsize=16, fontweight="bold",
				ha="center", va="bottom", color="black",
				bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.7))
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.title("Video chunk embeddings (PCA2) — clusters", fontsize=22)
	plt.xlabel("PC1", fontsize=18)
	plt.ylabel("PC2", fontsize=18)
	plt.tight_layout()
	png_path1 = os.path.join(outdir, "embeddings_2d_clusters.png")
	plt.savefig(png_path1, dpi=200)
	print(f"Saved plot: {png_path1}")

	# Plot 2: Labeled points with filename, large canvas and offset labels
	plt.figure(figsize=(40, 28))
	plt.scatter(XY[:, 0], XY[:, 1], c=cluster_ids, cmap="tab10", s=130, alpha=0.85, edgecolors="white", linewidths=0.5)
	# Compute per-cluster centroids if not already
	xrange = max(1e-6, float(XY[:, 0].max() - XY[:, 0].min()))
	yrange = max(1e-6, float(XY[:, 1].max() - XY[:, 1].min()))
	for i, lab in enumerate(labels):
		c = int(cluster_ids[i])
		cent = centroids.get(c, XY.mean(axis=0))
		v = XY[i] - cent
		norm = float(np.linalg.norm(v))
		if norm < 1e-6:
			v = np.array([1.0, 0.0], dtype=np.float32)
			norm = 1.0
		off = (v / norm) * np.array([0.04 * xrange, 0.04 * xrange])
		xy_text = XY[i] + off
		plt.annotate(
			lab,
			xy=(XY[i, 0], XY[i, 1]),
			xytext=(xy_text[0], xy_text[1]),
			fontsize=12,
			color="black",
			bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
			arrowprops=dict(arrowstyle="-", color="gray", lw=0.8, alpha=0.8),
		)
	plt.grid(True, linestyle="--", alpha=0.35)
	plt.title("Video chunk embeddings (PCA2) — labeled with filenames", fontsize=26)
	plt.xlabel("PC1", fontsize=20)
	plt.ylabel("PC2", fontsize=20)
	plt.tight_layout()
	png_path2 = os.path.join(outdir, "embeddings_2d_labeled_filenames.png")
	plt.savefig(png_path2, dpi=220)
	print(f"Saved plot: {png_path2}")

	# Optional: interactive HTML plot
	if os.environ.get("GENERATE_INTERACTIVE_HTML", "0") == "1":
		# Back-compat path via env var (not used by CLI)
		pass
	
	# Done in run(); HTML plotting handled by caller via flag


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Chunk a video, embed with a Swin-like video model, cluster, and project to 2D.")
	p.add_argument("--video", required=True, help="Path to input video")
	p.add_argument("--outdir", default="video_embed_out", help="Output directory")
	p.add_argument("--chunk-seconds", type=int, default=30, help="Chunk length in seconds (default: 30)")
	p.add_argument("--frames-per-chunk", type=int, default=32, help="Frames sampled per chunk (default: 32)")
	p.add_argument("--k", type=int, default=8, help="Number of KMeans clusters (default: 8)")
	p.add_argument("--prefer-hf", action="store_true", help="Prefer Hugging Face VideoSwin if available")
	p.add_argument("--save-clips", action="store_true", help="Save each chunk as a video clip with a filename matching the HTML label")
	p.add_argument("--clips-dir", default="", help="Directory to save clips (default: <outdir>/clips)")
	p.add_argument("--backend", default="clip", choices=["clip", "videomae", "videoswin", "mvit"], help="Embedding backend to use (default: clip)")
	p.add_argument("--interactive-html", action="store_true", help="Also write an interactive Plotly HTML scatter with hover tooltips")
	return p.parse_args()


if __name__ == "__main__":
	args = parse_args()
	# Run pipeline
	run(
		video_path=args.video,
		outdir=args.outdir,
		chunk_seconds=args.chunk_seconds,
		frames_per_chunk=args.frames_per_chunk,
		k_clusters=args.k,
		prefer_hf=args.prefer_hf,
		save_clips=args.save_clips,
		clips_dir=args.clips_dir or None,
		backend=args.backend,
	)

	# Write interactive HTML if requested (using CSV just saved)
	if args.interactive_html:
		try:
			import plotly.graph_objects as go
		except Exception:
			print("Plotly not installed; skipping interactive HTML. Install with: pip install plotly")
		else:
			# Load CSV we just wrote
			csv_path = os.path.join(args.outdir, "embeddings_2d.csv")
			xs: list = []
			ys: list = []
			clusters: list = []
			labels: list = []
			starts: list = []
			ends: list = []
			with open(csv_path, "r", encoding="utf-8") as f:
				import csv as _csv
				reader = _csv.DictReader(f)
				for row in reader:
					labels.append(row["label"])
					starts.append(int(row["start_sec"]))
					ends.append(int(row["end_sec"]))
					clusters.append(int(row["cluster"]))
					xs.append(float(row["x"]))
					ys.append(float(row["y"]))

			# Build figure
			cluster_to_points = {}
			for i, c in enumerate(clusters):
				cluster_to_points.setdefault(c, []).append(i)
			fig = go.Figure()
			palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
			for idx, (c, inds) in enumerate(sorted(cluster_to_points.items(), key=lambda kv: kv[0])):
				color = palette[idx % len(palette)]
				fig.add_trace(go.Scattergl(
					x=[xs[i] for i in inds],
					y=[ys[i] for i in inds],
					mode="markers",
					marker=dict(size=10, color=color, line=dict(width=0.5, color="#FFFFFF")),
					name=f"Cluster {c} (n={len(inds)})",
					customdata=[[labels[i], starts[i], ends[i], clusters[i]] for i in inds],
					hovertemplate=("%{customdata[0]}<br>Time: %{customdata[1]}s–%{customdata[2]}s<br>Cluster: %{customdata[3]}<extra></extra>"),
				))
			# Layout
			fig.update_layout(
				title="Video chunk embeddings (PCA2) — interactive",
				width=1600, height=1200,
				xaxis_title="PC1", yaxis_title="PC2",
				hovermode="closest",
				legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)", bordercolor="#333", borderwidth=1),
				margin=dict(l=60, r=20, t=60, b=60), template="plotly_white", dragmode="pan",
			)
			# Save
			html_out = os.path.join(args.outdir, "embeddings_2d_interactive.html")
			fig.write_html(html_out, include_plotlyjs="cdn", full_html=True)
			print(f"Saved interactive HTML: {html_out}") 