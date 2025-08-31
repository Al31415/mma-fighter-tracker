import argparse
import csv
import os
from typing import List, Dict, Any, Tuple

import plotly.graph_objects as go


def load_csv(csv_path: str) -> Tuple[List[float], List[float], List[int], List[str], List[int], List[int]]:
	xs: List[float] = []
	ys: List[float] = []
	clusters: List[int] = []
	labels: List[str] = []
	starts: List[int] = []
	ends: List[int] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			labels.append(row["label"])  # includes filename + chunk
			starts.append(int(row["start_sec"]))
			ends.append(int(row["end_sec"]))
			clusters.append(int(row["cluster"]))
			xs.append(float(row["x"]))
			ys.append(float(row["y"]))
	return xs, ys, clusters, labels, starts, ends


def build_figure(xs: List[float], ys: List[float], clusters: List[int], labels: List[str], starts: List[int], ends: List[int], title: str) -> go.Figure:
	# Group points by cluster for discrete coloring and legend
	cluster_to_points: Dict[int, List[int]] = {}
	for i, c in enumerate(clusters):
		cluster_to_points.setdefault(c, []).append(i)

	fig = go.Figure()
	palette = [
		"#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
		"#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
	]

	for idx, (c, inds) in enumerate(sorted(cluster_to_points.items(), key=lambda kv: kv[0])):
		color = palette[idx % len(palette)]
		trace = go.Scattergl(
			x=[xs[i] for i in inds],
			y=[ys[i] for i in inds],
			mode="markers",
			marker=dict(size=10, color=color, line=dict(width=0.5, color="#FFFFFF")),
			name=f"Cluster {c} (n={len(inds)})",
			customdata=[
				[labels[i], starts[i], ends[i], clusters[i]]
				for i in inds
			],
			hovertemplate=(
				"%{customdata[0]}<br>"
				"Time: %{customdata[1]}s–%{customdata[2]}s<br>"
				"Cluster: %{customdata[3]}<extra></extra>"
			),
		)
		fig.add_trace(trace)

	# Make it large and readable
	fig.update_layout(
		title=title,
		width=1600,
		height=1200,
		xaxis_title="PC1",
		yaxis_title="PC2",
					hovermode="closest",
		legend=dict(
			yanchor="top", y=0.99,
			xanchor="left", x=0.01,
			bgcolor="rgba(255,255,255,0.8)", bordercolor="#333", borderwidth=1,
		),
		margin=dict(l=60, r=20, t=60, b=60),
		template="plotly_white",
		dragmode="pan",
	)
	# Padding around points for better spacing
	if xs and ys:
		xmin, xmax = min(xs), max(xs)
		ymin, ymax = min(ys), max(ys)
		dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
		dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
		fig.update_xaxes(range=[xmin - dx, xmax + dx])
		fig.update_yaxes(range=[ymin - dy, ymax + dy])

	return fig


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Create an interactive Plotly viewer for video embeddings with hover tooltips.")
	p.add_argument("--csv", required=True, help="Path to embeddings_2d.csv")
	p.add_argument("--out", default="embeddings_2d_interactive.html", help="Output HTML file")
	p.add_argument("--title", default="Video chunk embeddings (PCA2) — interactive", help="Plot title")
	return p.parse_args()


def main() -> int:
	args = parse_args()
	xs, ys, clusters, labels, starts, ends = load_csv(args.csv)
	fig = build_figure(xs, ys, clusters, labels, starts, ends, args.title)
	fig.write_html(args.out, include_plotlyjs="cdn", full_html=True)
	print(f"Saved interactive HTML: {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main()) 