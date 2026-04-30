"""
visualize.py — Traffic Vision Analytics Dashboard
Reads from a session directory produced by main.py and renders
4 analytical panels covering flow, speed, spatial, and feature space.

Usage:
    python visualize.py                          # auto-picks latest session
    python visualize.py --session 20260430_114103
    python visualize.py --session 20260430_114103 --save  # save PNGs instead of show
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--session", default=None, help="Session ID under ./analysis/")
parser.add_argument("--save",    action="store_true", help="Save PNGs instead of showing")
args = parser.parse_args()

ANALYSIS_DIR = Path("./analysis")

if args.session:
    session_dir = ANALYSIS_DIR / args.session
else:
    sessions = sorted(ANALYSIS_DIR.iterdir())
    if not sessions:
        sys.exit("No sessions found under ./analysis/")
    session_dir = sessions[-1]

print(f"Visualising session: {session_dir.name}\n")

# Load data
with open(session_dir / "summary.json") as f:
    summary = json.load(f)

crossings_df = pd.read_csv(session_dir / "crossings.csv")
speeds_df    = pd.read_csv(session_dir / "speeds.csv")
features_df  = pd.read_csv(session_dir / "features.csv")

# Derived
line_counts   = {int(k): v for k, v in summary["line_counts"].items()}
total_frames  = summary["total_frames"]
fps           = summary.get("fps_processed", 30)

# ─── Theme ────────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#58a6ff"
WARN    = "#f0883e"
SUCCESS = "#3fb950"
MUTED   = "#8b949e"
TEXT    = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    MUTED,
    "axes.labelcolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})

PALETTE_LINES = sns.color_palette("husl", len(line_counts))

SAVE_DIR = session_dir / "plots"
if args.save:
    SAVE_DIR.mkdir(exist_ok=True)

def save_or_show(name):
    if args.save:
        path = SAVE_DIR / f"{name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"  Saved → {path}")
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# Traffic Flow
print("Panel A: Traffic Flow")
fig = plt.figure(figsize=(16, 5), facecolor=BG)
fig.suptitle("Panel A — Traffic Flow Overview", color=TEXT, fontsize=13, y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# A1: Crossing counts bar chart
ax1 = fig.add_subplot(gs[0])
bars = ax1.bar(
    [str(k) for k in sorted(line_counts)],
    [line_counts[k] for k in sorted(line_counts)],
    color=PALETTE_LINES,
    edgecolor=BG,
    linewidth=0.8,
)
for bar, val in zip(bars, [line_counts[k] for k in sorted(line_counts)]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
             str(val), ha="center", va="bottom", color=TEXT, fontsize=8)
ax1.set_title("Total Crossings per Line")
ax1.set_xlabel("Line Index")
ax1.set_ylabel("Count")
ax1.set_ylim(0, max(line_counts.values()) * 1.2 + 1)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax1.grid(axis="y")

# A2: Cumulative crossing timeline
ax2 = fig.add_subplot(gs[1])
if not crossings_df.empty:
    for i, color in zip(sorted(line_counts), PALETTE_LINES):
        sub = crossings_df[crossings_df["line_index"] == i].sort_values("frame")
        if sub.empty:
            continue
        frames_sec = sub["frame"] / fps
        ax2.step(frames_sec, range(1, len(sub) + 1),
                 where="post", color=color, linewidth=1.8, label=f"Line {i}")
    ax2.set_title("Cumulative Crossings Over Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Cumulative count")
    ax2.legend(fontsize=7, framealpha=0.2)
    ax2.grid()
else:
    ax2.text(0.5, 0.5, "No crossing events", ha="center", va="center",
             transform=ax2.transAxes, color=MUTED)

# A3: Speed-at-crossing box plot per line
ax3 = fig.add_subplot(gs[2])
if not crossings_df.empty and "speed_kmh" in crossings_df.columns:
    valid = crossings_df[crossings_df["speed_kmh"] > 0]
    if not valid.empty:
        line_groups = [
            valid[valid["line_index"] == i]["speed_kmh"].values
            for i in sorted(line_counts)
        ]
        bplot = ax3.boxplot(
            [g for g in line_groups if len(g) > 0],
            patch_artist=True,
            medianprops=dict(color=WARN, linewidth=2),
            whiskerprops=dict(color=MUTED),
            capprops=dict(color=MUTED),
            flierprops=dict(marker="o", color=MUTED, markersize=3),
        )
        for patch, color in zip(bplot["boxes"], PALETTE_LINES):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        labels_used = [str(i) for i, g in zip(sorted(line_counts), line_groups) if len(g) > 0]
        ax3.set_xticks(range(1, len(labels_used) + 1))
        ax3.set_xticklabels(labels_used)
        ax3.set_title("Speed at Crossing per Line")
        ax3.set_xlabel("Line Index")
        ax3.set_ylabel("Speed (km/h)")
        ax3.grid(axis="y")
    else:
        ax3.text(0.5, 0.5, "No speed data at crossings", ha="center", va="center",
                 transform=ax3.transAxes, color=MUTED)

save_or_show("panel_A_flow")

# Speed Analysis
print("Panel B: Speed Analysis")
fig = plt.figure(figsize=(16, 5), facecolor=BG)
fig.suptitle("Panel B — Speed Analysis", color=TEXT, fontsize=13, y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# B1: Speed distribution KDE + histogram
ax1 = fig.add_subplot(gs[0])
valid_speeds = speeds_df[speeds_df["speed_kmh"] > 0]["speed_kmh"]
if not valid_speeds.empty:
    ax1.hist(valid_speeds, bins=30, color=ACCENT, alpha=0.4, density=True, edgecolor=BG)
    valid_speeds.plot.kde(ax=ax1, color=ACCENT, linewidth=2)
    ax1.axvline(valid_speeds.mean(),   color=WARN,    linestyle="--", linewidth=1.2, label=f"Mean {valid_speeds.mean():.1f}")
    ax1.axvline(valid_speeds.median(), color=SUCCESS,  linestyle="--", linewidth=1.2, label=f"Median {valid_speeds.median():.1f}")
    ax1.set_title("Speed Distribution")
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7, framealpha=0.2)
    ax1.grid()

# B2: Rolling average speed over frames
ax2 = fig.add_subplot(gs[1])
if not speeds_df.empty:
    avg_per_frame = (
        speeds_df[speeds_df["speed_kmh"] > 0]
        .groupby("frame")["speed_kmh"]
        .mean()
        .reset_index()
    )
    ax2.plot(avg_per_frame["frame"] / fps, avg_per_frame["speed_kmh"],
             color=MUTED, linewidth=0.8, alpha=0.4, label="Raw")
    window = max(5, len(avg_per_frame) // 20)
    smoothed = avg_per_frame["speed_kmh"].rolling(window, center=True, min_periods=1).mean()
    ax2.plot(avg_per_frame["frame"] / fps, smoothed,
             color=ACCENT, linewidth=2, label=f"Rolling avg (w={window})")
    ax2.set_title("Average Speed Over Time")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Avg speed (km/h)")
    ax2.legend(fontsize=7, framealpha=0.2)
    ax2.grid()

# B3: Per-track speed heatmap (top 20 longest tracks)
ax3 = fig.add_subplot(gs[2])
if not speeds_df.empty:
    track_lengths = speeds_df.groupby("track_id")["frame"].count()
    top_tracks    = track_lengths.nlargest(20).index
    sub = speeds_df[speeds_df["track_id"].isin(top_tracks) & (speeds_df["speed_kmh"] > 0)]

    # Pivot: rows = track, cols = frame bucket
    BUCKETS = 30
    sub = sub.copy()
    sub["frame_bucket"] = pd.cut(sub["frame"], bins=BUCKETS, labels=False)
    pivot = sub.pivot_table(index="track_id", columns="frame_bucket",
                            values="speed_kmh", aggfunc="mean")

    cmap = LinearSegmentedColormap.from_list("speed", ["#0d1117", "#58a6ff", "#f0883e", "#ff3333"])
    sns.heatmap(pivot, ax=ax3, cmap=cmap, cbar_kws={"label": "km/h"},
                linewidths=0, xticklabels=False)
    ax3.set_title("Per-Track Speed Heatmap")
    ax3.set_xlabel("Time →")
    ax3.set_ylabel("Track ID")
    ax3.tick_params(axis="y", labelsize=7)

save_or_show("panel_B_speed")

# PANEL C
print("Panel C: Spatial Distribution")
fig = plt.figure(figsize=(14, 6), facecolor=BG)
fig.suptitle("Panel C — Spatial Distribution", color=TEXT, fontsize=13, y=1.02)
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)

if not crossings_df.empty and {"cx", "cy"}.issubset(crossings_df.columns):
    # C1: Scatter colored by speed
    ax1 = fig.add_subplot(gs[0])
    sc = ax1.scatter(
        crossings_df["cx"], crossings_df["cy"],
        c=crossings_df["speed_kmh"].clip(0, crossings_df["speed_kmh"].quantile(0.95)),
        cmap="plasma", s=18, alpha=0.7, linewidths=0,
    )
    ax1.invert_yaxis()  # image coords: y=0 at top
    cbar = plt.colorbar(sc, ax=ax1)
    cbar.set_label("Speed (km/h)", color=TEXT)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    ax1.set_title("Crossing Positions — Speed")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.grid()

    # C2: KDE density (traffic hotspots)
    ax2 = fig.add_subplot(gs[1])
    try:
        sns.kdeplot(
            data=crossings_df, x="cx", y="cy",
            fill=True, cmap="mako", thresh=0.02, levels=12, ax=ax2,
        )
    except Exception:
        ax2.scatter(crossings_df["cx"], crossings_df["cy"],
                    c=ACCENT, s=10, alpha=0.5)
    ax2.invert_yaxis()
    ax2.set_title("Traffic Density (KDE)")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.grid()
else:
    for i in range(2):
        ax = fig.add_subplot(gs[i])
        ax.text(0.5, 0.5, "No crossing position data", ha="center", va="center",
                transform=ax.transAxes, color=MUTED)

save_or_show("panel_C_spatial")

# Feature Space
print("Panel D: Feature Space")
fig = plt.figure(figsize=(16, 5), facecolor=BG)
fig.suptitle("Panel D — Feature Space Analysis", color=TEXT, fontsize=13, y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

GLCM_COLS = ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
SIFT_COLS = [c for c in features_df.columns if c.startswith("sift_")]

# D1: GLCM correlation matrix
ax1 = fig.add_subplot(gs[0])
if not features_df.empty and all(c in features_df.columns for c in GLCM_COLS):
    corr = features_df[GLCM_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    cmap_div = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr, ax=ax1, annot=True, fmt=".2f",
        cmap=cmap_div, center=0, vmin=-1, vmax=1,
        linewidths=1, linecolor=BG,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9},
    )
    labels = ["Contrast", "Homogeneity", "Energy", "Correlation"]
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_yticklabels(labels, rotation=0)
    ax1.set_title("GLCM Feature Correlations")

# D2: PCA on full SIFT+GLCM feature vector
ax2 = fig.add_subplot(gs[1])
feature_cols = SIFT_COLS + GLCM_COLS
if not features_df.empty and len(feature_cols) > 2:
    feat_matrix = features_df[feature_cols].dropna()
    if len(feat_matrix) > 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feat_matrix)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        track_ids = features_df.loc[feat_matrix.index, "track_id"].values
        unique_tracks = np.unique(track_ids)
        palette = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_tracks))))
        color_map = {t: palette[i % 20] for i, t in enumerate(unique_tracks)}
        colors = [color_map[t] for t in track_ids]

        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=12, alpha=0.6, linewidths=0)
        ax2.set_title(
            f"PCA — SIFT+GLCM Feature Space\n"
            f"PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
            f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%"
        )
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        ax2.grid()

        # Scree inset
        inset = ax2.inset_axes([0.68, 0.68, 0.30, 0.28])
        pca_full = PCA().fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
        inset.plot(range(1, min(11, len(cumvar)+1)), cumvar[:10],
                   color=ACCENT, linewidth=1.5, marker="o", markersize=3)
        inset.axhline(90, color=WARN, linestyle="--", linewidth=0.8)
        inset.set_title("Var %", fontsize=6, color=TEXT)
        inset.tick_params(labelsize=5)
        inset.set_facecolor(BG)
    else:
        ax2.text(0.5, 0.5, "Insufficient feature data for PCA",
                 ha="center", va="center", transform=ax2.transAxes, color=MUTED)

# D3: Per-track GLCM contrast over time (top 5 longest tracks)
ax3 = fig.add_subplot(gs[2])
if not features_df.empty and "glcm_contrast" in features_df.columns:
    track_obs = features_df.groupby("track_id")["frame"].count()
    top5 = track_obs.nlargest(5).index
    palette5 = sns.color_palette("husl", 5)
    for tid, color in zip(top5, palette5):
        sub = features_df[features_df["track_id"] == tid].sort_values("frame")
        ax3.plot(sub["frame"] / fps, sub["glcm_contrast"],
                 color=color, linewidth=1.5, alpha=0.8, label=f"ID {tid}")
    ax3.set_title("GLCM Contrast Over Time\n(top 5 tracks by duration)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("Contrast")
    ax3.legend(fontsize=7, framealpha=0.2)
    ax3.grid()

save_or_show("panel_D_features")

# Summary printout
print("\n Visualization complete")
print(f"  Session       : {session_dir.name}")
print(f"  Total frames  : {total_frames}")
print(f"  Crossings     : {len(crossings_df)}")
print(f"  Speed records : {len(speeds_df)}")
print(f"  Feature rows  : {len(features_df)}")
if args.save:
    print(f"  Plots saved → : {SAVE_DIR}")