"""
Combined EDA for Facial Similarity Detection Datasets.

Datasets:
  1. subjects_0-1999_72_imgs  – 2000 synthetic subjects, 72 .png each
  2. casia_webface_extracted   – 10,572 real subjects, variable .jpg each
  3. olivetti                  – 40 subjects, 10 grayscale 64x64 each (.npy)

Usage:
  python scripts/eda_combined.py          # CLI
  or import from notebook:
    from scripts.eda_combined import run_full_eda
    results = run_full_eda()
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True)

SAMPLE_SUBJECTS = 100   # subjects sampled per dataset for image-property analysis
SAMPLE_IMGS = 5         # images sampled per subject for pixel-level analysis
RNG = np.random.default_rng(42)

# Matplotlib style
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})
sns.set_style("whitegrid")
PALETTE = sns.color_palette("Set2", 3)


# ── Data classes ───────────────────────────────────────────────────────────
class DatasetMeta(NamedTuple):
    name: str
    subject_ids: list[str]
    img_counts: dict[str, int]       # subject_id -> n_images
    total_images: int
    sample_paths: dict[str, list[Path]]  # subject_id -> sampled image paths


# ── 1. Scanning ────────────────────────────────────────────────────────────
def scan_folder_dataset(base: Path, name: str, ext: str = "*") -> DatasetMeta:
    """Scan a dataset with structure: base/<subject_id>/<image_files>."""
    subjects = sorted(
        [d.name for d in base.iterdir() if d.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    img_counts: dict[str, int] = {}
    sample_paths: dict[str, list[Path]] = {}

    sampled = (
        RNG.choice(subjects, size=min(SAMPLE_SUBJECTS, len(subjects)), replace=False)
        if len(subjects) > SAMPLE_SUBJECTS
        else subjects
    )
    sampled_set = set(sampled)

    for sid in tqdm(subjects, desc=f"Scanning {name}", leave=False):
        sdir = base / sid
        imgs = [f for f in sdir.iterdir() if f.is_file() and not f.name.startswith(".")]
        img_counts[sid] = len(imgs)
        if sid in sampled_set and imgs:
            chosen = RNG.choice(imgs, size=min(SAMPLE_IMGS, len(imgs)), replace=False)
            sample_paths[sid] = list(chosen)

    total = sum(img_counts.values())
    return DatasetMeta(name, subjects, img_counts, total, sample_paths)


def scan_olivetti(base: Path) -> DatasetMeta:
    """Load Olivetti .npy files and structure like other datasets."""
    faces = np.load(base / "olivetti_faces.npy")       # (400, 64, 64)
    targets = np.load(base / "olivetti_faces_target.npy")  # (400,)
    subjects = sorted(set(targets.astype(int)))
    img_counts = {}
    sample_paths: dict[str, list[Path]] = {}

    for sid in subjects:
        mask = targets == sid
        img_counts[str(sid)] = int(mask.sum())

    return DatasetMeta(
        "olivetti",
        [str(s) for s in subjects],
        img_counts,
        len(faces),
        sample_paths,
    )


def scan_all() -> dict[str, DatasetMeta]:
    """Scan all datasets, return dict keyed by dataset name."""
    datasets = {}

    synth_path = RAW / "subjects_0-1999_72_imgs"
    if synth_path.exists():
        datasets["synthetic"] = scan_folder_dataset(synth_path, "synthetic")

    casia_path = RAW / "casia_webface_extracted"
    if casia_path.exists():
        datasets["casia_webface"] = scan_folder_dataset(casia_path, "casia_webface")

    olivetti_path = RAW / "olivetti"
    if olivetti_path.exists():
        datasets["olivetti"] = scan_olivetti(olivetti_path)

    return datasets


# ── 2. Summary table ───────────────────────────────────────────────────────
def build_summary(datasets: dict[str, DatasetMeta]) -> pd.DataFrame:
    rows = []
    for key, ds in datasets.items():
        counts = list(ds.img_counts.values())
        rows.append({
            "Dataset": ds.name,
            "Subjects": len(ds.subject_ids),
            "Total Images": ds.total_images,
            "Min Imgs/Subject": min(counts),
            "Max Imgs/Subject": max(counts),
            "Mean Imgs/Subject": np.mean(counts),
            "Median Imgs/Subject": np.median(counts),
            "Std Imgs/Subject": np.std(counts),
        })
    return pd.DataFrame(rows)


# ── 3. Balance analysis ───────────────────────────────────────────────────
def plot_balance(datasets: dict[str, DatasetMeta]) -> plt.Figure:
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (key, ds), color in zip(axes, datasets.items(), PALETTE):
        counts = list(ds.img_counts.values())
        ax.hist(counts, bins=min(50, len(set(counts))), color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{ds.name}\n({len(ds.subject_ids)} subjects)")
        ax.set_xlabel("Images per Subject")
        ax.set_ylabel("Number of Subjects")
        ax.axvline(np.mean(counts), color="red", ls="--", lw=1.5, label=f"mean={np.mean(counts):.1f}")
        ax.legend()

    fig.suptitle("Dataset Balance: Images per Subject Distribution", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "balance_distribution.png")
    return fig


def plot_balance_boxplot(datasets: dict[str, DatasetMeta]) -> plt.Figure:
    records = []
    for key, ds in datasets.items():
        for sid, cnt in ds.img_counts.items():
            records.append({"Dataset": ds.name, "Images": cnt})
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Dataset", y="Images", hue="Dataset", palette="Set2", ax=ax, legend=False)
    ax.set_title("Images per Subject – Cross-Dataset Comparison")
    ax.set_ylabel("Images per Subject")
    fig.tight_layout()
    fig.savefig(OUT / "balance_boxplot.png")
    return fig


# ── 4. Image property analysis ─────────────────────────────────────────────
def analyze_image_properties(datasets: dict[str, DatasetMeta]) -> pd.DataFrame:
    """Sample images from each dataset, collect resolution/mode/brightness/contrast."""
    records = []

    for key, ds in datasets.items():
        if key == "olivetti":
            # Handle numpy-based dataset separately
            faces = np.load(RAW / "olivetti" / "olivetti_faces.npy")
            targets = np.load(RAW / "olivetti" / "olivetti_faces_target.npy")
            sampled_indices = RNG.choice(len(faces), size=min(50, len(faces)), replace=False)
            for idx in sampled_indices:
                img_data = (faces[idx] * 255).astype(np.uint8)
                records.append({
                    "dataset": ds.name,
                    "subject_id": str(int(targets[idx])),
                    "width": 64,
                    "height": 64,
                    "mode": "L",
                    "brightness": float(img_data.mean()),
                    "contrast": float(img_data.std()),
                    "aspect_ratio": 1.0,
                    "file_size_kb": 0,
                })
            continue

        for sid, paths in tqdm(ds.sample_paths.items(), desc=f"Props: {ds.name}", leave=False):
            for p in paths:
                try:
                    img = Image.open(p)
                    arr = np.array(img.convert("L"))
                    w, h = img.size
                    records.append({
                        "dataset": ds.name,
                        "subject_id": sid,
                        "width": w,
                        "height": h,
                        "mode": img.mode,
                        "brightness": float(arr.mean()),
                        "contrast": float(arr.std()),
                        "aspect_ratio": round(w / h, 3),
                        "file_size_kb": round(p.stat().st_size / 1024, 1),
                    })
                except Exception:
                    pass

    return pd.DataFrame(records)


def plot_image_properties(props_df: pd.DataFrame) -> list[plt.Figure]:
    figs = []

    # Resolution scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, grp) in enumerate(props_df.groupby("dataset")):
        ax.scatter(grp["width"], grp["height"], alpha=0.5, label=name,
                   color=PALETTE[i % len(PALETTE)], s=20)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "resolution_scatter.png")
    figs.append(fig)

    # Brightness + Contrast violin
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(axes, ["brightness", "contrast"],
                               ["Mean Brightness (0-255)", "Pixel Std Dev (Contrast)"]):
        sns.violinplot(data=props_df, x="dataset", y=col, hue="dataset", palette="Set2", ax=ax, inner="box", legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
    fig.suptitle("Brightness & Contrast by Dataset", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "brightness_contrast.png")
    figs.append(fig)

    # Color mode pie charts
    n_ds = props_df["dataset"].nunique()
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4))
    if n_ds == 1:
        axes = [axes]
    for ax, (name, grp) in zip(axes, props_df.groupby("dataset")):
        mode_counts = grp["mode"].value_counts()
        ax.pie(mode_counts, labels=mode_counts.index, autopct="%1.0f%%",
               colors=sns.color_palette("pastel"))
        ax.set_title(f"{name}\nColor Modes")
    fig.tight_layout()
    fig.savefig(OUT / "color_modes.png")
    figs.append(fig)

    return figs


# ── 5. Intra-subject variation ─────────────────────────────────────────────
def compute_intra_variation(datasets: dict[str, DatasetMeta], target_size=(64, 64)) -> pd.DataFrame:
    """Per-subject pixel std across images (measures face diversity within a subject)."""
    records = []

    for key, ds in datasets.items():
        if key == "olivetti":
            faces = np.load(RAW / "olivetti" / "olivetti_faces.npy")
            targets = np.load(RAW / "olivetti" / "olivetti_faces_target.npy")
            for sid in sorted(set(targets.astype(int))):
                subj_imgs = faces[targets == sid]
                mean_std = subj_imgs.std(axis=0).mean()
                max_std = subj_imgs.std(axis=0).max()
                records.append({
                    "dataset": ds.name,
                    "subject_id": str(sid),
                    "mean_pixel_std": float(mean_std * 255),
                    "max_pixel_std": float(max_std * 255),
                })
            continue

        sampled_sids = list(ds.sample_paths.keys())
        for sid in tqdm(sampled_sids, desc=f"Variation: {ds.name}", leave=False):
            paths = ds.sample_paths[sid]
            if len(paths) < 2:
                continue
            arrays = []
            for p in paths:
                try:
                    img = Image.open(p).convert("L").resize(target_size)
                    arrays.append(np.array(img, dtype=np.float32))
                except Exception:
                    pass
            if len(arrays) >= 2:
                stack = np.stack(arrays)
                pixel_std = stack.std(axis=0)
                records.append({
                    "dataset": ds.name,
                    "subject_id": sid,
                    "mean_pixel_std": float(pixel_std.mean()),
                    "max_pixel_std": float(pixel_std.max()),
                })

    return pd.DataFrame(records)


def plot_intra_variation(var_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.boxplot(data=var_df, x="dataset", y="mean_pixel_std", hue="dataset", palette="Set2", ax=axes[0], legend=False)
    axes[0].set_title("Mean Pixel Std Dev Across Subject Images")
    axes[0].set_ylabel("Mean Pixel Std")
    axes[0].set_xlabel("")

    sns.boxplot(data=var_df, x="dataset", y="max_pixel_std", hue="dataset", palette="Set2", ax=axes[1], legend=False)
    axes[1].set_title("Max Pixel Std Dev Across Subject Images")
    axes[1].set_ylabel("Max Pixel Std")
    axes[1].set_xlabel("")

    fig.suptitle("Intra-Subject Variation (higher = more diverse poses/lighting)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "intra_subject_variation.png")
    return fig


# ── 6. Sample grid ─────────────────────────────────────────────────────────
def plot_sample_grid(datasets: dict[str, DatasetMeta], n_subjects=4, n_imgs=6) -> plt.Figure:
    """Show sample faces from each dataset in a grid."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(n_ds * n_subjects, n_imgs, figsize=(n_imgs * 1.8, n_ds * n_subjects * 1.8))

    row = 0
    for key, ds in datasets.items():
        if key == "olivetti":
            faces = np.load(RAW / "olivetti" / "olivetti_faces.npy")
            targets = np.load(RAW / "olivetti" / "olivetti_faces_target.npy")
            chosen_sids = RNG.choice(sorted(set(targets.astype(int))), size=n_subjects, replace=False)
            for sid in chosen_sids:
                subj_imgs = faces[targets == sid]
                for col in range(n_imgs):
                    ax = axes[row, col] if n_ds * n_subjects > 1 else axes[col]
                    if col < len(subj_imgs):
                        ax.imshow(subj_imgs[col], cmap="gray")
                    ax.axis("off")
                    if col == 0:
                        ax.set_ylabel(f"oliv/{sid}", fontsize=8, rotation=0, labelpad=35)
                row += 1
            continue

        sampled_sids = list(ds.sample_paths.keys())
        chosen = RNG.choice(sampled_sids, size=min(n_subjects, len(sampled_sids)), replace=False)
        for sid in chosen:
            paths = ds.sample_paths[sid]
            for col in range(n_imgs):
                ax = axes[row, col] if n_ds * n_subjects > 1 else axes[col]
                if col < len(paths):
                    try:
                        img = Image.open(paths[col])
                        ax.imshow(img)
                    except Exception:
                        pass
                ax.axis("off")
                if col == 0:
                    short = ds.name[:5]
                    ax.set_ylabel(f"{short}/{sid}", fontsize=8, rotation=0, labelpad=40)
            row += 1

    fig.suptitle("Sample Faces per Dataset", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "sample_grid.png")
    return fig


# ── 7. Combined distribution ──────────────────────────────────────────────
def plot_combined_overview(summary_df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subjects per dataset
    axes[0].barh(summary_df["Dataset"], summary_df["Subjects"], color=PALETTE[:len(summary_df)])
    axes[0].set_xlabel("Number of Subjects")
    axes[0].set_title("Subjects per Dataset")
    for i, v in enumerate(summary_df["Subjects"]):
        axes[0].text(v + max(summary_df["Subjects"]) * 0.01, i, f"{v:,}", va="center", fontsize=10)

    # Total images per dataset
    axes[1].barh(summary_df["Dataset"], summary_df["Total Images"], color=PALETTE[:len(summary_df)])
    axes[1].set_xlabel("Total Images")
    axes[1].set_title("Total Images per Dataset")
    for i, v in enumerate(summary_df["Total Images"]):
        axes[1].text(v + max(summary_df["Total Images"]) * 0.01, i, f"{v:,}", va="center", fontsize=10)

    fig.suptitle("Combined Dataset Overview", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "combined_overview.png")
    return fig


# ── Main ───────────────────────────────────────────────────────────────────
def run_full_eda():
    """Run complete EDA pipeline. Returns dict of results for notebook use."""
    print("=" * 60)
    print("COMBINED FACIAL SIMILARITY DATASET EDA")
    print("=" * 60)

    # 1. Scan
    print("\n[1/6] Scanning datasets...")
    datasets = scan_all()
    print(f"  Found {len(datasets)} datasets: {list(datasets.keys())}")

    # 2. Summary
    print("\n[2/6] Building summary...")
    summary_df = build_summary(datasets)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT / "dataset_summary.csv", index=False)

    # 3. Balance
    print("\n[3/6] Plotting balance analysis...")
    fig_balance = plot_balance(datasets)
    fig_boxplot = plot_balance_boxplot(datasets)

    # 4. Overview
    fig_overview = plot_combined_overview(summary_df)

    # 5. Image properties
    print("\n[4/6] Analyzing image properties (sampled)...")
    props_df = analyze_image_properties(datasets)
    props_df.to_csv(OUT / "image_properties.csv", index=False)
    print(f"  Sampled {len(props_df)} images")
    print(props_df.groupby("dataset")[["width", "height", "brightness", "contrast"]].describe().round(1))
    fig_props = plot_image_properties(props_df)

    # 6. Intra-subject variation
    print("\n[5/6] Computing intra-subject variation (sampled)...")
    var_df = compute_intra_variation(datasets)
    var_df.to_csv(OUT / "intra_variation.csv", index=False)
    fig_var = plot_intra_variation(var_df)

    # 7. Sample grid
    print("\n[6/6] Generating sample grid...")
    fig_grid = plot_sample_grid(datasets)

    print("\n" + "=" * 60)
    print(f"Done. Outputs saved to {OUT}/")
    print("=" * 60)

    plt.close("all")

    return {
        "datasets": datasets,
        "summary": summary_df,
        "properties": props_df,
        "variation": var_df,
    }


if __name__ == "__main__":
    run_full_eda()
