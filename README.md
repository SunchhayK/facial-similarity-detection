# Facial Similarity Detection

A deep learning system for detecting similar faces across images. Uses convolutional neural networks to extract facial embeddings and compares them via cosine similarity. Built as part of the CADT Masters Fellowship program.

## Project Goals

1. **Train a facial similarity model** — learn identity-preserving embeddings from a synthetic face dataset
2. **Benchmark existing models** — evaluate state-of-the-art face recognition models (ArcFace, CosFace, AdaFace, etc.) against our dataset

## Repository Structure

```
facial-similarity-detection/
├── dataset/
│   └── raw/
│       └── subjects_0-1999_72_imgs/   # 2,000 subjects × 72 images each
│           ├── 0/                      # Subject folder
│           │   ├── 0.png               # 112×112 RGBA face image
│           │   ├── 1.png
│           │   └── ...                 # 72 images per subject
│           ├── 1/
│           └── ...
├── eda.ipynb                # Comprehensive exploratory data analysis
├── eda_outputs/             # Generated plots from EDA
├── main.py                  # Application entry point
├── pyproject.toml           # Project config & dependencies
├── uv.lock                  # Locked dependency versions
├── .python-version          # Python 3.11
└── .github/workflows/
    └── pylint.yml           # CI: lint on every push
```

## Prerequisites

| Tool   | Version | Install                                            |
| ------ | ------- | -------------------------------------------------- |
| Python | 3.11+   | [python.org](https://www.python.org/downloads/)    |
| uv     | latest  | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git    | any     | [git-scm.com](https://git-scm.com/)                |

> **Note**: We use [uv](https://docs.astral.sh/uv/) as our package manager. It handles virtual environments, dependency resolution, and Python version management.

## Getting Started

### 1. Clone the repository

```bash
git clone git@github.com:Fellowship-Khmer-OCR/facial-similarity-detection.git
cd facial-similarity-detection
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv/` virtual environment and installs all dependencies from `uv.lock`. No need to manually activate the venv — `uv run` handles that.

### 3. Obtain the dataset

The dataset is **not** included in the repository (it's ~2.7 GB and gitignored). You need to obtain it separately.

**Ask a team member** for access to the dataset, then place it so the path matches:

```
dataset/raw/subjects_0-1999_72_imgs/
```

Verify your setup:

```bash
ls dataset/raw/subjects_0-1999_72_imgs/ | head -5
# Expected: 0  1  10  100  1000
```

#### Dataset Overview

| Property           | Value                                   |
| ------------------ | --------------------------------------- |
| Subjects           | 2,000 unique identities                 |
| Images per subject | 72                                      |
| Total images       | 144,000                                 |
| Resolution         | 112 × 112 pixels                        |
| Format             | PNG, RGBA                               |
| Type               | Synthetic (GAN/diffusion-generated)     |
| Variation          | Pose, lighting, accessories, expression |

### 4. Run the EDA notebook

```bash
uv run --with jupyter jupyter lab
```

Open `eda.ipynb` and run all cells. This produces 14 analysis plots in `eda_outputs/`.

### 5. Run the application

```bash
uv run python main.py
```

## Development Workflow

### Running code

Always use `uv run` to execute Python within the project's virtual environment:

```bash
uv run python main.py
uv run python -m pytest          # when tests exist
uv run ruff check .              # lint with ruff
uv run pylint $(git ls-files '*.py')  # lint with pylint
```

### Adding dependencies

```bash
uv add <package>          # add a runtime dependency
uv add --dev <package>    # add a dev-only dependency
```

This updates both `pyproject.toml` and `uv.lock`. **Always commit both files**.

### Code quality

- **Linter**: [Ruff](https://docs.astral.sh/ruff/) (fast Python linter) and [Pylint](https://pylint.pycqa.org/) (comprehensive analysis)
- **CI**: Every push triggers Pylint via GitHub Actions (`.github/workflows/pylint.yml`)
- **Style**: Follow standard Python conventions (PEP 8). Ruff handles formatting.

Run linters locally before pushing:

```bash
uv run ruff check .
uv run pylint $(git ls-files '*.py')
```

### Git workflow

1. Create a feature branch from `main`
2. Make changes and ensure linters pass
3. Commit with descriptive messages
4. Push and open a pull request
5. Get review before merging to `main`

```bash
git checkout -b feature/your-feature-name
# ... make changes ...
uv run ruff check .
git add -A
git commit -m "feat: description of change"
git push origin feature/your-feature-name
```

## Key Findings from EDA

Refer to `eda.ipynb` for full analysis. Key takeaways:

- **Perfectly balanced** dataset — 72 images per subject, no resampling needed
- **Pre-aligned faces** at 112×112 — skip face detection preprocessing
- **Alpha channel is unused** — convert RGBA → RGB before training (saves 25% memory)
- **Raw pixel distances overlap heavily** between same-person and different-person pairs — confirms need for learned embeddings
- **Synthetic data** — may not generalize to real-world faces without domain adaptation
- **Dataset-specific normalization** should be used when training from scratch (computed in EDA Section 6)

## Recommended Train/Val/Test Split

Split by **subject**, never by image:

| Split | Subjects                 | Images  |
| ----- | ------------------------ | ------- |
| Train | 0–1399 (1,400 subjects)  | 100,800 |
| Val   | 1400–1699 (300 subjects) | 21,600  |
| Test  | 1700–1999 (300 subjects) | 21,600  |

## Troubleshooting

### `.DS_Store` errors on macOS

If you see `ValueError: invalid literal for int()` when iterating subject directories, ensure the code filters directories:

```python
subject_dirs = sorted([d for d in BASE.iterdir() if d.is_dir()])
```

### uv cache permission errors

If `uv sync` fails with cache errors:

```bash
UV_CACHE_DIR=./.uv_cache uv sync
```

### Jupyter not found

Jupyter is not a project dependency (it's a tool). Install it on-the-fly:

```bash
uv run --with jupyter jupyter lab
```
