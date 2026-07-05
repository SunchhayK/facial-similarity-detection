"""
Face Similarity Model — FaceNet + ArcFace Inspired (Custom CNN)
=================================================================

Improvements over the baseline triplet-only version:

  1. SE-Residual backbone (Squeeze-and-Excitation blocks), similar in spirit
     to the IResNet backbones used in the official ArcFace implementation.
     SE blocks let the network re-weight feature channels per-image, which
     helps with skin-tone / lighting variation — relevant since we can't
     rely on the dataset to cover every ethnicity.

  2. ArcFace additive angular margin loss (Deng et al., "ArcFace: Additive
     Angular Margin Loss for Deep Face Recognition", arXiv:1801.07698),
     implemented as an alternative/companion to triplet loss.
     ArcFace tends to train more stably than pure triplet mining because
     every sample gets gradient signal from ALL identities each step,
     not just the ones that happen to co-occur in a mined triplet.

  3. Combined loss mode: ArcFace (global structure across all identities)
     + batch-hard triplet (local fine-grained boundary refinement) — this
     mirrors what several state-of-the-art face rec papers do (train
     primarily with a margin-softmax variant, sometimes fine-tune the tail
     end with triplet/pair losses for the hardest cases).

  4. Explicit fairness / domain-generalization notes for the
     "dataset has no Cambodian faces" problem — see the big comment block
     near the bottom (GENERALIZATION_NOTES). This is a data/evaluation
     problem, not something a loss function alone fixes — read it before
     assuming this model will work equally well on Cambodian faces.

Usage:
    # ArcFace only (recommended starting point — most stable to train)
    python face_similarity_v2.py --data_root /path/to/faces --loss_type arcface --epochs 30

    # Combined ArcFace + batch-hard triplet
    python face_similarity_v2.py --data_root /path/to/faces --loss_type combined --epochs 40

    # Triplet only (original approach)
    python face_similarity_v2.py --data_root /path/to/faces --loss_type triplet --epochs 30
"""

import argparse
import csv
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Dataset + PK Sampler  (same as baseline — identity-folder structure)
# ---------------------------------------------------------------------------


class FaceIdentityDataset(Dataset):
    """Loads face images from a manifest CSV with columns:
    path, subject_id, source, split.

    Filters rows by `split`, maps each unique `subject_id` to an integer
    label, and resolves image paths relative to `project_root`.
    """

    def __init__(
        self,
        manifest_csv: str,
        project_root: str,
        split: str = "train",
        transform=None,
        min_images_per_identity: int = 2,
    ):
        self.project_root = Path(project_root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.identity_to_idx: dict[str, int] = {}
        self.idx_to_paths: dict[int, list[Path]] = {}

        # --- Parse manifest, group by subject_id for the requested split ---
        subject_paths: dict[str, list[Path]] = {}
        with open(manifest_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["split"] != split:
                    continue
                img_path = self.project_root / row["path"]
                subject_id = row["subject_id"]
                subject_paths.setdefault(subject_id, []).append(img_path)

        # --- Assign integer labels, respecting min_images_per_identity ---
        label_idx = 0
        for subject_id in sorted(subject_paths):
            paths = sorted(subject_paths[subject_id])
            if len(paths) < min_images_per_identity:
                continue
            self.identity_to_idx[subject_id] = label_idx
            self.idx_to_paths[label_idx] = paths
            for p in paths:
                self.samples.append((p, label_idx))
            label_idx += 1

        self.num_identities = label_idx
        print(
            f"[Dataset] Loaded {len(self.samples)} images across "
            f"{self.num_identities} identities (split={split!r}) "
            f"from {manifest_csv}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class PKSampler(Sampler):
    def __init__(self, dataset: FaceIdentityDataset, p: int, k: int, iterations: int):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.iterations = iterations
        self.labels = list(dataset.idx_to_paths.keys())
        self.label_to_sample_indices = {label: [] for label in self.labels}
        for sample_idx, (_, label) in enumerate(dataset.samples):
            self.label_to_sample_indices[label].append(sample_idx)

    def __len__(self):
        return self.iterations

    def __iter__(self):
        for _ in range(self.iterations):
            batch = []
            chosen_labels = random.sample(self.labels, min(self.p, len(self.labels)))
            for label in chosen_labels:
                indices = self.label_to_sample_indices[label]
                chosen = (
                    random.sample(indices, self.k)
                    if len(indices) >= self.k
                    else random.choices(indices, k=self.k)
                )
                batch.extend(chosen)
            yield batch


def get_transforms(img_size: int = 112, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                # Broader photometric jitter than the baseline: helps the model
                # not over-fit to the specific skin-tone / lighting distribution
                # of DigiFace1M + CASIA-WebFace.
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.1)
                ),  # simulates occlusion/masks
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# ---------------------------------------------------------------------------
# 2. Pretrained ResNet18 backbone → embedding
# ---------------------------------------------------------------------------


class EmbeddingNet(nn.Module):
    """
    ResNet18 (ImageNet-pretrained) → D-dim L2-normalized embedding.

    Replaces the custom SE-ResNet to leverage strong pretrained features.
    Only the final FC layer and BN are randomly initialized; all conv
    layers start from ImageNet weights, giving much better convergence
    on small/medium datasets.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the original classification head
        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )  # output: (B, 512, 1, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512, embedding_dim)
        self.bn_fc = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        x = self.features(x).flatten(1)  # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# 3. ArcFace Additive Angular Margin Loss
# ---------------------------------------------------------------------------


class ArcMarginProduct(nn.Module):
    """
    ArcFace head: replaces a normal FC classifier with one that operates
    on the angle between the embedding and each class's weight vector,
    adding an angular margin `m` to the target class before scaling by `s`.

    logits_i = s * cos(theta_i + m)   for the ground-truth class
    logits_j = s * cos(theta_j)       for all other classes

    This is used ONLY during training (to shape the embedding space via
    cross-entropy over identities). At inference you discard this head and
    just compare raw embeddings with cosine similarity.

    Supports margin warmup: call set_margin() to gradually increase m
    from 0 to the target value over the first few epochs.
    """

    def __init__(self, embedding_dim, num_classes, s=16.0, m=0.35):
        super().__init__()
        self.s = s
        self.target_m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # Start with zero margin, warm up gradually
        self.set_margin(0.0)

    def set_margin(self, m: float):
        """Update the angular margin (used for warmup)."""
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # embeddings are already L2-normalized by EmbeddingNet
        w = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w)  # (B, num_classes)
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # numerical safety: if theta + m > pi, fall back to a linear penalty
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.s
        return logits


# ---------------------------------------------------------------------------
# 4. Batch-Hard Triplet Loss (same as baseline)
# ---------------------------------------------------------------------------


def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    dot_product = embeddings @ embeddings.t()
    sq_norms = torch.diag(dot_product)
    distances = sq_norms.unsqueeze(0) - 2.0 * dot_product + sq_norms.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    mask = (distances == 0.0).float()
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances) * (1.0 - mask)
    return distances


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        distances = pairwise_distances(embeddings)
        labels = labels.unsqueeze(1)
        same_mask = labels == labels.t()
        diff_mask = ~same_mask
        eye = torch.eye(distances.size(0), device=distances.device, dtype=torch.bool)
        positive_mask = same_mask & (~eye)

        pos_dist = distances.clone()
        pos_dist[~positive_mask] = -1.0
        hardest_positive, _ = pos_dist.max(dim=1)

        max_dist = distances.max().detach() + 1.0
        neg_dist = distances.clone()
        neg_dist[~diff_mask] = max_dist
        hardest_negative, _ = neg_dist.min(dim=1)

        losses = F.relu(hardest_positive - hardest_negative + self.margin)
        valid = positive_mask.any(dim=1)
        if valid.sum() == 0:
            return losses.sum() * 0.0
        return losses[valid].mean()


# ---------------------------------------------------------------------------
# 5. Training loop — supports triplet / arcface / combined
# ---------------------------------------------------------------------------


def train(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Device] Using {device}  |  loss_type={args.loss_type}")

    train_transform = get_transforms(args.img_size, train=True)
    train_dataset = FaceIdentityDataset(
        manifest_csv=args.manifest_csv,
        project_root=args.project_root,
        split=args.split,
        transform=train_transform,
    )

    # ArcFace = classification loss → needs standard shuffle batching with all identities.
    # PK sampler only serves 16 identities/batch → 464/480 class weights get zero
    # positive gradient → ArcFace head can't learn.
    # Triplet/combined needs PK sampler for same-identity pairs.
    if args.loss_type == "arcface":
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(str(device) == "cuda"),
            drop_last=True,
        )
        print(f"[DataLoader] Standard shuffle, batch_size={args.batch_size}")
    else:
        batch_size = args.p * args.k
        iterations_per_epoch = max(1, len(train_dataset) // batch_size)
        sampler = PKSampler(
            train_dataset, p=args.p, k=args.k, iterations=iterations_per_epoch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(str(device) == "cuda"),
        )
        print(f"[DataLoader] PK sampler, p={args.p} k={args.k}")

    print("Initializing model...")
    model = EmbeddingNet(embedding_dim=args.embedding_dim).to(device)
    print("Model loaded...")

    arc_head = None
    if args.loss_type in ("arcface", "combined"):
        arc_head = ArcMarginProduct(
            args.embedding_dim,
            train_dataset.num_identities,
            s=args.arc_scale,
            m=args.arc_margin,
        ).to(device)

    triplet_loss_fn = BatchHardTripletLoss(margin=args.triplet_margin)
    ce_loss_fn = nn.CrossEntropyLoss()

    # Differential LR: backbone features need lower LR to preserve pretrained weights
    backbone_lr = args.lr * args.backbone_lr_factor
    param_groups = [
        {"params": model.features.parameters(), "lr": backbone_lr},
        {"params": list(model.dropout.parameters()) + list(model.fc.parameters()) + list(model.bn_fc.parameters()), "lr": args.lr},
    ]
    if arc_head is not None:
        param_groups.append({"params": arc_head.parameters(), "lr": args.lr})

    all_params = [p for group in param_groups for p in group["params"]]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Margin warmup config
    warmup_epochs = min(5, args.epochs // 3)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    print(
        f"Starting to train... "
        f"(backbone_lr={backbone_lr:.1e}, head_lr={args.lr:.1e}, "
        f"margin_warmup={warmup_epochs} epochs)"
    )

    best_metric = float("-inf")  # higher is better (accuracy)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Margin warmup: linearly ramp from 0 to target margin
        if arc_head is not None and epoch <= warmup_epochs:
            warmup_m = args.arc_margin * (epoch / warmup_epochs)
            arc_head.set_margin(warmup_m)
        elif arc_head is not None and epoch == warmup_epochs + 1:
            arc_head.set_margin(args.arc_margin)

        model.train()
        if arc_head is not None:
            arc_head.train()

        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)

            loss = 0.0
            if args.loss_type == "triplet":
                loss = triplet_loss_fn(embeddings, labels)
            elif args.loss_type == "arcface":
                logits = arc_head(embeddings, labels)
                loss = ce_loss_fn(logits, labels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            elif args.loss_type == "combined":
                logits = arc_head(embeddings, labels)
                arc_loss = ce_loss_fn(logits, labels)
                trip_loss = triplet_loss_fn(embeddings, labels)
                loss = args.arc_weight * arc_loss + args.triplet_weight * trip_loss
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = running_loss / max(1, num_batches)
        current_m = arc_head.m if arc_head else 0.0
        train_acc = correct / max(1, total) if total > 0 else 0.0

        log_parts = [
            f"Epoch [{epoch}/{args.epochs}] loss={avg_loss:.4f}",
            f"lr={scheduler.get_last_lr()[0]:.6f}",
            f"margin={current_m:.3f}",
        ]
        if total > 0:
            log_parts.append(f"train_acc={train_acc:.4f}")
        print("  ".join(log_parts))

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "img_size": args.img_size,
            "loss_type": args.loss_type,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))

        # Checkpoint selection: accuracy for arcface/combined, loss for triplet
        if total > 0:
            if train_acc > best_metric:
                best_metric = train_acc
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  -> New best model saved (train_acc={train_acc:.4f})")
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
                print(f"  -> New best model saved (loss={best_loss:.4f})")

    print(
        "Training complete. NOTE: the ArcMarginProduct head is discarded at "
        "inference time — only EmbeddingNet's weights matter for similarity."
    )

    # --- Post-training evaluation ---
    if getattr(args, "eval_after", False):
        print(
            f"\n[Post-Train] Running FAR/FRR evaluation on '{args.eval_split}' split..."
        )
        eval_transform = get_transforms(args.img_size, train=False)
        eval_dataset = FaceIdentityDataset(
            manifest_csv=args.manifest_csv,
            project_root=args.project_root,
            split=args.eval_split,
            transform=eval_transform,
        )
        if len(eval_dataset) > 0:
            embeddings, labels = extract_embeddings(model, eval_dataset, device)
            results = compute_far_frr(embeddings, labels, max_pairs=args.max_pairs)
            print_eval_summary(results)
        else:
            print(f"  No images found for split '{args.eval_split}' — skipping eval.")


# ---------------------------------------------------------------------------
# 6. Inference: compare two faces via cosine similarity
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_similarity(
    model, img_path1, img_path2, device, img_size=112, threshold=0.35
):
    """
    Returns (cosine_distance, is_same_person).
    cosine_distance = 1 - cosine_similarity, so 0 = identical direction,
    2 = opposite. Calibrate `threshold` on a labeled validation set (see
    GENERALIZATION_NOTES below for why this matters more than usual here).
    """
    model.eval()
    transform = get_transforms(img_size, train=False)
    img1 = transform(Image.open(img_path1).convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(Image.open(img_path2).convert("RGB")).unsqueeze(0).to(device)
    emb1, emb2 = model(img1), model(img2)
    cosine_sim = F.cosine_similarity(emb1, emb2).item()
    cosine_dist = 1.0 - cosine_sim
    return cosine_dist, cosine_dist < threshold


# ---------------------------------------------------------------------------
# 6b. Evaluation: FAR / FRR / EER
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=64):
    """Extract embeddings for all images in dataset. Returns (embeddings, labels) tensors."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(str(device) == "cuda"),
    )
    all_embs, all_labels = [], []
    for images, labels in loader:
        embs = model(images.to(device)).cpu()
        all_embs.append(embs)
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


def compute_far_frr(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor | None = None,
    max_pairs: int = 500_000,
) -> dict:
    """
    Compute FAR, FRR, and EER over a sweep of cosine-similarity thresholds.

    Generates genuine (same-identity) and impostor (different-identity) pairs,
    computes cosine similarity for each, then sweeps thresholds.

    Args:
        embeddings: (N, D) L2-normalized embeddings.
        labels: (N,) integer identity labels.
        thresholds: optional 1-D tensor of thresholds to evaluate.
        max_pairs: cap on total pairs to avoid OOM on large sets.

    Returns:
        Dict with keys: thresholds, far, frr, eer, eer_threshold,
        num_genuine, num_impostor.
    """
    n = len(embeddings)
    # Build all pair indices
    idx_i, idx_j = [], []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i.append(i)
            idx_j.append(j)
    idx_i = torch.tensor(idx_i)
    idx_j = torch.tensor(idx_j)

    # Subsample if too many pairs
    total_pairs = len(idx_i)
    if total_pairs > max_pairs:
        perm = torch.randperm(total_pairs)[:max_pairs]
        idx_i, idx_j = idx_i[perm], idx_j[perm]
        print(f"  [Eval] Subsampled {max_pairs} pairs from {total_pairs}")

    # Cosine similarities (embeddings already L2-normed)
    sims = (embeddings[idx_i] * embeddings[idx_j]).sum(dim=1)
    is_genuine = labels[idx_i] == labels[idx_j]
    is_impostor = ~is_genuine

    num_genuine = is_genuine.sum().item()
    num_impostor = is_impostor.sum().item()
    print(f"  [Eval] Pairs — genuine: {num_genuine}, impostor: {num_impostor}")

    if num_genuine == 0 or num_impostor == 0:
        print("  [Eval] WARNING: need both genuine and impostor pairs for FAR/FRR.")
        return {"error": "insufficient_pairs"}

    genuine_sims = sims[is_genuine]
    impostor_sims = sims[is_impostor]

    if thresholds is None:
        thresholds = torch.linspace(0.0, 1.0, steps=201)

    far_list, frr_list = [], []
    for t in thresholds:
        # FAR: fraction of impostor pairs incorrectly accepted (sim >= threshold)
        far = (impostor_sims >= t).float().mean().item()
        # FRR: fraction of genuine pairs incorrectly rejected (sim < threshold)
        frr = (genuine_sims < t).float().mean().item()
        far_list.append(far)
        frr_list.append(frr)

    far_t = torch.tensor(far_list)
    frr_t = torch.tensor(frr_list)

    # EER: where FAR ≈ FRR (find crossing point)
    diff = (far_t - frr_t).abs()
    eer_idx = diff.argmin().item()
    eer = (far_t[eer_idx].item() + frr_t[eer_idx].item()) / 2.0
    eer_threshold = thresholds[eer_idx].item()

    return {
        "thresholds": thresholds.tolist(),
        "far": far_list,
        "frr": frr_list,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "num_genuine": num_genuine,
        "num_impostor": num_impostor,
    }


def print_eval_summary(results: dict) -> None:
    """Print a formatted evaluation summary with key FAR/FRR operating points."""
    if "error" in results:
        print(f"  [Eval] Error: {results['error']}")
        return

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (FAR / FRR)")
    print("=" * 60)
    print(f"  Genuine pairs:  {results['num_genuine']}")
    print(f"  Impostor pairs: {results['num_impostor']}")
    print(
        f"  EER:            {results['eer']:.4f} @ threshold={results['eer_threshold']:.4f}"
    )

    # Print FAR/FRR at common operating points
    thresholds = results["thresholds"]
    far = results["far"]
    frr = results["frr"]
    print("\n  Threshold    FAR        FRR")
    print("  ---------    -------    -------")
    for target_t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        # Find closest threshold
        idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - target_t))
        print(f"  {thresholds[idx]:.2f}         {far[idx]:.4f}     {frr[idx]:.4f}")
    print("=" * 60 + "\n")


def evaluate(args):
    """Load a checkpoint and evaluate FAR/FRR on val or test split."""
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[Device] Using {device}")

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embedding_dim = ckpt.get("embedding_dim", args.embedding_dim)
    img_size = ckpt.get("img_size", args.img_size)

    model = EmbeddingNet(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[Model] Loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    eval_transform = get_transforms(img_size, train=False)
    eval_dataset = FaceIdentityDataset(
        manifest_csv=args.manifest_csv,
        project_root=args.project_root,
        split=args.eval_split,
        transform=eval_transform,
    )

    if len(eval_dataset) == 0:
        print(f"No images found for split '{args.eval_split}'")
        return

    print(f"[Eval] Extracting embeddings for {len(eval_dataset)} images...")
    embeddings, labels = extract_embeddings(model, eval_dataset, device)

    print("[Eval] Computing FAR/FRR...")
    results = compute_far_frr(embeddings, labels, max_pairs=args.max_pairs)
    print_eval_summary(results)
    return results


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="FaceNet+ArcFace-inspired face similarity training & evaluation"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or evaluate")

    # --- Shared args ---
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--manifest_csv",
        type=str,
        default="dataset/splits/manifest.csv",
        help="Path to manifest CSV (columns: path, subject_id, source, split)",
    )
    parent.add_argument(
        "--project_root",
        type=str,
        default=".",
        help="Root directory for resolving image paths in manifest",
    )
    parent.add_argument("--img_size", type=int, default=112)
    parent.add_argument("--embedding_dim", type=int, default=256)
    parent.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    # --- Train subcommand ---
    train_parser = subparsers.add_parser(
        "train", parents=[parent], help="Train the model"
    )
    train_parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to load from manifest (train/val/test)",
    )
    train_parser.add_argument(
        "--loss_type",
        type=str,
        default="arcface",
        choices=["triplet", "arcface", "combined"],
    )
    train_parser.add_argument("--triplet_margin", type=float, default=0.3)
    train_parser.add_argument("--arc_scale", type=float, default=32.0)
    train_parser.add_argument("--arc_margin", type=float, default=0.35)
    train_parser.add_argument("--arc_weight", type=float, default=1.0)
    train_parser.add_argument("--triplet_weight", type=float, default=0.5)
    train_parser.add_argument("--batch_size", type=int, default=128,
                              help="Batch size for arcface mode (default: 128)")
    train_parser.add_argument("--p", type=int, default=16)
    train_parser.add_argument("--k", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                              help="Backbone LR = lr * this factor (default: 0.1)")
    train_parser.add_argument("--weight_decay", type=float, default=5e-4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument(
        "--eval_after",
        action="store_true",
        default=True,
        help="Run FAR/FRR evaluation on val set after training",
    )
    train_parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Split to evaluate on after training (default: val)",
    )
    train_parser.add_argument("--max_pairs", type=int, default=500_000)

    # --- Evaluate subcommand ---
    eval_parser = subparsers.add_parser(
        "evaluate", parents=[parent], help="Evaluate FAR/FRR"
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Which split to evaluate on (val/test)",
    )
    eval_parser.add_argument("--max_pairs", type=int, default=500_000)

    args = parser.parse_args(argv)

    # Default to train mode if no subcommand given
    if args.mode is None:
        args.mode = "train"
        # Re-parse with train defaults
        train_parser.parse_args(namespace=args)

    return args


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)


# ---------------------------------------------------------------------------
# GENERALIZATION_NOTES: making this work on Cambodian faces with zero
# Cambodian training data
# ---------------------------------------------------------------------------
#
# No loss function or architecture trick fully substitutes for the model
# never having seen the population it's being evaluated on. Be direct with
# stakeholders about this. That said, several things measurably help:
#
# 1. DATA COMPOSITION MATTERS MORE THAN LOSS CHOICE.
#    - CASIA-WebFace is predominantly East Asian (Chinese) and Western
#      celebrity faces. It gives some transferable East/Southeast Asian
#      facial structure priors, but Cambodian (Khmer) facial feature
#      distributions, skin tone range, and bone structure differ from
#      Chinese/Korean/Japanese populations in ways that matter for a model
#      this size.
#    - DigiFace1M's synthetic identities are generated from a parametric
#      3D model with some ethnicity/appearance variation, but "some Asian-
#      looking synthetic faces" is not the same as real Khmer face
#      statistics — synthetic faces also have their own texture/lighting
#      "tell" that real photos don't share (a well-documented sim-to-real
#      gap in face recognition literature).
#    - If at all possible, add a real, even small and unlabeled or weakly-
#      labeled set of Cambodian faces for fine-tuning or at minimum
#      evaluation (with informed consent from the people photographed —
#      face data is sensitive biometric data in most legal frameworks,
#      including Cambodia's growing data protection regulation and the
#      EU's GDPR if any subject is an EU resident). Public research sets
#      that broaden Southeast Asian representation beyond CASIA (e.g.
#      RFW / BUPT-Balancedface for benchmarking, or region-specific
#      academic datasets you can request access to) are worth searching
#      for before you assume you have to collect from scratch.
#
# 2. FINE-TUNING WITH A SMALL LOCAL SET IS SURPRISINGLY EFFECTIVE.
#    Even 20-50 Cambodian identities x 5-10 images each, used to fine-tune
#    just the last stage + FC of EmbeddingNet (freeze stages 1-3) for a
#    handful of epochs at a low LR (~1e-5), typically closes a meaningful
#    chunk of the domain gap without needing to retrain from scratch.
#
# 3. EVALUATE PER-SUBGROUP, NOT JUST OVERALL ACCURACY.
#    A model can look great on an aggregate validation accuracy number
#    while performing far worse on the specific subgroup you care about.
#    Build a held-out validation set of Cambodian face pairs (same-person
#    and different-person) specifically, and report accuracy/FAR/FRR on
#    that subset separately from your main validation metric. This is the
#    single most important step — you cannot know if the model works for
#    Cambodian faces unless you test on Cambodian faces.
#
# 4. AUGMENTATION HELPS AT THE MARGIN, NOT AS A SUBSTITUTE FOR DATA.
#    The broadened ColorJitter/blur/erasing augmentations added above make
#    the model less brittle to lighting and skin-tone shifts it wasn't
#    exactly trained on, and SE blocks give the network a mechanism to
#    re-weight channels per-image rather than relying on a fixed filter
#    bank tuned to the training distribution's average appearance. Both
#    help generalization somewhat — neither is a substitute for #1-#3.
#
# 5. BE CAUTIOUS ABOUT DEPLOYMENT CLAIMS.
#    If this model will be used for any real identification/verification
#    decision involving real people (as opposed to a research prototype),
#    document its known training-data composition and validated accuracy
#    per subgroup, and disclose the gap honestly rather than assuming
#    good aggregate numbers transfer to an unrepresented population.
# ---------------------------------------------------------------------------
