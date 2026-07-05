"""
Gallery Manager — Persistent face embedding store with similarity search.

Manages a gallery of face embeddings organized by identity.
Supports multi-image enrollment per identity (mean-embedding ranking)
and top-K similarity search with confidence flagging.

Storage layout:
    gallery/
    ├── gallery.json          # metadata + embedding vectors
    └── images/
        └── <identity_id>/
            ├── img_001.jpg
            └── img_002.jpg
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from face_similarity import EmbeddingNet, get_transforms


class GalleryManager:
    """Persistent face embedding gallery with enroll/search/manage operations.

    Each identity can hold multiple face images. At search time, per-identity
    embeddings are averaged (and re-normalized) to produce a single centroid
    for cosine-similarity ranking.
    """

    def __init__(
        self,
        gallery_dir: str = "./gallery",
        checkpoint_path: str = "./checkpoints/best.pt",
        device: str | None = None,
        threshold: float = 0.5,
    ):
        self.gallery_dir = Path(gallery_dir)
        self.images_dir = self.gallery_dir / "images"
        self.gallery_path = self.gallery_dir / "gallery.json"
        self.threshold = threshold

        self.device = self._resolve_device(device)
        self.model, self.img_size = self._load_model(checkpoint_path)
        self.gallery = self._load_gallery()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self, checkpoint_path: str) -> tuple[EmbeddingNet, int]:
        """Load EmbeddingNet from a training checkpoint."""
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        embedding_dim = ckpt.get("embedding_dim", 256)
        img_size = ckpt.get("img_size", 112)

        model = EmbeddingNet(embedding_dim=embedding_dim).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        print(
            f"[Gallery] Model loaded from {ckpt_path} "
            f"(epoch {ckpt.get('epoch', '?')}, dim={embedding_dim})"
        )
        return model, img_size

    def _load_gallery(self) -> dict:
        """Load gallery from disk or initialize empty."""
        if self.gallery_path.exists():
            with open(self.gallery_path) as f:
                gallery = json.load(f)
            total_images = sum(
                len(data["images"]) for data in gallery["identities"].values()
            )
            print(
                f"[Gallery] Loaded {len(gallery['identities'])} identities, "
                f"{total_images} images"
            )
            return gallery
        return {"identities": {}}

    def _save_gallery(self) -> None:
        """Persist gallery to disk atomically."""
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.gallery_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.gallery, f, indent=2)
        tmp_path.replace(self.gallery_path)

    @torch.no_grad()
    def _embed(self, image_path: str | Path) -> list[float]:
        """Compute L2-normalized embedding for a single face image."""
        transform = get_transforms(self.img_size, train=False)
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)
        embedding = self.model(tensor).cpu().squeeze(0)
        return embedding.tolist()

    def _mean_embedding(self, identity_id: str) -> np.ndarray | None:
        """Compute L2-normalized mean embedding for an identity."""
        data = self.gallery["identities"].get(identity_id)
        if not data or not data["images"]:
            return None
        embeddings = np.array([img["embedding"] for img in data["images"]])
        mean = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-8:
            return None
        return mean / norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(self, identity_id: str, label: str, image_path: str) -> dict:
        """Enroll a face image under an identity.

        Creates the identity if new, appends if existing.
        Image is copied into the gallery directory for persistence.

        Returns:
            Dict with enrollment summary.

        Raises:
            FileNotFoundError: If image_path does not exist.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        embedding = self._embed(image_path)

        # Copy image into gallery
        identity_dir = self.images_dir / identity_id
        identity_dir.mkdir(parents=True, exist_ok=True)

        existing_count = len(list(identity_dir.glob("*")))
        dest_filename = f"img_{existing_count + 1:03d}{image_path.suffix}"
        dest_path = identity_dir / dest_filename
        shutil.copy2(image_path, dest_path)

        # Upsert identity
        if identity_id not in self.gallery["identities"]:
            self.gallery["identities"][identity_id] = {
                "label": label,
                "images": [],
            }

        self.gallery["identities"][identity_id]["images"].append(
            {
                "filename": dest_filename,
                "embedding": embedding,
                "added_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Allow label correction on subsequent enrollments
        self.gallery["identities"][identity_id]["label"] = label

        self._save_gallery()

        total = len(self.gallery["identities"][identity_id]["images"])
        print(f"[Enroll] '{label}' (id={identity_id}) — {total} image(s) total")

        return {
            "identity_id": identity_id,
            "label": label,
            "image_count": total,
            "stored_at": str(dest_path),
        }

    def search(self, query_image_path: str, top_k: int = 5) -> list[dict]:
        """Search gallery for faces similar to query image.

        Returns top-K identities ranked by cosine similarity (descending).
        Each result includes a 'confident' flag based on self.threshold.

        Returns:
            List of dicts with identity_id, label, similarity, confident, image_count.
        """
        if not self.gallery["identities"]:
            print("[Search] Gallery is empty. Enroll faces first.")
            return []

        query_emb = np.array(self._embed(query_image_path))

        results = []
        for identity_id, data in self.gallery["identities"].items():
            mean_emb = self._mean_embedding(identity_id)
            if mean_emb is None:
                continue

            similarity = float(np.dot(query_emb, mean_emb))
            results.append(
                {
                    "identity_id": identity_id,
                    "label": data["label"],
                    "similarity": round(similarity, 4),
                    "confident": similarity >= self.threshold,
                    "image_count": len(data["images"]),
                }
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def list_identities(self) -> list[dict]:
        """List all enrolled identities with metadata."""
        entries = []
        for identity_id, data in self.gallery["identities"].items():
            entries.append(
                {
                    "identity_id": identity_id,
                    "label": data["label"],
                    "image_count": len(data["images"]),
                    "first_enrolled": (
                        data["images"][0]["added_at"] if data["images"] else None
                    ),
                }
            )
        return entries

    def delete_identity(self, identity_id: str) -> bool:
        """Remove an identity and all its images from the gallery.

        Returns:
            True if deleted, False if identity not found.
        """
        if identity_id not in self.gallery["identities"]:
            print(f"[Delete] Identity '{identity_id}' not found.")
            return False

        label = self.gallery["identities"][identity_id]["label"]
        del self.gallery["identities"][identity_id]

        # Remove stored images
        identity_dir = self.images_dir / identity_id
        if identity_dir.exists():
            shutil.rmtree(identity_dir)

        self._save_gallery()
        print(f"[Delete] Removed '{label}' (id={identity_id})")
        return True

    def get_identity_image_paths(self, identity_id: str) -> list[Path]:
        """Get filesystem paths to all stored images for an identity."""
        if identity_id not in self.gallery["identities"]:
            return []
        identity_dir = self.images_dir / identity_id
        if not identity_dir.exists():
            return []
        return sorted(identity_dir.iterdir())

    @property
    def identity_count(self) -> int:
        return len(self.gallery["identities"])

    @property
    def total_image_count(self) -> int:
        return sum(len(data["images"]) for data in self.gallery["identities"].values())
