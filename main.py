"""
Unified CLI entry point for the Face Similarity Detection pipeline.

Subcommands:
    train       Train the embedding model (delegates to face_similarity.py)
    evaluate    Evaluate FAR/FRR metrics (delegates to face_similarity.py)
    enroll      Add a face image to the gallery
    search      Find similar faces in the gallery
    list        Show all enrolled identities
    delete      Remove an identity from the gallery

Examples:
    python main.py train --loss_type triplet --epochs 30
    python main.py evaluate --checkpoint ./checkpoints/best.pt --eval_split val
    python main.py enroll --id sokha_001 --label "Sokha" --image face.jpg
    python main.py search --image query.jpg --top_k 5
    python main.py list
    python main.py delete --id sokha_001
"""

import argparse
import sys

from gallery import GalleryManager


# ------------------------------------------------------------------
# CLI definition
# ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Face Similarity Detection — training, evaluation, and gallery search"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train: delegate to face_similarity.py ---
    train_p = subparsers.add_parser(
        "train",
        help="Train the model (all args forwarded to face_similarity.py)",
    )
    train_p.add_argument(
        "train_args", nargs=argparse.REMAINDER,
        help="Arguments forwarded to face_similarity.py train",
    )

    # --- evaluate: delegate to face_similarity.py ---
    eval_p = subparsers.add_parser(
        "evaluate",
        help="Evaluate FAR/FRR (all args forwarded to face_similarity.py)",
    )
    eval_p.add_argument(
        "eval_args", nargs=argparse.REMAINDER,
        help="Arguments forwarded to face_similarity.py evaluate",
    )

    # --- Shared gallery args ---
    gallery_parent = argparse.ArgumentParser(add_help=False)
    gallery_parent.add_argument(
        "--checkpoint", default="./checkpoints/best.pt",
        help="Path to model checkpoint (default: ./checkpoints/best.pt)",
    )
    gallery_parent.add_argument(
        "--gallery_dir", default="./gallery",
        help="Gallery storage directory (default: ./gallery)",
    )

    # --- enroll ---
    enroll_p = subparsers.add_parser(
        "enroll", parents=[gallery_parent],
        help="Enroll a face image into the gallery",
    )
    enroll_p.add_argument("--id", required=True, help="Unique identity ID")
    enroll_p.add_argument("--label", required=True, help="Display name for the identity")
    enroll_p.add_argument("--image", required=True, help="Path to face image")
    enroll_p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)",
    )

    # --- search ---
    search_p = subparsers.add_parser(
        "search", parents=[gallery_parent],
        help="Search gallery for similar faces",
    )
    search_p.add_argument("--image", required=True, help="Path to query face image")
    search_p.add_argument(
        "--top_k", type=int, default=5, help="Number of results (default: 5)",
    )
    search_p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)",
    )

    # --- list ---
    subparsers.add_parser(
        "list", parents=[gallery_parent],
        help="List all enrolled identities",
    )

    # --- delete ---
    delete_p = subparsers.add_parser(
        "delete", parents=[gallery_parent],
        help="Delete an identity from the gallery",
    )
    delete_p.add_argument("--id", required=True, help="Identity ID to delete")

    return parser


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------


def cmd_train(args):
    """Delegate to face_similarity.py train."""
    from face_similarity import parse_args as fs_parse_args, train

    argv = ["train"] + (args.train_args or [])
    fs_args = fs_parse_args(argv)
    train(fs_args)


def cmd_evaluate(args):
    """Delegate to face_similarity.py evaluate."""
    from face_similarity import parse_args as fs_parse_args, evaluate

    argv = ["evaluate"] + (args.eval_args or [])
    fs_args = fs_parse_args(argv)
    evaluate(fs_args)


def cmd_enroll(args):
    """Enroll a face image into the gallery."""
    gm = GalleryManager(
        gallery_dir=args.gallery_dir,
        checkpoint_path=args.checkpoint,
        threshold=getattr(args, "threshold", 0.5),
    )
    result = gm.enroll(
        identity_id=args.id,
        label=args.label,
        image_path=args.image,
    )
    print(f"\nEnrolled: {result['label']} ({result['identity_id']})")
    print(f"  Images: {result['image_count']}")
    print(f"  Stored: {result['stored_at']}")


def cmd_search(args):
    """Search the gallery for similar faces."""
    gm = GalleryManager(
        gallery_dir=args.gallery_dir,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    results = gm.search(query_image_path=args.image, top_k=args.top_k)

    if not results:
        print("No results. Gallery may be empty.")
        return

    print(f"\nTop-{args.top_k} matches (threshold={args.threshold}):")
    print(f"{'Rank':<6}{'Label':<20}{'Similarity':<12}{'Confident':<10}{'Images':<8}")
    print("-" * 56)
    for i, r in enumerate(results, 1):
        conf = "✓" if r["confident"] else "✗"
        print(
            f"{i:<6}{r['label']:<20}{r['similarity']:<12.4f}"
            f"{conf:<10}{r['image_count']:<8}"
        )


def cmd_list(args):
    """List all enrolled identities."""
    gm = GalleryManager(
        gallery_dir=args.gallery_dir,
        checkpoint_path=args.checkpoint,
    )
    identities = gm.list_identities()

    if not identities:
        print("Gallery is empty.")
        return

    print(f"\nGallery: {len(identities)} identities, {gm.total_image_count} images")
    print(f"{'ID':<20}{'Label':<20}{'Images':<8}{'First Enrolled':<26}")
    print("-" * 74)
    for entry in identities:
        print(
            f"{entry['identity_id']:<20}{entry['label']:<20}"
            f"{entry['image_count']:<8}{entry['first_enrolled'] or 'N/A':<26}"
        )


def cmd_delete(args):
    """Delete an identity from the gallery."""
    gm = GalleryManager(
        gallery_dir=args.gallery_dir,
        checkpoint_path=args.checkpoint,
    )
    deleted = gm.delete_identity(args.id)
    if deleted:
        print(f"Identity '{args.id}' deleted.")
    else:
        print(f"Identity '{args.id}' not found.")
        sys.exit(1)


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

COMMANDS = {
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "enroll": cmd_enroll,
    "search": cmd_search,
    "list": cmd_list,
    "delete": cmd_delete,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
