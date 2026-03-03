"""
AUROC Evaluation Script for DeepfakeDetector_v3.

Loads a trained Detector checkpoint and evaluates it on:
  - FaceForensics++ test split  (per-method + aggregate)
  - Celeb-DF test/val split     (cross-dataset generalization)

Outputs: AUROC, accuracy, precision, recall, F1 per dataset,
         and saves a combined ROC curve plot.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from models.detector import Detector
from utils.weights import load_weights
from utils.preprocess import build_preprocess
from datasets.faceforensics import FaceForensicsDataset
from datasets.celeb_df import CelebDFDataset


# ── weight discovery ─────────────────────────────────────────────────
WEIGHT_CANDIDATES = [
    "training/weights/dinov3_best_v4.pth",
    "training/weights/dinov3_best_v3.pth",
    "training/weights/dinov3_best_v2.pth",
    "training/weights/dinov3_best.pth",
    "training/weights/dinov2_best.pth",
]


def _find_weights(explicit: str | None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    for p in WEIGHT_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "No checkpoint found.  Pass --weights <path> or place a .pth in training/weights/"
    )


# ── evaluation core ──────────────────────────────────────────────────
@torch.no_grad()
def collect_predictions(
    detector: Detector,
    loader: DataLoader,
    device: str,
    desc: str = "Evaluating",
):
    """Run the detector on a paired dataset loader and return labels + probs."""
    detector.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []

    for real_imgs, fake_imgs, _ in tqdm(loader, desc=desc, leave=False):
        imgs = torch.cat([real_imgs, fake_imgs], dim=0).to(device)
        labels = torch.cat(
            [torch.zeros(real_imgs.size(0)), torch.ones(fake_imgs.size(0))],
            dim=0,
        ).long()

        logits, _ = detector.forward(imgs)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake)

        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    return np.array(all_labels), np.array(all_probs)


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    preds = (probs >= threshold).astype(int)
    metrics = {
        "AUROC": roc_auc_score(labels, probs),
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "Samples": int(len(labels)),
    }
    return metrics


def print_metrics(name: str, m: dict):
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  AUROC      : {m['AUROC']:.4f}")
    print(f"  Accuracy   : {m['Accuracy']:.4f}")
    print(f"  Precision  : {m['Precision']:.4f}")
    print(f"  Recall     : {m['Recall']:.4f}")
    print(f"  F1 Score   : {m['F1']:.4f}")
    print(f"  Samples    : {m['Samples']}")


def save_roc_plot(curves: dict, save_path: str = "roc_curve.png"):
    """Plot and save ROC curves for each dataset/method."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed – skipping ROC plot.")
        return

    plt.figure(figsize=(8, 6))

    for name, (labels, probs) in curves.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f"{name}  (AUROC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Deepfake Detector – ROC Curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nROC curve saved to {save_path}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AUROC evaluation for DeepfakeDetector")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--plot", type=str, default="roc_curve.png", help="Path to save ROC plot")
    parser.add_argument(
        "--ff-root", type=str, default="data/processed_ff", help="FaceForensics++ root"
    )
    parser.add_argument(
        "--celeb-root", type=str, default="data/processed_celeb", help="Celeb-DF root"
    )
    parser.add_argument(
        "--celeb-only", action="store_true", default=False,
        help="Only evaluate cross-dataset AUROC on Celeb-DF (skip FF++)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────
    weights_path = _find_weights(args.weights)
    print(f"Checkpoint: {weights_path}")

    detector = Detector(device=device)
    load_weights(detector, weights_path)
    detector.eval()

    preprocess = build_preprocess(image_size=detector.encoder.image_size)

    num_workers = 0 if os.name == "nt" else 4
    roc_curves: dict[str, tuple] = {}
    all_labels_global = []
    all_probs_global = []

    # ── FaceForensics++ (test split, per method) ─────────────────
    ff_methods = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    if not args.celeb_only and os.path.isdir(args.ff_root):
        for method in ff_methods:
            ds = FaceForensicsDataset(
                args.ff_root, method=method, split="test", transform=preprocess
            )
            if len(ds) == 0:
                print(f"  [skip] FF++ {method}: no test frames found")
                continue

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )
            labels, probs = collect_predictions(
                detector, loader, device, desc=f"FF++ {method}"
            )
            m = compute_metrics(labels, probs, threshold=args.threshold)
            print_metrics(f"FF++ – {method} (test)", m)

            roc_curves[f"FF++ {method}"] = (labels, probs)
            all_labels_global.append(labels)
            all_probs_global.append(probs)

        # FF++ aggregate
        if all_labels_global:
            agg_l = np.concatenate(all_labels_global)
            agg_p = np.concatenate(all_probs_global)
            m = compute_metrics(agg_l, agg_p, threshold=args.threshold)
            print_metrics("FF++ – ALL METHODS (test)", m)
            roc_curves["FF++ All Methods"] = (agg_l, agg_p)
    else:
        print(f"FF++ root not found at {args.ff_root} – skipping.")

    # ── Celeb-DF (test / val) ────────────────────────────────────
    if os.path.isdir(args.celeb_root):
        ds_celeb = CelebDFDataset(args.celeb_root, split="test", transform=preprocess)
        if len(ds_celeb) == 0:
            ds_celeb = CelebDFDataset(args.celeb_root, split="val", transform=preprocess)
        if len(ds_celeb) > 0:
            loader = DataLoader(
                ds_celeb,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device == "cuda"),
            )
            labels, probs = collect_predictions(
                detector, loader, device, desc="Celeb-DF"
            )
            m = compute_metrics(labels, probs, threshold=args.threshold)
            print_metrics("Celeb-DF (cross-dataset)", m)
            roc_curves["Celeb-DF"] = (labels, probs)
        else:
            print("Celeb-DF: no frames found – skipping.")
    else:
        print(f"Celeb-DF root not found at {args.celeb_root} – skipping.")

    # ── ROC plot ─────────────────────────────────────────────────
    if roc_curves:
        save_roc_plot(roc_curves, save_path=args.plot)
    else:
        print("\nNo data was evaluated – nothing to plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
