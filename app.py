"""
Robust wrapper to run Wavelet-CLIP inference.

This script will attempt to clone the official Wavelet-CLIP repo into ./wavelet-clip
if it cannot find required modules locally. It then dynamically imports the
WaveletCLIP model and wavelet utility functions and runs inference on a video.

Note: pretrained weights must be available at training/weights/clip_wavelet_best.pth
or provide a URL and run `scripts/download_weights.sh <url>` first.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
WCLIP_DIR = PROJECT_ROOT / "wavelet-clip"


def ensure_wavelet_repo():
    """Clone the official Wavelet-CLIP repo into ./wavelet-clip if missing."""
    if WCLIP_DIR.exists():
        return True
    print("Wavelet-CLIP repo not found locally. Cloning into ./wavelet-clip...")
    try:
        subprocess.check_call(["git", "clone", "https://github.com/lalithbharadwajbaru/wavelet-clip.git", str(WCLIP_DIR)])
        return True
    except Exception as e:
        print("Failed to clone wavelet-clip:", e)
        return False


def import_wavelet_modules():
    """Add wavelet-clip to sys.path and import required modules dynamically."""
    if not ensure_wavelet_repo():
        raise RuntimeError("wavelet-clip repository not available. Please clone it manually.")

    sys.path.insert(0, str(WCLIP_DIR))
    # Try to import model and utils from that repo
    try:
        model_mod = importlib.import_module("model")
        utils_mod = importlib.import_module("wavelet_utils")
        return model_mod, utils_mod
    except Exception as e:
        # Attempt from training or src folders if structure differs
        alt_path = WCLIP_DIR / "training"
        if alt_path.exists():
            sys.path.insert(0, str(alt_path))
            try:
                model_mod = importlib.import_module("model")
                utils_mod = importlib.import_module("wavelet_utils")
                return model_mod, utils_mod
            except Exception:
                pass
        raise


def check_weights(weights_path=None):
    """Return True if the Wavelet-CLIP weights file exists."""
    if weights_path is None:
        weights_path = PROJECT_ROOT / "training" / "weights" / "clip_wavelet_best.pth"
    return Path(weights_path).exists()


def video_to_wavelet_features(video_path, preprocess, apply_wavelet_transform, max_frames=16):
    import cv2
    from PIL import Image
    import torch

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = preprocess(Image.fromarray(img)).unsqueeze(0).cuda()
        wavelet_features = apply_wavelet_transform(img)
        frames.append(wavelet_features)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video")
    return torch.cat(frames, dim=0)


def detect_fake_video(video_path, weights_path=None, max_frames=16):
    import torch
    import clip

    model_mod, utils_mod = import_wavelet_modules()

    # Load CLIP and preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # Instantiate WaveletCLIP (assumes default constructor exists in the repo)
    WaveletCLIP = getattr(model_mod, "WaveletCLIP", None)
    if WaveletCLIP is None:
        raise RuntimeError("WaveletCLIP class not found in model module")

    model = WaveletCLIP().to(device)

    # Determine weights path
    if weights_path is None:
        weights_path = PROJECT_ROOT / "training" / "weights" / "clip_wavelet_best.pth"
    else:
        weights_path = Path(weights_path)

    # If weights are missing, offer a CLIP-only baseline instead of failing hard.
    if not Path(weights_path).exists():
        # Baseline: use CLIP textual similarity to estimate fakeness
        print(f"Warning: Weights not found at {weights_path}.")
        print("Falling back to a CLIP ViT-L/14 text-similarity baseline. To run the real Wavelet-CLIP detector, place clip_wavelet_best.pth in training/weights/ or provide a URL to download it.")

        # CLIP baseline: compare frames against two prompts and return average 'fake' probability
        prompts = ["a real person", "an AI generated face"]
        text_tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        import torch
        img_feats = []
        cap = None
        try:
            cap = None
            # reuse the video_to_wavelet_features code but bypass wavelet transform: sample frames and encode with CLIP
            import cv2
            from PIL import Image
            cap = cv2.VideoCapture(str(video_path))
            count = 0
            for _ in range(max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
                with torch.no_grad():
                    imf = clip_model.encode_image(img)
                    imf = imf / imf.norm(dim=-1, keepdim=True)
                    img_feats.append(imf)
                count += 1
            if count == 0:
                raise RuntimeError("No frames extracted from video for baseline CLIP run")
            img_feats = torch.cat(img_feats, dim=0)
            # similarity: image_feats x text_feats^T
            sims = (img_feats @ text_feats.T)
            probs = torch.softmax(sims, dim=1)
            # probability of the 'fake' prompt is column 1
            prob_fake = probs[:, 1].mean().item()
            print(f"(CLIP baseline) Video '{video_path}': Fakeness probability = {prob_fake:.3f}")
            return prob_fake
        finally:
            if cap is not None:
                cap.release()

    # Load checkpoint and run Wavelet-CLIP detector
    state = torch.load(str(weights_path), map_location=device)
    # Try to load state dict — repo may save differently (tweak as needed)
    try:
        model.load_state_dict(state)
    except Exception:
        # If the checkpoint contains a nested dict
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            raise

    model.eval()

    features = video_to_wavelet_features(video_path, preprocess, utils_mod.apply_wavelet_transform, max_frames=max_frames)

    with torch.no_grad():
        logits = model(features.cuda() if device == "cuda" else features)
        prob_fake = torch.softmax(logits, dim=1)[:, 1].mean().item()

    print(f"Video '{video_path}': Fakeness probability = {prob_fake:.3f}")
    return prob_fake


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("video", help="Path to video file")
    p.add_argument("--weights", help="Path to weights file (optional)")
    p.add_argument("--frames", type=int, default=16, help="Max frames to sample")
    args = p.parse_args()

    detect_fake_video(args.video, weights_path=args.weights, max_frames=args.frames)
