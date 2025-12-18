#!/usr/bin/env bash
# Helper script placeholder to download weights if a public URL is provided.
# Usage: ./scripts/download_weights.sh <weights_url>

if [ -z "$1" ]; then
  echo "Usage: $0 <weights_url>"
  echo "This script will download the file and place it into training/weights/ as clip_wavelet_best.pth"
  exit 1
fi

mkdir -p training/weights
curl -L "$1" -o training/weights/clip_wavelet_best.pth
if [ $? -ne 0 ]; then
  echo "Download failed. Please download the weights manually and place them in training/weights/"
  exit 2
fi

echo "Weights downloaded to training/weights/clip_wavelet_best.pth"