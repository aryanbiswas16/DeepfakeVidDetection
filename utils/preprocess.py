from __future__ import annotations

import torch
from torchvision import transforms

# Minimal preprocess function; tuned later if DINOv2 provides a specific pipeline
def build_preprocess(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# Backwards-compatible aliases (in case older code uses different names)
buildpreprocess = build_preprocess


def stack_frames(frames, preprocess_fn, device: str):
    tensor_list = [preprocess_fn(f) for f in frames]
    return torch.stack(tensor_list).to(device)


stackframes = stack_frames
