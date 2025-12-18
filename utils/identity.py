from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

try:
    from facenet_pytorch import InceptionResnetV1  # type: ignore
except Exception:  # facenet-pytorch is optional
    InceptionResnetV1 = None  # type: ignore

class IdentityMatcher:
    def __init__(self, device="cuda"):
        self.device = device
        # Load pretrained FaceNet
        if InceptionResnetV1 is None:
            self.model = None
        else:
            try:
                self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
            except Exception:
                # Fallback: weights/model not available
                self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def get_embedding(self, image):
        """Returns (1, 512) embedding or None if model not available."""
        if self.model is None:
            return None
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.cpu().numpy()

    def compute_similarity(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return None
        # Cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2.T)[0][0] / (norm1 * norm2)
