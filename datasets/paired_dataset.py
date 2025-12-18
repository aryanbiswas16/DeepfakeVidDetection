import os
from torch.utils.data import Dataset
from PIL import Image

class PairedDataset(Dataset):
    """Dataset that returns paired real/fake images originating from same source.

    Expects two directories: real_dir and fake_dir. Filenames should share a common
    basename so we can pair them (e.g. "video001_frame000.jpg" present in both).
    Returns: (real_img, fake_img, basename)
    """
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform

        real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        fake_files = set([f for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])

        # Keep only pairs that exist in both
        self.pairs = []
        for f in real_files:
            if f in fake_files:
                self.pairs.append(f)

        if len(self.pairs) == 0:
            raise RuntimeError(f"No paired files found between {real_dir} and {fake_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fname = self.pairs[idx]
        real_path = os.path.join(self.real_dir, fname)
        fake_path = os.path.join(self.fake_dir, fname)
        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')
        if self.transform:
            real_img = self.transform(real_img)
            fake_img = self.transform(fake_img)
        return real_img, fake_img, fname
