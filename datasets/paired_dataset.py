import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class PairedDataset(Dataset):
    """
    Load paired real/fake frames.
    
    Directory structure:
        data/train/real/video_001_frame_0000.jpg
        data/train/fake/video_001_frame_0000.jpg  (same filename!)
    """
    
    def __init__(self, real_dir, fake_dir, transform=None):
        super().__init__()
        
        self.transform = transform
        
        # Get all files from real directory
        self.real_files = sorted([f for f in os.listdir(real_dir) 
                                  if f.endswith(('.jpg', '.png'))])
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        
        # Verify all real files have corresponding fake files
        missing = []
        for fname in self.real_files:
            if not os.path.exists(os.path.join(fake_dir, fname)):
                missing.append(fname)
        
        if missing:
            print(f"Warning: {len(missing)} real files have no fake counterpart")
            self.real_files = [f for f in self.real_files if f not in missing]
        
        print(f"Loaded {len(self.real_files)} paired real/fake frames")
    
    def __len__(self):
        return len(self.real_files)
    
    def __getitem__(self, idx):
        fname = self.real_files[idx]
        
        # Load real frame
        real_path = os.path.join(self.real_dir, fname)
        real_img = Image.open(real_path).convert('RGB')
        
        # Load fake frame (same filename)
        fake_path = os.path.join(self.fake_dir, fname)
        fake_img = Image.open(fake_path).convert('RGB')
        
        # Apply preprocessing
        if self.transform:
            real_img = self.transform(real_img)
            fake_img = self.transform(fake_img)
        
        return real_img, fake_img, fname
