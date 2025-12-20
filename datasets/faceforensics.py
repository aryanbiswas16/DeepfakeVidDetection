import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class FaceForensicsDataset(Dataset):
    """
    FaceForensics++ Dataset.
    """
    
    def __init__(self, root_dir, method="Deepfakes", compression="c23", transform=None, split="train"):
        super().__init__()
        self.root = Path(root_dir)
        self.transform = transform
        
        # Paths to FRAMES
        self.real_dir = self.root / "original_sequences" / "youtube" / compression / "frames"
        self.fake_dir = self.root / "manipulated_sequences" / method / compression / "frames"
        
        self.pairs = []
        
        if not self.real_dir.exists() or not self.fake_dir.exists():
            return

        # Get all fake video folders (e.g. 000_003)
        fake_videos = sorted([d.name for d in self.fake_dir.iterdir() if d.is_dir()])
        
        for vid in fake_videos:
            # vid is like 000_003
            parts = vid.split('_')
            target_id = parts[0] # 000 is the target (background)
            
            # Train/Val split based on ID
            try:
                vid_id = int(target_id)
            except:
                continue
                
            # FF++ split: 0-719 train, 720-859 val, 860-999 test
            if split == "all":
                pass # Use everything
            elif split == "train" and vid_id >= 720:
                continue
            elif split == "val" and (vid_id < 720 or vid_id >= 860):
                continue
            elif split == "test" and vid_id < 860:
                continue
                
            real_vid_path = self.real_dir / target_id
            fake_vid_path = self.fake_dir / vid
            
            if not real_vid_path.exists():
                continue
                
            # Get frames
            fake_frames = sorted(list(fake_vid_path.glob("*.jpg")) + list(fake_vid_path.glob("*.png")))
            
            for ff in fake_frames:
                frame_name = ff.name
                rf = real_vid_path / frame_name
                
                if rf.exists():
                    self.pairs.append((str(rf), str(ff)))
                    
        print(f"FaceForensics++ ({split}): Loaded {len(self.pairs)} paired frames.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        try:
            real_path, fake_path = self.pairs[idx]
            
            real_img = Image.open(real_path).convert('RGB')
            fake_img = Image.open(fake_path).convert('RGB')
            
            if self.transform:
                real_img = self.transform(real_img)
                fake_img = self.transform(fake_img)
                
            return real_img, fake_img, "ff++"
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}. Skipping...")
            # Pick a random index to replace the bad sample
            import random
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
