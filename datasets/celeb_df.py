import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CelebDFDataset(Dataset):
    """
    Celeb-DF v2 Dataset.
    """
    
    def __init__(self, root_dir, transform=None, split="train"):
        super().__init__()
        self.root = Path(root_dir)
        self.transform = transform
        
        self.real_dir = self.root / "Celeb-real" / "frames"
        self.fake_dir = self.root / "Celeb-synthesis" / "frames"
        
        self.pairs = []
        
        if not self.real_dir.exists() or not self.fake_dir.exists():
            return
            
        fake_videos = sorted([d.name for d in self.fake_dir.iterdir() if d.is_dir()])
        
        for vid in fake_videos:
            # vid: id0_id1_0000
            parts = vid.split('_')
            if len(parts) < 3: continue
            
            target_id = parts[0] # id0
            video_idx = parts[2] # 0000
            real_vid_name = f"{target_id}_{video_idx}"
            
            # Simple split hash
            h = hash(real_vid_name) % 10
            if split == "train" and h >= 2: # 80% train (using 2 as cutoff for val to be small? No, usually 8 for train)
                 # Let's use user's request: ALL Celeb-DF is VAL.
                 # So if split is train, return empty?
                 # The user said "DFC v2 in the validation".
                 pass
            
            # Actually, the user wants to use this dataset ONLY for validation.
            # So if split="train", we can just skip everything if we want to enforce that.
            # But usually the class should support both.
            # I'll stick to standard split logic, but the user will only instantiate it with split='val'.
            
            real_vid_path = self.real_dir / real_vid_name
            fake_vid_path = self.fake_dir / vid
            
            if not real_vid_path.exists():
                continue
            
            fake_frames = sorted(list(fake_vid_path.glob("*.jpg")) + list(fake_vid_path.glob("*.png")))
            
            for ff in fake_frames:
                frame_name = ff.name
                rf = real_vid_path / frame_name
                
                if rf.exists():
                    self.pairs.append((str(rf), str(ff)))
                    
        print(f"Celeb-DF ({split}): Loaded {len(self.pairs)} paired frames.")

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
                
            return real_img, fake_img, "celeb-df"
        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}. Skipping...")
            # Pick a random index to replace the bad sample
            import random
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
