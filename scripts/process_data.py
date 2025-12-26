import os
import sys
import cv2
import numpy as np
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add project root to python path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.face_crop import FaceCropper

def extract_frames_from_folder(video_folder, output_root, frames_per_video=10, crop_faces=True, cropper=None):
    """
    Extracts frames from all videos in video_folder to output_root/video_name/frame_x.jpg
    """
    video_folder = Path(video_folder)
    output_root = Path(output_root)
    
    videos = sorted([f for f in video_folder.iterdir() if f.suffix.lower() in ['.mp4', '.avi', '.mov']])
    
    print(f"Processing {len(videos)} videos from {video_folder}...")
    
    for vid in tqdm(videos):
        out_dir = output_root / vid.stem
        if out_dir.exists() and any(out_dir.iterdir()):
            continue # Skip if already extracted
            
        out_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(vid))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            continue
            
        if total_frames < frames_per_video:
            indices = list(range(total_frames))
        else:
            step = total_frames // frames_per_video
            indices = [i * step for i in range(frames_per_video)]
            
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                save_img = frame
                if crop_faces and cropper is not None:
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        crop, meta = cropper.crop(pil_img, return_metadata=True)
                        if crop is not None:
                            save_img = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
                    except Exception:
                        pass # Keep original frame if cropping fails
                
                cv2.imwrite(str(out_dir / f"frame_{idx:04d}.jpg"), save_img)
        cap.release()

def main():
    # Initialize cropper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    cropper = FaceCropper(device=device, margin_px=0.5, padding_ratio=0.3)

    # 1. Process FaceForensics++ (Training)
    # Real
    ff_real_src = "data/train/real"
    ff_real_dst = "data/processed_ff/original_sequences/youtube/c23/frames"
    if os.path.exists(ff_real_src):
        extract_frames_from_folder(ff_real_src, ff_real_dst, crop_faces=True, cropper=cropper)
    
    # Fake (All methods in data/train/fake)
    ff_fake_root = Path("data/train/fake")
    if ff_fake_root.exists():
        for method_dir in ff_fake_root.iterdir():
            if method_dir.is_dir():
                method_name = method_dir.name
                # FF++ structure usually expects manipulated_sequences/{method}/c23/frames
                ff_fake_dst = Path(f"data/processed_ff/manipulated_sequences/{method_name}/c23/frames")
                print(f"Processing FF++ method: {method_name}")
                extract_frames_from_folder(method_dir, ff_fake_dst, crop_faces=True, cropper=cropper)

    # 2. Process Celeb-DF (Validation)
    # Real
    celeb_real_src = "data/val/real"
    celeb_real_dst = "data/processed_celeb/Celeb-real/frames"
    if os.path.exists(celeb_real_src):
        extract_frames_from_folder(celeb_real_src, celeb_real_dst, crop_faces=True, cropper=cropper)
        
    # Fake
    celeb_fake_src = "data/val/fake"
    celeb_fake_dst = "data/processed_celeb/Celeb-synthesis/frames"
    if os.path.exists(celeb_fake_src):
        extract_frames_from_folder(celeb_fake_src, celeb_fake_dst, crop_faces=True, cropper=cropper)

if __name__ == "__main__":
    main()