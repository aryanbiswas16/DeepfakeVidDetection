"""
Script: extract_paired_frames.py
Extract uniformly sampled frames from paired real/fake video files.
Usage:
    python scripts/extract_paired_frames.py --real_dir real_videos --fake_dir fake_videos --out_dir data/train --num_frames 16

This will look for matching basenames in real_dir and fake_dir and write frames to
out_dir/real and out_dir/fake with matching filenames.
"""
import os
import cv2
import argparse


def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # fallback: read sequentially until done
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames[:num_frames]

    import numpy as np
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)
    cap.release()
    return frames


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', required=True)
    parser.add_argument('--fake_dir', required=True)
    parser.add_argument('--out_dir', default='data/train')
    parser.add_argument('--num_frames', type=int, default=16)
    args = parser.parse_args()

    real_files = {os.path.splitext(f)[0]: os.path.join(args.real_dir, f) for f in os.listdir(args.real_dir) if f.lower().endswith(('.mp4','.mov','.avi'))}
    fake_files = {os.path.splitext(f)[0]: os.path.join(args.fake_dir, f) for f in os.listdir(args.fake_dir) if f.lower().endswith(('.mp4','.mov','.avi'))}

    common = set(real_files.keys()).intersection(set(fake_files.keys()))
    if not common:
        print('No matching basenames found between real and fake directories.')
        return

    out_real = os.path.join(args.out_dir, 'real')
    out_fake = os.path.join(args.out_dir, 'fake')
    ensure_dir(out_real)
    ensure_dir(out_fake)

    for base in sorted(common):
        rpath = real_files[base]
        fpath = fake_files[base]
        print(f'Processing pair: {base}')
        r_frames = extract_frames(rpath, args.num_frames)
        f_frames = extract_frames(fpath, args.num_frames)
        n = min(len(r_frames), len(f_frames))
        for i in range(n):
            rname = f"{base}_frame_{i:04d}.jpg"
            fname = f"{base}_frame_{i:04d}.jpg"
            cv2.imwrite(os.path.join(out_real, rname), r_frames[i])
            cv2.imwrite(os.path.join(out_fake, fname), f_frames[i])

    print('Done.')

if __name__ == '__main__':
    main()
