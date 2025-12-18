"""Smoke test to verify Detector forward pass with random inputs."""
import torch
from PIL import Image

from models.detector import Detector
from utils.preprocess import stack_frames


def make_dummy_frames(num=4, size=(224,224)):
    imgs = []
    for i in range(num):
        # make a gray PIL image
        img = Image.new('RGB', size, color=(int(127+i)%255, 127, 127))
        imgs.append(img)
    return imgs


def run():
    device = 'cpu'
    det = Detector(device=device)
    preprocess = det.preprocess
    frames = make_dummy_frames(8, (det.encoder.image_size, det.encoder.image_size))
    batch = stack_frames(frames, preprocess, device)
    result = det.predict_video(batch)
    print('Smoke test result:', result)

if __name__ == '__main__':
    run()
