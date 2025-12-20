import streamlit as st
import torch
import tempfile
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
from models.detector import Detector
from utils.video_io import read_video_frames
from utils.preprocess import build_preprocess, stack_frames
from utils.weights import load_weights
from utils.identity import IdentityMatcher

st.set_page_config(page_title="Deepfake Inspector v3", layout="wide")
st.title("🕵️ DINOv3 AI Video Inspector (v3)")
st.sidebar.header("Settings")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_detector():
    det = Detector(device=device)
    # Try load default weights
    weight_path = os.path.join(PROJECT_ROOT, "training", "weights", "dinov2_best.pth")
    # Only load into the full detector object, as the weights file contains {'head': ..., 'encoder': ...}
    loaded = load_weights(det, weight_path)
    return det, loaded

detector, weights_loaded = load_detector()
if not weights_loaded:
    st.sidebar.warning("⚠️ Pretrained weights not found. Running in untrained mode.")
else:
    st.sidebar.success("✅ Pretrained weights loaded.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Analyze Video")
    video_file = st.file_uploader("Upload MP4", type=["mp4", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        frames = read_video_frames(tfile.name, num_frames=8)
        st.image(frames[0], caption="First Frame", width=300)
        if st.button("Detect Deepfake"):
            with st.spinner("Analyzing frames..."):
                preprocess = detector.preprocess
                input_tensor = stack_frames(frames, preprocess, device)
                result = detector.predict_video(input_tensor)
                prob = result['score']
                if prob > 0.5:
                    st.error(f"🚨 Result: FAKE ({prob*100:.1f}%)")
                else:
                    st.success(f"✅ Result: REAL ({(1-prob)*100:.1f}%)")
                st.progress(prob)

with col2:
    st.subheader("2. Identity Verification (Optional)")
    ref_file = st.file_uploader("Upload Reference Photo", type=["jpg", "png"]) 
    if ref_file and video_file:
        ref_img = Image.open(ref_file).convert("RGB")
        st.image(ref_img, width=150, caption="Reference ID")
        if st.button("Verify Identity"):
            matcher = IdentityMatcher(device)
            ref_emb = matcher.get_embedding(ref_img)
            vid_emb = matcher.get_embedding(frames[len(frames)//2])
            if ref_emb is None or vid_emb is None:
                st.warning("Identity model not available (missing facenet-pytorch).")
            else:
                sim = matcher.compute_similarity(ref_emb, vid_emb)
                st.metric("Similarity Score", f"{sim:.2f}")
                if sim > 0.7:
                    st.success("Identity Matches")
                else:
                    st.error("Identity Mismatch")
