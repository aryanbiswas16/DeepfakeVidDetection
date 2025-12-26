import streamlit as st
import torch
import tempfile
import os
import sys
from io import BytesIO
from PIL import Image

# --- 1. Setup & Configuration ---
# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.detector import Detector
from utils.video_io import read_video_frames
from utils.preprocess import stack_frames
from utils.weights import load_weights
from utils.identity import IdentityMatcher
from utils.face_crop import FaceCropper

st.set_page_config(page_title="Deepfake Inspector v3", layout="wide")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Resource Loading ---
@st.cache_resource
def load_detector():
    """Loads the Deepfake Detector model and weights."""
    det = Detector(device=DEVICE)
    weight_path = os.path.join(PROJECT_ROOT, "training", "weights", "dinov3_best_v3.pth")
    loaded = load_weights(det, weight_path)
    return det, loaded

@st.cache_resource
def load_cropper():
    """Loads the Face Cropper with training parameters."""
    # Matches scripts/process_data.py: margin_px=0.5, padding_ratio=0.3
    return FaceCropper(device=DEVICE, margin_px=0.5, padding_ratio=0.3)

@st.cache_resource
def load_identity_matcher():
    """Loads the Identity Matcher."""
    return IdentityMatcher(device=DEVICE)

# --- 3. Helper Functions ---
def simulate_compression(frames, quality=90):
    """
    Simulates JPG compression artifacts to match training data distribution.
    """
    processed_frames = []
    for frame in frames:
        buffer = BytesIO()
        frame.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        processed_frames.append(compressed)
    return processed_frames

# --- 4. Main Application ---
def main():
    st.title("🕵️ DINOv3 AI Video Inspector (v3)")
    
    # --- Sidebar Settings ---
    st.sidebar.header("Settings")
    
    # Load Models
    detector, weights_loaded = load_detector()
    cropper = load_cropper()
    
    if not weights_loaded:
        st.sidebar.warning("⚠️ Pretrained weights not found. Running in untrained mode.")
    else:
        st.sidebar.success("✅ Pretrained weights loaded.")
        
    threshold = st.sidebar.slider(
        "Detection Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.70, 
        step=0.05, 
        help="Adjust sensitivity. Higher values reduce false positives."
    )

    # --- Main Layout ---
    col1, col2 = st.columns(2)

    # --- Column 1: Deepfake Detection ---
    with col1:
        st.subheader("1. Analyze Video")
        video_file = st.file_uploader("Upload MP4", type=["mp4", "mov"])
        
        # Initialize frames variable to ensure scope visibility
        frames = []
        
        if video_file:
            # Save uploaded file temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            tfile.close()
            
            # Process Video
            frames = read_video_frames(tfile.name, num_frames=8)
            st.caption(f"Extracted {len(frames)} frames from video.")
            
            # Crop Faces
            cropped_frames = []
            for f in frames:
                # return_metadata=False returns just the PIL Image or None
                c = cropper.crop(f, return_metadata=False)
                if c is not None: 
                    cropped_frames.append(c)
            
            if not cropped_frames:
                st.error(f"No faces detected in the {len(frames)} extracted frames.")
            else:
                st.info(f"Analyzed {len(cropped_frames)}/{len(frames)} frames with detected faces.")
                
                # Display Faces
                num_images = len(cropped_frames)
                num_cols = min(4, num_images)
                cols = st.columns(num_cols)
                
                for i in range(num_images):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        st.image(cropped_frames[i], caption=f"Frame {i+1}", use_container_width=True)
                
                # Run Detection
                if st.button("Detect Deepfake"):
                    with st.spinner("Analyzing frames..."):
                        # 1. Simulate Compression (Fix for distribution shift)
                        processed_frames = simulate_compression(cropped_frames)
                        
                        # 2. Preprocess & Stack
                        # Use detector's internal preprocess to ensure consistency
                        input_tensor = stack_frames(processed_frames, detector.preprocess, DEVICE)
                        
                        # 3. Inference
                        result = detector.predict_video(input_tensor)
                        prob = result['score']
                        
                        # 4. Display Results
                        st.write("---")
                        if prob > threshold:
                            st.error(f"🚨 **FAKE DETECTED**")
                            st.metric("Fake Probability", f"{prob*100:.2f}%", delta=f"Threshold: {threshold*100:.0f}%")
                        else:
                            st.success(f"✅ **REAL VIDEO**")
                            st.metric("Fake Probability", f"{prob*100:.2f}%", delta=f"Threshold: {threshold*100:.0f}%", delta_color="inverse")
                        
                        st.progress(prob)
                        
                        with st.expander("See per-frame details"):
                            st.json(result['details'])
            
            # Cleanup
            os.unlink(tfile.name)

    # --- Column 2: Identity Verification ---
    with col2:
        st.subheader("2. Identity Verification (Optional)")
        ref_file = st.file_uploader("Upload Reference Photo", type=["jpg", "png"]) 
        
        if ref_file and video_file:
            ref_img = Image.open(ref_file).convert("RGB")
            st.image(ref_img, width=150, caption="Reference ID")
            
            if st.button("Verify Identity"):
                matcher = load_identity_matcher()
                
                # Get embeddings
                ref_emb = matcher.get_embedding(ref_img)
                
                # Use the middle frame from the video for comparison
                if len(frames) > 0:
                    vid_emb = matcher.get_embedding(frames[len(frames)//2])
                    
                    if ref_emb is None or vid_emb is None:
                        st.warning("Identity model not available (missing facenet-pytorch) or face not detected.")
                    else:
                        sim = matcher.compute_similarity(ref_emb, vid_emb)
                        st.metric("Similarity Score", f"{sim:.2f}")
                        
                        if sim > 0.7:
                            st.success("Identity Matches")
                        else:
                            st.error("Identity Mismatch")
                else:
                    st.warning("Please upload a video and ensure frames are extracted first.")

if __name__ == "__main__":
    main()
