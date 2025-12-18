import streamlit as st
import tempfile
import os
from pathlib import Path

st.title("Wavelet-CLIP Deepfake Detector Demo")

st.write("Upload a short .mp4/.avi video and run the Wavelet-CLIP detector (requires the wavelet-clip repo and pretrained weights).")

uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.flush()
    st.video(tfile.name)

    st.write("Running detection — this may take a while depending on your GPU and model weights...")

    try:
        # Import local app wrapper
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app import detect_fake_video, check_weights

        weights_present = check_weights()
        if not weights_present:
            st.warning("Wavelet-CLIP weights not found at training/weights/clip_wavelet_best.pth. The app will run a CLIP-only baseline instead. Provide weights for the full detector.")

        run_baseline = st.checkbox("Run CLIP-only baseline if weights are missing", value=True)

        if not weights_present and not run_baseline:
            st.error("No weights available and baseline disabled. Upload weights to training/weights/clip_wavelet_best.pth or enable baseline.")
        else:
            prob = detect_fake_video(tfile.name)
            st.success(f"Fakeness probability: {prob:.3f}")
    except Exception as e:
        st.error(f"Failed to run detector: {e}")
        st.write("Make sure the official Wavelet-CLIP repo is cloned into ./wavelet-clip and weights placed at training/weights/clip_wavelet_best.pth")
    finally:
        try:
            os.unlink(tfile.name)
        except Exception:
            pass
