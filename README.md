# DeepfakeDetector v3

A state-of-the-art deepfake detection system using **DINOv3** (Vision Transformer) as a feature extractor and a custom linear classifier. This project leverages transfer learning to achieve high accuracy with minimal training data.

## Features

- **Advanced Architecture**: Uses Meta's DINOv3 (ViT-B/16) for robust frame encoding.
- **Face-Focused**: Automatically detects and crops faces using MTCNN.
- **Robustness**: Trained with augmentation (compression, blur) to handle real-world video artifacts.
- **Interactive App**: Includes a Streamlit web interface for easy testing.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/DeepfakeDetector_v3.git
    cd DeepfakeDetector_v3
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Run the Web App
The easiest way to use the detector is via the Streamlit app.

```bash
python -m streamlit run apps/streamlit_app.py
```
Upload a video file (`.mp4`, `.mov`) to analyze it.

### 2. Training (Optional)
If you want to retrain the model on your own data:

1.  **Prepare Data**: Place your videos in `data/train/real` and `data/train/fake`.
2.  **Process Data**: Extract frames and crop faces.
    ```bash
    python scripts/process_data.py
    ```
3.  **Train**:
    ```bash
    python training/train.py
    ```
    Weights will be saved to `training/weights/`.

## Project Structure

- `models/`: Contains the PyTorch model definitions (`Detector`, `FrameEncoder`).
- `training/`: Training scripts and saved weights.
- `utils/`: Helper functions for preprocessing, face cropping, and I/O.
- `apps/`: Streamlit application.
- `scripts/`: Data processing scripts.

## License

[MIT License](LICENSE)
