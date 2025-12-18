This folder is for your training data.

Structure:
data/
  train/
    real/  <- Put extracted REAL frames here
    fake/  <- Put extracted FAKE frames here
  val/
    real/  <- (Optional) Validation REAL frames
    fake/  <- (Optional) Validation FAKE frames

IMPORTANT:
Filenames must match between real and fake folders for the PairedDataset to work.
Example:
  data/train/real/video1_frame0.jpg
  data/train/fake/video1_frame0.jpg

Use scripts/extract_paired_frames.py to generate these from video files.
