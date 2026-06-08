# 🔥 Viral Moment Detector Pro

AI‑powered video analysis that detects and ranks viral moments into A, B, C tiers based on multiple engagement metrics (audio energy, motion, scene changes, face emotion, object detection, speech analysis, etc.).

## Features

- **Fast mode**: audio energy, motion intensity, scene changes – quick analysis.
- **Balanced mode**: adds face emotion, object detection, and speech‑based detectors (jokes, arguments, emotional phrases, shocking statements).
- **Full mode**: adds CLIP video understanding and narrative analysis (Phi‑3) – requires high‑end hardware.
- **Streamlit dashboard** for easy video selection and result preview.
- **CLI mode** for batch processing (see `--once` flag).
- **Automatic clip cutting** with smart boundary refinement (sentence‑aware, dynamic gap merging).
- **JSON metadata** for each clip (scores, start/end times, detector contributions).
- **Webhook support** (Discord/Slack) for Tier A alerts.

## Requirements

- Python 3.10 or higher
- FFmpeg (must be in PATH or provided via config)
- Recommended: 8GB+ RAM, GPU optional (for faster processing in full mode)

## Installation

1. Clone or download the project.
2. Create a virtual environment:
   ```bash
   python -m venv venv