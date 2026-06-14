# 🔥 Viral Moment Detector Pro

AI-powered video intelligence that automatically discovers, scores, and extracts high-engagement moments from long-form content.

Designed for creators, editors, agencies, and content automation workflows.

---

## 🎯 What It Does

Viral Moment Detector Pro analyzes video content using multiple AI detectors and engagement signals to identify moments most likely to capture audience attention.

The system assigns every detected moment a score and automatically categorizes clips into:

| Tier      | Meaning                         |
| --------- | ------------------------------- |
| 🏆 A Tier | Highly viral potential          |
| ⭐ B Tier  | Strong engagement potential     |
| 📌 C Tier | Notable moments worth reviewing |

The final result is a collection of ranked highlight clips with detailed metadata explaining exactly why each clip was selected.

---

# 🧠 Detection Pipeline

The system combines multiple independent detectors:

### Audio Analysis

* Audio energy spikes
* Loudness changes
* Excitement detection

### Motion Analysis

* Rapid movement detection
* Action intensity scoring
* Optical flow analysis

### Scene Analysis

* Scene transition detection
* Visual change tracking

### Emotion Detection

* Face emotion recognition
* Excitement indicators
* Reaction moments

### Object Detection

* Person detection
* Object presence tracking
* Context awareness

### Speech Intelligence

* Joke detection
* Emotional statements
* Arguments and conflict
* Surprising statements
* Viral trigger phrases

### Advanced AI Analysis (Full Mode)

* CLIP visual understanding
* Narrative understanding
* Context-aware scoring using Phi-3

---

# ⚡ Analysis Modes

## Fast Mode

Optimized for speed.

Includes:

* Audio Energy
* Motion Detection
* Scene Change Detection

Best for:

* Quick previews
* Large batches
* Low-end hardware

---

## Balanced Mode

Recommended mode.

Includes:

* Everything from Fast Mode
* Face Emotion Detection
* Object Detection
* Speech Analysis

Best for:

* Most creators
* Daily use
* High-quality results

---

## Full Mode

Maximum analysis quality.

Includes:

* Everything from Balanced Mode
* CLIP Video Understanding
* Narrative Analysis (Phi-3)

Best for:

* Production workflows
* Research
* Maximum detection accuracy

---

# ✂️ Smart Clip Generation

Detected moments are automatically refined using:

* Sentence-aware clipping
* Dynamic gap merging
* Context preservation
* Boundary optimization

This prevents awkward cuts and creates clips that feel natural when viewed independently.

---

# 📊 Output

Each generated clip includes:

```json
{
  "tier": "A",
  "score": 94.3,
  "start_time": 125.2,
  "end_time": 148.9,
  "detectors": {
    "audio": 0.92,
    "speech": 0.87,
    "emotion": 0.90
  }
}
```

Generated outputs:

* Ranked highlight clips
* Viral score reports
* JSON metadata
* Detector contribution breakdowns

---

# 🖥️ Dashboard

Streamlit dashboard included.

Features:

* Video upload
* Mode selection
* Real-time progress tracking
* Clip preview
* Score visualization
* Metadata inspection

Launch:

```bash
streamlit run app.py
```

---

# 💻 Command Line Usage

Analyze a video once:

```bash
python app.py --once video.mp4
```

Batch workflows and automation are supported through CLI execution.

---

# 🔔 Notifications

Automatically send high-confidence clips to:

* Discord
* Slack

Tier A clips can trigger webhook alerts immediately after processing.

---

# 📁 Project Structure

```text
viral-moment-detector/
│
├── viral_detector/
│   ├── detectors/
│   ├── analyzers/
│   ├── scoring/
│   └── clipping/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Requirements

Minimum:

* Python 3.10+
* FFmpeg
* 8 GB RAM

Recommended:

* 16 GB RAM
* NVIDIA GPU
* SSD Storage

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/xdiimario-pixel/viral-moment-detector.git
cd viral-moment-detector
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🛣️ Planned Features

* [ ] Multi-video batch processing
* [ ] Thumbnail generation
* [ ] Viral title generation
* [ ] YouTube integration
* [ ] TikTok optimization profile
* [ ] Multi-GPU support
* [ ] Cloud deployment

---

# 📄 License

MIT License

---

Built with Python, Streamlit, Whisper, CLIP, YOLO, DeepFace, and custom engagement-scoring algorithms.
