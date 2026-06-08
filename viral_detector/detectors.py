# viral_detector/detectors.py
import os
import re
import time
import json
import threading
import math
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from .config import DetectionConfig, DetectionMethod
from .utils import DetectorBase, LoggerFactory, VideoMetrics, _round_ts
from .models import DetectionResult
from sentence_transformers import CrossEncoder


def compute_semantic_scores(segments: List[Dict]) -> Dict[str, List[float]]:
    """Return semantic scores for argument, emotion, shocking."""
    if not segments:
        return {}
    texts = [seg['text'] for seg in segments]
    model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
    label_sets = {
        "argument_detection": ["argument", "disagreement", "debate"],
        "emotional_phrase": ["emotional", "happy", "sad", "anger", "love", "fear"],
        "shocking_statement": ["shocking", "surprising", "unbelievable"]
    }
    results = {name: [] for name in label_sets}
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        for det, labels in label_sets.items():
            preds = model.predict(batch, labels)       # shape (batch, len(labels))
            scores = [max(p) * 100 for p in preds]     # convert to 0-100 scale
            results[det].extend(scores)
    return results
# ==================== AUDIO ENERGY DETECTOR ====================
class AudioEnergyDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.AUDIO_ENERGY)

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        try:
            import librosa
        except ImportError:
            self.logger.warning("librosa not installed")
            return []
        results = []
        try:
            y, sr = self._get_audio(video_path)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            energy_score = np.clip(rms * 200, 0.0, 100.0)
            hop_length = 512
            times = librosa.frames_to_time(np.arange(len(energy_score)), sr=sr, hop_length=hop_length)
            threshold = self.config.audio_energy_threshold
            current_bin = None
            peak_score = 0.0
            peak_time = 0.0
            for i, (t, score) in enumerate(zip(times, energy_score)):
                bin_start = int(t)
                if bin_start != current_bin:
                    if current_bin is not None and peak_score > threshold:
                        results.append(DetectionResult(self.method, peak_time, float(peak_score)))
                    current_bin = bin_start
                    peak_score = 0.0
                    peak_time = t
                if score > peak_score:
                    peak_score = score
                    peak_time = t
            if current_bin is not None and peak_score > threshold:
                results.append(DetectionResult(self.method, peak_time, float(peak_score)))
            self.logger.debug(f"Audio: Found {len(results)} high-energy moments")
        except Exception as e:
            self.logger.error(f"Audio analysis failed: {e}")
        return self._apply_cooldown(results)

# ==================== MOTION DETECTOR ====================
class MotionDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.MOTION)

    def _compute_percentile_scores(self, motion_arr: np.ndarray, percentile: float = 95.0) -> np.ndarray:
        """
        Convert raw motion values to 0-100 scores using a percentile as the maximum.
        Values above the percentile are clipped to 100.
        """
        if len(motion_arr) == 0:
            return np.array([])
        max_val = np.percentile(motion_arr, percentile)
        if max_val <= 0:
            return np.zeros_like(motion_arr)
        scores = (motion_arr / max_val) * 100.0
        return np.clip(scores, 0.0, 100.0)

    def analyze(self, video_path: Path, frame_queue=None) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        if frame_queue is None:
            # Original code: open video and read frames
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    self.logger.error(f"Could not open video: {video_path}")
                    return []
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
            finally:
                if cap:
                    cap.release()
            if self.config.processing_mode == "fast":
                return self._fast_motion(video_path, fps)
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
            target_width = 480
            prev_gray = None
            frame_idx = 0
            sample_rate = max(10, int(fps // 2))
            self.logger.debug(f"Motion detection: sampling every {sample_rate} frames, target width {target_width}")
            motion_scores = []
            timestamps = []
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        h, w = frame.shape[:2]
                        scale = target_width / w
                        new_h = int(h * scale)
                        resized = cv2.resize(frame, (target_width, new_h))
                        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        if prev_gray is not None:
                            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore
                            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                            avg_motion = np.mean(magnitude)
                            motion_scores.append(avg_motion)
                            timestamps.append(frame_idx / fps)
                        prev_gray = gray
                    frame_idx += 1
            finally:
                cap.release()
        else:
            # Use frame_queue from FrameProvider
            target_width = 480
            prev_gray = None
            motion_scores = []
            timestamps = []
            frame_count = 0
            # Get first item
            item = frame_queue.get()
            if item is None:
                print("[MotionDetector] Received sentinel immediately", flush=True)
                return []
            gray, idx, ts = item
            prev_gray = gray
            frame_count += 1
            while True:
                item = frame_queue.get()
                if item is None:
                    print(f"[MotionDetector] Received sentinel after {frame_count} frames", flush=True)
                    break
                gray, idx, ts = item
                frame_count += 1
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # type: ignore
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    avg_motion = np.mean(magnitude)
                    motion_scores.append(avg_motion)
                    timestamps.append(ts)
                prev_gray = gray

        if not motion_scores:
            return []
        motion_arr = np.array(motion_scores)
        results = []
        if self.config.use_fixed_motion_scaling:
            for ts, raw in zip(timestamps, motion_arr):
                score = min(raw * self.config.motion_fixed_scale, 100.0)
                if score > self.config.motion_threshold:
                    results.append(DetectionResult(self.method, ts, float(score)))
        else:
            norm_scores = self._compute_percentile_scores(motion_arr, percentile=95.0)
            for ts, score in zip(timestamps, norm_scores):
                if score > self.config.motion_threshold:
                    results.append(DetectionResult(self.method, ts, float(score)))
        self.logger.debug(f"Motion: Found {len(results)} high-motion moments")
        return self._apply_cooldown(results)

    def _fast_motion(self, video_path: Path, fps: float) -> List[DetectionResult]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        prev_gray = None
        frame_idx = 0
        sample_rate = max(5, int(fps // 4))
        motion_scores = []
        timestamps = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_rate == 0:
                    h, w = frame.shape[:2]
                    scale = 320 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(frame, (new_w, new_h))
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        motion = np.sum(thresh) / 255.0
                        motion_score = min((motion / (gray.shape[0] * gray.shape[1])) * 100, 100.0)
                        motion_scores.append(motion_score)
                        timestamps.append(frame_idx / fps)
                    prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()
        if not motion_scores:
            return []
        motion_arr = np.array(motion_scores)
        results = []
        if self.config.use_fixed_motion_scaling:
            for ts, raw in zip(timestamps, motion_arr):
                score = min(raw * self.config.motion_fixed_scale, 100.0)
                if score > self.config.motion_threshold:
                    results.append(DetectionResult(self.method, ts, float(score)))
        else:
            norm_scores = self._compute_percentile_scores(motion_arr, percentile=95.0)
            for ts, score in zip(timestamps, norm_scores):
                if score > self.config.motion_threshold:
                    results.append(DetectionResult(self.method, ts, float(score)))
        return self._apply_cooldown(results)

# ==================== SCENE CHANGE DETECTOR ====================
class SceneChangeDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.SCENE_CHANGES)

    def analyze(self, video_path: Path, frame_queue=None) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        results = []
        if frame_queue is None:
            # Original code: open video and read frames
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return results
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                sample_rate = max(5, int(fps // 3))
                prev_hist = None
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        if prev_hist is not None:
                            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                            change_score = (1 - correlation) * 100
                            if change_score > self.config.scene_change_threshold:
                                results.append(DetectionResult(self.method, frame_idx / fps, change_score))
                        prev_hist = hist
                    frame_idx += 1
            except Exception as e:
                self.logger.error(f"Scene change analysis failed: {e}")
            finally:
                if cap:
                    cap.release()
        else:
            # Use frame_queue from FrameProvider
            # The queue provides tuples: (gray_frame, frame_idx, timestamp)
            prev_hist = None
            frame_count = 0
            while True:
                item = frame_queue.get()
                if item is None:
                    print(f"[SceneChangeDetector] Received sentinel after {frame_count} frames", flush=True)
                    break
                gray, idx, ts = item
                frame_count += 1
                # Compute histogram
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                if prev_hist is not None:
                    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    change_score = (1 - correlation) * 100
                    if change_score > self.config.scene_change_threshold:
                        results.append(DetectionResult(self.method, ts, change_score))
                prev_hist = hist
        return self._apply_cooldown(results)

# ==================== SILENCE DETECTOR ====================
class SilenceDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.SILENCE_DETECTION)

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        try:
            import librosa
        except ImportError:
            return []
        results = []
        try:
            y, sr = self._get_audio(video_path)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            silence_threshold_db = -40.0
            hop_length = 2048
            times = librosa.frames_to_time(np.arange(len(rms_db)), sr=sr, hop_length=hop_length)
            for t, db in zip(times, rms_db):
                if db < silence_threshold_db:
                    silence_depth = silence_threshold_db - db
                    score = min((silence_depth / 30.0) * 100.0, 100.0)
                    if score > self.config.silence_threshold:
                        results.append(DetectionResult(self.method, float(t), score))
        except Exception as e:
            self.logger.error(f"Silence detection failed: {e}")
        return self._apply_cooldown(results)

# ==================== FACE EMOTION DETECTOR ====================
class FaceEmotionDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.FACE_EMOTION)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'  # type: ignore
        if not os.path.exists(cascade_path):
            self.logger.warning(f"Face cascade not found at {cascade_path}, face detection disabled")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def analyze(self, video_path: Path, frame_queue=None) -> List[DetectionResult]:
        if self.face_cascade is None:
            self.logger.warning("Face cascade not loaded, skipping face emotion detection")
            return []
        video_path = self._normalize_path(video_path)
        results = []
        if frame_queue is None:
            # Original code: open video and read frames
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return results
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                sample_rate = max(30, int(fps * 2))
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        h, w = frame.shape[:2]
                        if h > 480:
                            scale = 480 / h
                            new_w, new_h = int(w * scale), 480
                            frame_resized = cv2.resize(frame, (new_w, new_h))
                        else:
                            frame_resized = frame
                        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                        if len(faces) == 0:
                            frame_idx += 1
                            continue
                        try:
                            from deepface import DeepFace
                        except ImportError:
                            self.logger.warning("DeepFace not installed")
                            return results
                        emotion_analysis = DeepFace.analyze(frame_resized, actions=['emotion'], enforce_detection=False)
                        if emotion_analysis:
                            if isinstance(emotion_analysis, list):
                                data = emotion_analysis[0]
                            elif isinstance(emotion_analysis, dict):
                                data = emotion_analysis
                            else:
                                continue
                            if isinstance(data, dict):
                                dominant = data.get('dominant_emotion')
                                scores = data.get('emotion', {})
                                if dominant and scores:
                                    frame_score = self._get_emotion_score(dominant, scores)
                                    if frame_score > 25.0:
                                        results.append(DetectionResult(self.method, frame_idx / fps, frame_score, scores.get(dominant, 0.0)))
                    frame_idx += 1
            except Exception as e:
                self.logger.debug(f"DeepFace frame analysis failed: {e}")
            finally:
                if cap:
                    cap.release()
        else:
            # Use frame_queue from FrameProvider
            # The queue provides tuples: (rgb_frame_resized, frame_idx, timestamp)
            frame_count = 0
            while True:
                item = frame_queue.get()
                if item is None:
                    print(f"[FaceEmotionDetector] Received sentinel after {frame_count} frames", flush=True)
                    break
                frame_resized, idx, ts = item
                frame_count += 1
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if len(faces) == 0:
                    continue
                try:
                    from deepface import DeepFace
                except ImportError:
                    self.logger.warning("DeepFace not installed")
                    return results
                emotion_analysis = DeepFace.analyze(frame_resized, actions=['emotion'], enforce_detection=False)
                if emotion_analysis:
                    if isinstance(emotion_analysis, list):
                        data = emotion_analysis[0]
                    elif isinstance(emotion_analysis, dict):
                        data = emotion_analysis
                    else:
                        continue
                    if isinstance(data, dict):
                        dominant = data.get('dominant_emotion')
                        scores = data.get('emotion', {})
                        if dominant and scores:
                            frame_score = self._get_emotion_score(dominant, scores)
                            if frame_score > 25.0:
                                results.append(DetectionResult(self.method, ts, frame_score, scores.get(dominant, 0.0)))
        return self._apply_cooldown(results)

    def _get_emotion_score(self, dominant: str, scores: Dict[str, float]) -> float:
        weights = {'happy': 1.0, 'surprise': 0.9, 'sad': 0.2, 'angry': 0.6,
                   'fear': 0.5, 'disgust': 0.1, 'neutral': 0.1}
        weight = weights.get(dominant.lower(), 0.1)
        return scores.get(dominant.lower(), 0.0) * weight

# ==================== OBJECT DETECTOR ====================
class ObjectDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.OBJECT_DETECTION)
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolov8n.pt")
                if self.config.use_gpu and torch.cuda.is_available():
                    self._model.to('cuda')
                self.logger.info("YOLOv8 model loaded")
            except Exception as e:
                self.logger.error(f"YOLO model load failed: {e}")
                self._model = None
        return self._model

    def analyze(self, video_path: Path, frame_queue=None) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        model = self._load_model()
        if model is None:
            return []
        results = []
        if frame_queue is None:
            # Original code: open video and read frames
            cap = None
            try:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    return results
                fps = max(cap.get(cv2.CAP_PROP_FPS), 1.0)
                sample_rate = max(30, int(fps))
                max_frames = 50
                frames = []
                timestamps = []
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        h, w = frame.shape[:2]
                        scale = 640 / max(h, w)
                        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                        resized = cv2.resize(frame, (new_w, new_h))
                        frames.append(resized)
                        timestamps.append(frame_idx / fps)
                        if len(frames) >= max_frames:
                            batch_results = model(frames, verbose=False)
                            for i, res in enumerate(batch_results):
                                if res.boxes is not None:
                                    cls = res.boxes.cls.cpu().numpy()
                                    conf = res.boxes.conf.cpu().numpy()
                                    frame_score = self._get_frame_score(cls, conf)
                                    if frame_score > 25.0:
                                        results.append(DetectionResult(self.method, timestamps[i], frame_score, np.mean(conf) if len(conf)>0 else 0.0))
                            frames = []
                            timestamps = []
                    frame_idx += 1
                if frames:
                    batch_results = model(frames, verbose=False)
                    for i, res in enumerate(batch_results):
                        if res.boxes is not None:
                            cls = res.boxes.cls.cpu().numpy()
                            conf = res.boxes.conf.cpu().numpy()
                            frame_score = self._get_frame_score(cls, conf)
                            if frame_score > 25.0:
                                results.append(DetectionResult(self.method, timestamps[i], frame_score, np.mean(conf) if len(conf)>0 else 0.0))
            except Exception as e:
                self.logger.error(f"YOLO analysis failed: {e}")
            finally:
                if cap:
                    cap.release()
        else:
            # Use frame_queue from FrameProvider
            # The queue provides tuples: (resized_rgb_frame, frame_idx, timestamp)
            max_frames = 50
            frames = []
            timestamps = []
            frame_count = 0
            while True:
                item = frame_queue.get()
                if item is None:
                    print(f"[ObjectDetector] Received sentinel after {frame_count} frames", flush=True)
                    break
                frame_resized, idx, ts = item
                frame_count += 1
                frames.append(frame_resized)
                timestamps.append(ts)
                if len(frames) >= max_frames:
                    batch_results = model(frames, verbose=False)
                    for i, res in enumerate(batch_results):
                        if res.boxes is not None:
                            cls = res.boxes.cls.cpu().numpy()
                            conf = res.boxes.conf.cpu().numpy()
                            frame_score = self._get_frame_score(cls, conf)
                            if frame_score > 25.0:
                                results.append(DetectionResult(self.method, timestamps[i], frame_score, np.mean(conf) if len(conf)>0 else 0.0))
                    frames = []
                    timestamps = []
            if frames:
                batch_results = model(frames, verbose=False)
                for i, res in enumerate(batch_results):
                    if res.boxes is not None:
                        cls = res.boxes.cls.cpu().numpy()
                        conf = res.boxes.conf.cpu().numpy()
                        frame_score = self._get_frame_score(cls, conf)
                        if frame_score > 25.0:
                            results.append(DetectionResult(self.method, timestamps[i], frame_score, np.mean(conf) if len(conf)>0 else 0.0))
        return self._apply_cooldown(results)

    def _get_frame_score(self, cls: np.ndarray, conf: np.ndarray) -> float:
        viral_weights = {0: 0.8, 15: 1.0, 16: 1.0, 17: 0.9, 2: 0.6,
                         3: 0.6, 67: 0.7, 46: 0.7, 47: 0.7, 32: 0.6}
        scores = [c_conf * viral_weights.get(int(c_id), 0.2) * 100 for c_id, c_conf in zip(cls, conf)]
        return max(scores) if scores else 0.0

# ==================== SPEECH DETECTOR (BASE) ====================
class SpeechDetector(DetectorBase):
    whisper_model = None  # kept for backward compatibility if needed, but not used in this refactor
    _model_lock = threading.Lock()
    transcription_cache = OrderedDict()
    _cache_lock = threading.Lock()
    _transcription_in_progress = {}
    _progress_lock = threading.Lock()

    # New: class-level cache for faster-whisper models
    _faster_whisper_models = {}
    _fw_lock = threading.Lock()

    def __init__(self, config: DetectionConfig, method: DetectionMethod):
        super().__init__(config, method)
        self.max_cache_size = 10

    def _get_faster_whisper_model(self):
        """Return a shared faster-whisper model instance (singleton per (model_name, device, compute_type))."""
        model_name = "tiny"   # force tiny model
        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        key = (model_name, device, compute_type) 

        with self._fw_lock:
            if key not in self._faster_whisper_models: 
                self.logger.info(f"Loading faster-whisper model {model_name} on {device} with {compute_type}")
                from faster_whisper import WhisperModel # type: ignore
                self._faster_whisper_models[key] = WhisperModel(model_name, device=device, compute_type=compute_type)
            else: 
                self.logger.debug(f"Reusing existing faster-whisper model for {model_name} on {device}")
            return self._faster_whisper_models[key]

    def _load_transcriber(self):
        # Legacy method kept for compatibility; not used when faster-whisper is active.
        # If you still need original whisper, keep it; otherwise you can deprecate.
        if SpeechDetector.whisper_model is None:
            with SpeechDetector._model_lock:
                if SpeechDetector.whisper_model is None:
                    try:
                        import whisper
                        model_name = self.config.whisper_model_name
                        SpeechDetector.whisper_model = whisper.load_model(model_name)
                        self.logger.info(f"Whisper {model_name} model loaded")
                    except Exception as e:
                        self.logger.error(f"Whisper load failed: {e}")
                        SpeechDetector.whisper_model = None
        return SpeechDetector.whisper_model

    def transcribe_audio(self, video_path: Path) -> Optional[List[Dict]]:
        import tempfile
        import soundfile as sf
        import math
        import threading
        import gc
        from pathlib import Path

        print("[DEBUG] transcribe_audio: started (chunked mode)", flush=True)

        video_path = self._normalize_path(video_path)
        resolved_path = video_path.resolve()
        cache_key = f"{resolved_path}_{resolved_path.stat().st_mtime}"
        video_str = str(resolved_path)  # Define early
        print("[DEBUG] transcribe_audio: cache_key", cache_key, flush=True)

        # Check per‑video cache
        with self._cache_lock:
            if cache_key in SpeechDetector.transcription_cache:
                self.logger.debug(f"Cache hit for {video_path.name}")
                print(f"[DEBUG] transcribe_audio: cache hit, returning {len(SpeechDetector.transcription_cache[cache_key])} segments", flush=True)
                return SpeechDetector.transcription_cache[cache_key]

        # Check if another thread is already transcribing this video (robust)
        is_in_progress = False
        event = None
        with self._progress_lock:
            if video_str in SpeechDetector._transcription_in_progress:
                event = SpeechDetector._transcription_in_progress[video_str]
                is_in_progress = True

        if is_in_progress and event is not None:
            self.logger.debug(f"Waiting for ongoing transcription of {video_path.name}")
            print("[DEBUG] transcribe_audio: waiting for another thread", flush=True)
            event.wait()
            with self._cache_lock:
                if cache_key in SpeechDetector.transcription_cache:
                    print(f"[DEBUG] transcribe_audio: after wait, cache hit, returning {len(SpeechDetector.transcription_cache[cache_key])} segments", flush=True)
                    return SpeechDetector.transcription_cache[cache_key]

        # Start transcription
        event = threading.Event()
        with self._progress_lock:
            SpeechDetector._transcription_in_progress[video_str] = event

        try:
            # Load full audio
            y, sr = self._get_audio(video_path)
            if y.size == 0:
                self.logger.error("Empty audio array")
                return None
            print(f"[DEBUG] transcribe_audio: audio loaded, shape {y.shape}, sr {sr}, duration {len(y)/sr:.1f}s", flush=True)

            # Chunk parameters
            chunk_seconds = 30
            overlap_seconds = 1.0
            samples_per_chunk = int(chunk_seconds * sr)
            overlap_samples = int(overlap_seconds * sr)
            step = samples_per_chunk - overlap_samples
            total_samples = len(y)
            num_chunks = int(math.ceil((total_samples - overlap_samples) / step))

            self.logger.info(f"Transcribing {video_path.name} using {num_chunks} chunks (chunk size {chunk_seconds}s, overlap {overlap_seconds}s)")
            print(f"[DEBUG] transcribe_audio: splitting into {num_chunks} chunks", flush=True)

            model = self._get_faster_whisper_model()
            print("[DEBUG] transcribe_audio: model obtained", flush=True)

            all_segments = []

            for i in range(num_chunks):
                start_sample = i * step
                end_sample = min(start_sample + samples_per_chunk, total_samples)
                chunk = y[start_sample:end_sample]
                chunk_duration = len(chunk) / sr

                # Skip chunks that are too short (< 0.5s) or empty
                if len(chunk) == 0 or chunk_duration < 0.5:
                    print(f"[DEBUG] Skipping chunk {i+1}/{num_chunks} (empty or too short: {chunk_duration:.2f}s)", flush=True)
                    continue

                offset = start_sample / sr

                # Write chunk to a temporary WAV file (PCM16 for robustness)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_audio = Path(tmp.name)
                sf.write(str(temp_audio), chunk, sr, format='WAV', subtype='PCM_16')
                print(f"[DEBUG] transcribe_audio: chunk {i+1}/{num_chunks} written to {temp_audio}", flush=True)

                # Transcribe this chunk (add vad_filter and language for better memory/accuracy)
                segments_generator, _ = model.transcribe(
                    str(temp_audio),
                    word_timestamps=False,
                    vad_filter=True,
                    vad_parameters=dict(min_speech_duration_ms=250),
                    language='en'
                )
                for seg in segments_generator:
                    all_segments.append({
                        "start": seg.start + offset,
                        "end": seg.end + offset,
                        "text": seg.text,
                    })

                # Clean up temp file
                temp_audio.unlink()
                print(f"[DEBUG] transcribe_audio: chunk {i+1}/{num_chunks} done, total segments: {len(all_segments)}", flush=True)

                # Free memory between chunks (if GPU)
                if hasattr(self.config, 'use_gpu') and self.config.use_gpu:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not all_segments:
                self.logger.error("No segments returned from transcription")
                return None

            # Merge overlapping/duplicate segments (improved logic)
            merged_segments = []
            for seg in all_segments:
                if not merged_segments:
                    merged_segments.append(seg)
                else:
                    last = merged_segments[-1]
                    # Check if segments overlap in time AND have identical text
                    if seg['start'] < last['end'] and seg['text'].strip() == last['text'].strip():
                        # Skip duplicate in overlap region
                        continue
                    merged_segments.append(seg)

            print(f"[DEBUG] transcribe_audio: total segments after merge: {len(merged_segments)}", flush=True)
            if merged_segments:
                print(f"[DEBUG] transcribe_audio: last segment end time: {merged_segments[-1]['end']:.2f}s", flush=True)
            print(f"[TRANSCRIPT FINAL] Segments: {len(merged_segments)}, last end: {merged_segments[-1]['end']:.2f}s", flush=True)

            # Store in cache
            with self._cache_lock:
                SpeechDetector.transcription_cache[cache_key] = merged_segments
                while len(SpeechDetector.transcription_cache) > self.max_cache_size:
                    SpeechDetector.transcription_cache.popitem(last=False)

            self.logger.info(f"Transcribed {len(merged_segments)} segments, last end: {merged_segments[-1]['end']:.2f}s")
            return merged_segments

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            print(f"[DEBUG] transcribe_audio: exception: {e}", flush=True)
            return None
        finally:
            event.set()
            with self._progress_lock:
                if video_str in SpeechDetector._transcription_in_progress:
                    del SpeechDetector._transcription_in_progress[video_str]

# ==================== JOKE DETECTOR ====================
class JokeDetector(SpeechDetector):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.JOKE_DETECTION)
        self.humor_classifier = None

    def _load_humor_classifier(self):
        if self.humor_classifier is None:
            try:
                from transformers import pipeline
                model_name = 'mohameddhiab/humor-no-humor'
                self.humor_classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
                self.logger.info("Joke detector model loaded (humor-no-humor)")
            except Exception as e:
                self.logger.error(f"Joke detector load failed: {e}")
                self.humor_classifier = None

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        segments = self.transcribe_audio(video_path)
        # Limit to first 100 segments for speed (most viral moments are early)
        MAX_SEGMENTS = 100
        if len(segments) > MAX_SEGMENTS:
            self.logger.info(f"Limiting joke detector to first {MAX_SEGMENTS} of {len(segments)} segments")
            segments = segments[:MAX_SEGMENTS]
        if not segments:
            return [] 
        self._load_humor_classifier()
        if self.humor_classifier is None:
            return []
        results = [] 
        batch_size = self.config.joke_batch_size
        texts = []
        midpoints = []
        for seg in tqdm(segments, desc="Joke detector", unit="segment"):
            text = seg['text'].strip()
            if len(text) < 5:
                continue
            texts.append(text)
            midpoints.append((seg['start'] + seg['end']) / 2)
            if len(texts) >= batch_size: 
                try:
                    preds = self.humor_classifier(texts)
                    print(f"[JOKE] Processed batch up to segment {len(midpoints)} / {len(segments)}", flush=True)
                    for i, p in enumerate(preds):
                        humor_score = 0.0
                        for item in p:
                            if isinstance(item, dict) and item.get('label', '').lower() in ('humor', 'funny', 'joke'):
                                humor_score = float(item.get('score', 0)) * 100
                                break
                        if humor_score > 40.0:
                            results.append(DetectionResult(self.method, midpoints[i], humor_score))
                except Exception as e:
                    self.logger.debug(f"Humor batch failed: {e}")
                texts = []
                midpoints = []
        if texts:
            try:
                preds = self.humor_classifier(texts)
                for i, p in enumerate(preds):
                    humor_score = 0.0
                    for item in p:
                        if isinstance(item, dict) and item.get('label', '').lower() in ('humor', 'funny', 'joke'):
                            humor_score = float(item.get('score', 0)) * 100
                            break
                    if humor_score > 40.0:
                        results.append(DetectionResult(self.method, midpoints[i], humor_score))
            except Exception as e:
                self.logger.debug(f"Humor batch failed: {e}")
        return self._apply_cooldown(results)

# ==================== ARGUMENT DETECTOR ====================
class ArgumentDetector(SpeechDetector):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.ARGUMENT_DETECTION)
        self.argument_patterns = [
            re.compile(r'\b(but|however|although|yet|though)\b', re.IGNORECASE),
            re.compile(r'\b(disagree|wrong|incorrect|mistake|lies)\b', re.IGNORECASE),
            re.compile(r'\?{2,}'), re.compile(r'!{2,}')
        ]

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        print("[DEBUG] Entering ArgumentDetector", flush=True)
        video_path = self._normalize_path(video_path)
        segments = self.transcribe_audio(video_path)
        if not segments:
            return []
        results = []
        for seg in segments:
            text = seg['text']
            text_upper = text.upper()
            midpoint = (seg['start'] + seg['end']) / 2
            pattern_matches = sum(1 for pat in self.argument_patterns if pat.search(text_upper))
            words = text.split()
            caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)
            caps_ratio = (caps_words / max(len(words), 1)) * 100
            score = min((pattern_matches * 20) + (caps_ratio * 0.5), 100.0)
            if score > 30.0:
                results.append(DetectionResult(self.method, midpoint, score))
        return self._apply_cooldown(results)

# ==================== EMOTIONAL PHRASE DETECTOR ====================
class EmotionalPhraseDetector(SpeechDetector):
    _vader_analyzer = None

    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.EMOTIONAL_PHRASE)
        self.word_pattern = re.compile(r'\b\w+\b')

    def _load_vader(self):
        if EmotionalPhraseDetector._vader_analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                EmotionalPhraseDetector._vader_analyzer = SentimentIntensityAnalyzer()
                self.logger.info("VADER sentiment analyzer loaded")
            except ImportError:
                self.logger.warning("vaderSentiment not installed")
                EmotionalPhraseDetector._vader_analyzer = None
        return EmotionalPhraseDetector._vader_analyzer

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        print("[DEBUG] Entering EmotionalPhraseDetector", flush=True)
        video_path = self._normalize_path(video_path)
        segments = self.transcribe_audio(video_path)
        if not segments:
            return []
        self._load_vader()
        if EmotionalPhraseDetector._vader_analyzer is None:
            return []
        emotional_lexicon = {'love','hate','amazing','terrible','death','crying','screaming',
            'heartbroken','ecstatic','devastated','incredible','horrifying'}
        results = []
        for seg in segments:
            text = seg['text'].strip()
            if len(text) < 5:
                continue
            midpoint = (seg['start'] + seg['end']) / 2
            sentiment = EmotionalPhraseDetector._vader_analyzer.polarity_scores(text)
            compound = abs(sentiment['compound'])
            words = set(self.word_pattern.findall(text.lower()))
            intense = len(words.intersection(emotional_lexicon))
            score = min((compound * 50) + (intense * 15), 100.0)
            if score > 30.0:
                results.append(DetectionResult(self.method, midpoint, score))
        return self._apply_cooldown(results)

# ==================== SHOCKING STATEMENT DETECTOR ====================
class ShockingStatementDetector(SpeechDetector):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.SHOCKING_STATEMENT)
        self.shocking_patterns = [
            re.compile(r'\b(holy shit|oh my god|wtf|what the fuck|omg|jesus christ)\b', re.IGNORECASE),
            re.compile(r'\b(can\'?t believe|no way|unbelievable|insane|mind[- ]blown)\b', re.IGNORECASE),
            re.compile(r'\b(secret|revealed|exposed|confession|truth[ -]bomb)\b', re.IGNORECASE)
        ]

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        print("[DEBUG] Entering ShockingStatementDetector", flush=True)
        video_path = self._normalize_path(video_path)
        segments = self.transcribe_audio(video_path)
        if not segments:
            return []
        results = []
        for seg in segments:
            text = seg['text']
            midpoint = (seg['start'] + seg['end']) / 2
            pattern_matches = sum(1 for pat in self.shocking_patterns if pat.search(text))
            exclamations = text.count('!')
            questions = text.count('?')
            caps_bonus = sum(1 for w in text.split() if w.isupper()) * 2
            score = min((pattern_matches * 30) + (exclamations * 6) + (questions * 4) + caps_bonus, 100.0)
            if score > 25.0:
                results.append(DetectionResult(self.method, midpoint, score))
        return self._apply_cooldown(results)

# ==================== VIDEO UNDERSTANDING DETECTOR (CLIP) ====================
class VideoUnderstandingDetector(DetectorBase):
    clip_model = None
    clip_processor = None

    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.VIDEO_UNDERSTANDING)

    def _load_clip_model(self):
        if VideoUnderstandingDetector.clip_model is None:
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.logger.info("Loading CLIP model...")
                VideoUnderstandingDetector.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                VideoUnderstandingDetector.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                if self.config.use_gpu and torch.cuda.is_available():
                    VideoUnderstandingDetector.clip_model = VideoUnderstandingDetector.clip_model.to('cuda')  # type: ignore[attr-defined]
                self.logger.info("CLIP model loaded")
            except Exception as e:
                self.logger.error(f"CLIP load failed: {e}")
                VideoUnderstandingDetector.clip_model = None
                VideoUnderstandingDetector.clip_processor = None
        return VideoUnderstandingDetector.clip_model, VideoUnderstandingDetector.clip_processor

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        model, processor = self._load_clip_model()
        if model is None or processor is None:
            return []
        cap = None
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(frame_count, self.config.clip_max_frames)
            sample_interval = max(1, frame_count // max_frames) if max_frames > 0 else 30
            viral_prompts = ["funny moment", "emotional crying scene", "argument or fight",
                             "surprising shocking moment", "high action movement", "crowd reaction"]
            frame_batch = []
            timestamps = []
            frame_idx = 0
            batch_size = self.config.clip_batch_size
            results = []
            processed_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    h, w = frame.shape[:2]
                    scale = 640 / max(h, w)
                    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                    rgb_frame = cv2.cvtColor(cv2.resize(frame, (new_w, new_h)), cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    frame_batch.append(pil_image)
                    timestamps.append(frame_idx / fps)
                    processed_frames += 1
                    if len(frame_batch) >= batch_size or processed_frames >= max_frames:
                        if frame_batch:
                            inputs = processor(text=viral_prompts, images=frame_batch, return_tensors="pt", padding=True)  # type: ignore[call-arg]
                            if self.config.use_gpu and torch.cuda.is_available():
                                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = outputs.logits_per_image.softmax(dim=1)
                            for i, prob in enumerate(probs):
                                max_sim = min(prob.max().cpu().item() * 100, 100.0)
                                if max_sim > 30.0:
                                    results.append(DetectionResult(self.method, timestamps[i], max_sim, float(prob.max().cpu().item())))
                        frame_batch = []
                        timestamps = []
                    if processed_frames >= max_frames:
                        break
                frame_idx += 1
            return self._apply_cooldown(results)
        except Exception as e:
            self.logger.error(f"CLIP analysis failed: {e}")
            return []
        finally:
            if cap:
                cap.release()

# ==================== HOOK DETECTOR ====================
class HookDetector(DetectorBase):
    hook_keywords = None

    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.HOOK_DETECTION)

    def _load_hook_keywords(self):
        if HookDetector.hook_keywords is None:
            hooks_file = Path("viral_hooks.json")
            default_hooks = ["hook", "pay attention", "listen up", "let me tell you", "you won't believe",
                             "shocking", "amazing", "insane", "crazy", "mind blowing", "unbelievable"]
            try:
                if hooks_file.exists():
                    HookDetector.hook_keywords = json.loads(hooks_file.read_text())
                else:
                    HookDetector.hook_keywords = default_hooks
                    hooks_file.write_text(json.dumps(default_hooks, indent=2))
            except Exception as e:
                self.logger.warning(f"Failed to load hooks, using defaults: {e}")
                HookDetector.hook_keywords = default_hooks

    def analyze(self, video_path: Path, all_results: Optional[Dict[float, List[DetectionResult]]] = None,
                transcript: Optional[List[Dict]] = None) -> List[DetectionResult]:
        video_path = self._normalize_path(video_path)
        self._load_hook_keywords()
        
        # Get duration using OpenCV (reliable, no ffprobe hang)
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
        else:
            duration = 0.0
        
        analysis_end = min(3.0, duration)
        weights = self.config.hook_weights
        relevant_methods = {
            DetectionMethod.AUDIO_ENERGY: weights.get("audio_energy", 0.2),
            DetectionMethod.MOTION: weights.get("motion", 0.1),
            DetectionMethod.SCENE_CHANGES: weights.get("scene_changes", 0.1),
            DetectionMethod.FACE_EMOTION: weights.get("face_emotion", 0.2),
            DetectionMethod.SHOCKING_STATEMENT: weights.get("shocking_statement", 0.2),
            DetectionMethod.EMOTIONAL_PHRASE: weights.get("emotional_phrase", 0.2),
        }
        weighted = 0.0
        total = 0.0
        if all_results:
            for ts, dets in all_results.items():
                if ts <= analysis_end:
                    for res in dets:
                        if res.method in relevant_methods:
                            w = relevant_methods[res.method]
                            weighted += res.score * w
                            total += w
        base_score = weighted / total if total > 0 else 0.0
        keyword_boost = 1.0
        if transcript:
            first_sentence = " ".join(seg['text'] for seg in transcript if seg['start'] < analysis_end).lower()
            if HookDetector.hook_keywords:
                found = [kw for kw in HookDetector.hook_keywords if kw.lower() in first_sentence]
                if found:
                    keyword_boost = 1.5
                    self.logger.debug(f"Hook keywords found: {found}")
        final_score = min(base_score * keyword_boost, 100.0)
        self.logger.info(f"Hook score (first {analysis_end}s): {final_score:.1f}")
        return [DetectionResult(self.method, 0.0, final_score)]

# ==================== NARRATIVE ANALYSIS DETECTOR ====================
class NarrativeAnalysisDetector(DetectorBase):
    def __init__(self, config: DetectionConfig):
        super().__init__(config, DetectionMethod.NARRATIVE_ANALYSIS)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.model is not None:
            return
        try:
            import accelerate
        except ImportError:
            self.logger.warning("`accelerate` not installed. Skipping narrative analysis (Phi-3 model).")
            return
        if self.config.use_gpu:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                self.logger.info("Loading Phi-3-mini on GPU (quantized)...")
                quantization = BitsAndBytesConfig(load_in_4bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quantization,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
                self.logger.info("Phi-3-mini loaded on GPU")
                return
            except Exception as e:
                self.logger.warning(f"GPU load failed: {e}, falling back to CPU")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.logger.info("Loading Phi-3-mini on CPU...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-4k-instruct",
                torch_dtype=torch.float32,
                device_map="cpu", 
                trust_remote_code=True
            )     
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)
            self.logger.info("Phi-3-mini loaded on CPU")
        except Exception as e:
            self.logger.error(f"Phi-3-mini load failed: {e}")
            self.model = None 
            self.tokenizer = None

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        return []

    def analyze_moment(self, transcript_segment: str) -> float:
        if self.model is None or self.tokenizer is None:
            return 0.0
        prompt = f"""<|user|>
Analyze this video transcript segment for narrative engagement:
Transcript: {transcript_segment}
Provide a JSON response with 'score' (0-100 viral narrative potential) and 'explanation'.
<|end|>
<|assistant|>
```json
{{"score": 85, "explanation": "Strong story hook"}}
```"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=False)  # type: ignore
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return min(float(data.get("score", 0)), 100.0)
        except Exception as e:
            self.logger.debug(f"Narrative analysis failed: {e}")
        return 0.0 