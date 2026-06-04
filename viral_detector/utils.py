# viral_detector/utils.py
import os
import json
import logging
import subprocess
import threading
import time
import tempfile
import bisect
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
import psutil

from .config import DetectionConfig, DetectionMethod
from .models import DetectionResult   # ADDED

# ---------------------------- HELPER FUNCTIONS -------------------------------
def _round_ts(ts: float) -> float:
    return round(ts, 2)

# ==================== LOGGING ====================
class EmojiFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.emoji_map = {
            '📂': '[FOLDER]', '📁': '[DIR]', '⏱': '[TIME]', '⏱️': '[TIME]',
            '🎯': '[TARGET]', '🔄': '[REFRESH]', '👀': '[WATCH]', '🆕': '[NEW]',
            '📝': '[EDIT]', '📥': '[IN]', '✂️': '[CUT]', '✂': '[CUT]',
            '✅': '[OK]', '❌': '[FAIL]', '⏭': '[SKIP]', '⏭️': '[SKIP]',
            '📊': '[STATS]', '🎬': '[VIDEO]', '🛑': '[STOP]', '⏳': '[WAIT]',
            '🎥': '[VID]', '🏆': '[TOP]', '⭐': '[STAR]', '📌': '[PIN]',
        }
    def filter(self, record):
        try:
            msg = record.getMessage()
            for emoji, text in self.emoji_map.items():
                msg = msg.replace(emoji, text)
            record.msg = msg
            record.args = ()
        except Exception:
            pass
        return True

class LoggerFactory:
    _root_initialized = False
    _root_log_file = None

    @staticmethod
    def configure_root(log_file: Path, level: int = logging.DEBUG):
        if LoggerFactory._root_initialized:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        root = logging.getLogger()
        root.setLevel(level)
        root.addHandler(file_handler)
        root.addHandler(console_handler)
        root.addFilter(EmojiFilter())
        LoggerFactory._root_initialized = True
        LoggerFactory._root_log_file = log_file

    @staticmethod
    def create(name: str, log_file: Optional[Path] = None) -> logging.Logger:
        if not LoggerFactory._root_initialized and log_file:
            LoggerFactory.configure_root(log_file)
        return logging.getLogger(name)

# ==================== VIDEO UTILITIES ====================
class VideoMetrics:
    def __init__(self, ffprobe_path: Path):
        self.ffprobe_path = ffprobe_path
        self.logger = LoggerFactory.create(__name__)
        self._cache = OrderedDict()
        self._max_cache_size = 100
        self._cache_lock = threading.Lock()

    def _prune_cache(self):
        with self._cache_lock:
            while len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

    def get_video_info(self, video_path: Path) -> Dict[str, Union[float, int]]:
        if not video_path.exists():
            self.logger.error(f"Video not found: {video_path}")
            return {"duration": 0.0, "fps": 30.0, "frame_count": 0}
        video_path = video_path.resolve()
        video_str = str(video_path)
        with self._cache_lock:
            if video_str in self._cache and all(k in self._cache[video_str] for k in ("duration", "fps", "frame_count")):
                self._cache.move_to_end(video_str)
                return self._cache[video_str].copy()
        cmd = [str(self.ffprobe_path), '-v', 'error', '-show_entries', 'format=duration',
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        duration = 0.0
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
            else:
                self.logger.error(f"ffprobe failed for {video_path}: {result.stderr}")
        except Exception as e:
            self.logger.error(f"Failed to get duration: {e}")
        fps = 30.0
        frame_count = 0
        cap = None
        try:
            cap = cv2.VideoCapture(video_str)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.logger.error(f"Could not open video for FPS/frame count: {video_path}")
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
        finally:
            if cap:
                cap.release()
        with self._cache_lock:
            if video_str not in self._cache:
                self._cache[video_str] = {}
            self._cache[video_str]["duration"] = duration
            self._cache[video_str]["fps"] = fps
            self._cache[video_str]["frame_count"] = frame_count
            self._prune_cache()
        return {"duration": duration, "fps": fps, "frame_count": frame_count}

    def get_duration(self, video_path: Path) -> float:
        return self.get_video_info(video_path)["duration"]

    def get_fps(self, video_path: Path) -> float:
        return self.get_video_info(video_path)["fps"]

# ==================== DETECTOR BASE ====================
class DetectorBase:
    _audio_cache = OrderedDict()
    _audio_cache_lock = threading.Lock()

    def __init__(self, config: DetectionConfig, method: DetectionMethod):
        self.config = config
        self.method = method
        self.logger = LoggerFactory.create(__name__)
        self.video_metrics = VideoMetrics(config.ffprobe_path)

    def _normalize_path(self, video_path: Path) -> Path:
        return video_path.resolve()

    def analyze(self, video_path: Path) -> List[DetectionResult]:
        raise NotImplementedError

    def _apply_cooldown(self, results: List[DetectionResult], min_gap: float = 1.0) -> List[DetectionResult]:
        if not results:
            return results
        results = sorted(results, key=lambda r: r.timestamp)
        filtered = [results[0]]
        for res in results[1:]:
            if res.timestamp - filtered[-1].timestamp >= min_gap:
                filtered.append(res)
        return filtered

    def _get_audio(self, video_path: Path) -> Tuple[np.ndarray, int]:
        from .detectors import AudioEnergyDetector  # local import to avoid circular
        video_path = self._normalize_path(video_path)
        key = str(video_path)
        with self._audio_cache_lock:
            if key in self._audio_cache:
                self._audio_cache.move_to_end(key)
                self.logger.debug(f"Audio cache hit for {video_path.name}")
                return self._audio_cache[key]
        import librosa
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_audio = Path(tmp.name)
        try:
            cmd = [str(self.config.ffmpeg_path), '-i', str(video_path), '-vn',
                   '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', str(temp_audio)]
            subprocess.run(cmd, capture_output=True, timeout=300, check=False)
            y, sr = librosa.load(str(temp_audio), sr=16000)
            with self._audio_cache_lock:
                self._audio_cache[key] = (y, int(sr))
                while len(self._audio_cache) > self.config.audio_cache_max:
                    self._audio_cache.popitem(last=False)
            return y, int(sr)
        finally:
            if temp_audio.exists():
                temp_audio.unlink()

    @classmethod
    def clear_audio_cache(cls):
        with cls._audio_cache_lock:
            cls._audio_cache.clear()

# ==================== FEATURE EXTRACTOR ====================
class FeatureExtractor:
    def __init__(self, config: DetectionConfig):
        self.feature_names = []
        for method in DetectionMethod:
            self.feature_names.append(f"{method.value}_mean")
            self.feature_names.append(f"{method.value}_std")

    def extract(self, all_results: Dict[float, List[DetectionResult]]) -> pd.DataFrame:
        data = []
        for ts, dets in all_results.items():
            for det in dets:
                data.append({'timestamp': ts, 'method': det.method.value, 'score': det.score})
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df['timestamp_bin'] = df['timestamp'].round(2) + 0.5
        grouped = df.groupby(['timestamp_bin', 'method'])['score'].agg(['mean', 'std']).unstack(fill_value=0)
        # Flatten columns – ignore Pylance false positive
        grouped.columns = [f"{col[1]}_{col[0]}" if col[0] != '' else f"{col[1]}" for col in grouped.columns]  # type: ignore[attr-defined]
        for name in self.feature_names:
            if name not in grouped.columns:
                grouped[name] = 0.0
        grouped = grouped.reset_index().rename(columns={'timestamp_bin': 'timestamp'})
        grouped['video_id'] = 'temp'
        return grouped[self.feature_names + ['timestamp', 'video_id']]

# ==================== VIRAL PATTERN MEMORY ====================
class ViralPatternMemory:
    def __init__(self, storage_path: Path, similarity_threshold: float = 0.85, boost_factor: float = 1.15):
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.boost_factor = boost_factor
        self.patterns = []
        self.logger = LoggerFactory.create(__name__)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', [])
                self.logger.info(f"Loaded {len(self.patterns)} patterns")
            except Exception as e:
                self.logger.error(f"Failed to load patterns: {e}")

    def save(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump({'patterns': self.patterns}, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save patterns: {e}")

    def add_pattern(self, features: List[float], score: float, video_name: str, timestamp: float, min_score_threshold: float = 70.0):
        if score < min_score_threshold:
            return
        with self._lock:
            for p in self.patterns[-10:]:
                sim = self._cosine_similarity(features, p['features'])
                if sim > 0.95:
                    self.logger.debug(f"Pattern duplicate (sim={sim:.3f}), skipping")
                    return
            pattern = {'features': features, 'score': score, 'video': video_name,
                       'timestamp': timestamp, 'added': datetime.now().isoformat()}
            self.patterns.append(pattern)
            if len(self.patterns) > 500:
                self.patterns = self.patterns[-500:]
            self.save()

    def find_similar_boost(self, features: List[float]) -> float:
        with self._lock:
            if not self.patterns:
                return 1.0
            best_sim = 0.0
            for p in self.patterns:
                sim = self._cosine_similarity(features, p['features'])
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= self.similarity_threshold:
                boost = 1.0 + (best_sim - self.similarity_threshold) * (self.boost_factor - 1.0) / (1.0 - self.similarity_threshold)
                boost = min(boost, self.boost_factor)
                self.logger.debug(f"Pattern match sim={best_sim:.3f} boost={boost:.2f}")
                return boost
            return 1.0

    @staticmethod
    def _cosine_similarity(a, b):
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x,y in zip(a,b))
        na = sum(x*x for x in a)**0.5
        nb = sum(y*y for y in b)**0.5
        return dot / (na*nb) if na and nb else 0.0

# ==================== ML MODEL MANAGER ====================
class MLModelManager:
    def __init__(self, model_path: str = "viral_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = []
        for method in DetectionMethod:
            self.feature_names.append(f"{method.value}_mean")
            self.feature_names.append(f"{method.value}_std")

    def train(self, csv_path: str) -> None:
        df = pd.read_csv(csv_path)
        for col in self.feature_names:
            if col not in df:
                df[col] = 0.0
        X = df[self.feature_names]
        y = df['viral']
        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts)>1 and class_counts[0]>0 else 1.0
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                       scale_pos_weight=scale_pos_weight, random_state=42)
        self.model.fit(X_scaled, y)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names}, self.model_path)
        print(f"Model trained and saved to {self.model_path}")

    def load(self) -> bool:
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded")
        X = features_df[self.feature_names].fillna(0) 
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1] * 100    