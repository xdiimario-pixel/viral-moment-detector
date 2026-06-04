# viral_detector/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

class DetectionMethod(str, Enum):
    AUDIO_ENERGY = "audio_energy"
    MOTION = "motion_intensity"
    SCENE_CHANGES = "scene_changes"
    SILENCE_DETECTION = "silence_detection"
    FACE_EMOTION = "face_emotion"
    OBJECT_DETECTION = "object_detection"
    JOKE_DETECTION = "joke_detection"
    ARGUMENT_DETECTION = "argument_detection"
    EMOTIONAL_PHRASE = "emotional_phrase"
    SHOCKING_STATEMENT = "shocking_statement"
    VIDEO_UNDERSTANDING = "video_understanding"
    HOOK_DETECTION = "hook_detection"
    NARRATIVE_ANALYSIS = "narrative_analysis"

class ContentProfile(str, Enum):
    PODCAST = "podcast"
    GAMING = "gaming"
    REACTION = "reaction"  

PROFILE_WEIGHTS = {
    ContentProfile.PODCAST: {
        DetectionMethod.ARGUMENT_DETECTION: 1.5,
        DetectionMethod.EMOTIONAL_PHRASE: 1.5,
        DetectionMethod.SHOCKING_STATEMENT: 1.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
        DetectionMethod.FACE_EMOTION: 1.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
        DetectionMethod.AUDIO_ENERGY: 1.0,
        DetectionMethod.MOTION: 0.5,
        DetectionMethod.SCENE_CHANGES: 0.5,
        DetectionMethod.OBJECT_DETECTION: 0.5,
    },
    ContentProfile.GAMING: {
        DetectionMethod.AUDIO_ENERGY: 1.5,
        DetectionMethod.MOTION: 1.5,
        DetectionMethod.FACE_EMOTION: 1.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
        DetectionMethod.SHOCKING_STATEMENT: 1.5,
        DetectionMethod.SCENE_CHANGES: 1.0,
        DetectionMethod.OBJECT_DETECTION: 1.0,
        DetectionMethod.ARGUMENT_DETECTION: 0.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
    },
    ContentProfile.REACTION: {
        DetectionMethod.FACE_EMOTION: 1.5,
        DetectionMethod.AUDIO_ENERGY: 1.5,
        DetectionMethod.EMOTIONAL_PHRASE: 1.5,
        DetectionMethod.SHOCKING_STATEMENT: 1.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
        DetectionMethod.MOTION: 1.0,
        DetectionMethod.SCENE_CHANGES: 1.0,
        DetectionMethod.OBJECT_DETECTION: 0.5,
        DetectionMethod.HOOK_DETECTION: 1.5,
        DetectionMethod.ARGUMENT_DETECTION: 0.5,
    },
}  

class ViralityTier(str, Enum):
    TIER_A = "A"
    TIER_B = "B"
    TIER_C = "C"

@dataclass
class DetectionConfig:
    watch_folder: Path
    output_folder: Path
    ffmpeg_path: Path
    ffprobe_path: Path

    clip_duration: int = 15
    min_moment_duration: float = 2.0
    max_moment_duration: float = 60.0
    silence_hop_length: int = 2048   # default

    audio_energy_threshold: float = 60.0
    motion_threshold: float = 40.0
    scene_change_threshold: float = 35.0
    silence_threshold: float = 20.0
    moment_threshold: float = 30.0
    gap_tolerance: float = 1.5
    smoothing_window: int = 1   # moving average window size (must be odd)

    content_profile: Optional[ContentProfile] = None
    _profile_multipliers: Dict[DetectionMethod, float] = field(default_factory=dict)

    tier_a_threshold: float = 80.0
    tier_b_threshold: float = 50.0
    tier_c_threshold: float = 20.0
    hook_penalty_threshold: float = 20.0
    hook_weights: Dict[str, float] = field(default_factory=lambda: {
        "audio_energy": 0.2, "motion": 0.1, "scene_changes": 0.1,
        "face_emotion": 0.2, "shocking_statement": 0.2, "emotional_phrase": 0.2
    })

    processing_mode: str = "balanced"   # fast / balanced / full
    enabled_methods: Set[DetectionMethod] = field(default_factory=set)

    supported_formats: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    log_file: str = "viral_detector.log"
    state_file: str = ".processed_videos.json"
    ml_model_path: Optional[str] = None
    use_gpu: bool = False
    use_semantic_detectors: bool = False   # set to True to activate
    enable_profiling: bool = False

    # Smart boundaries & captions
    smart_boundaries: bool = True
    boundary_tolerance: float = 0.5
    add_captions: bool = False 
    caption_export_srt: bool = False   # if True, generate .srt instead of burning
    caption_style: Dict[str, str] = field(default_factory=lambda: {
        "font": "Arial", "fontsize": "24", "fontcolor": "white",
        "bordercolor": "black", "borderw": "1", "alignment": "2"
    })
    vertical_export: bool = False
    vertical_aspect: str = "1:1"


    # Resource control
    max_detector_workers: int = 4
    max_cutter_workers: int = 3
    min_available_ram_gb: float = 2.0

    # Whisper (direct transcription)
    whisper_model_name: str = "tiny"

    # Motion detection streaming & scaling
    use_streaming_motion: bool = False
    use_fixed_motion_scaling: bool = False
    motion_fixed_scale: float = 10.0

    # Feature fusion
    fusion_enabled: bool = False
    fusion_boost_factor: float = 1.2
    fusion_high_threshold: float = 70.0
    fusion_min_overlap: float = 0.5

    # Audio cache size
    audio_cache_max: int = 10

    # Detector weights
    detector_weights: Dict[str, float] = field(default_factory=lambda: {
        "shocking_statement": 1.5,
        "face_emotion": 1.2,
        "emotional_phrase": 1.2,
        "audio_energy": 1.0,
        "motion": 0.8,
        "scene_changes": 0.6,
        "silence_detection": 0.2,
        "object_detection": 0.5,
        "joke_detection": 1.0,
        "argument_detection": 1.0,
        "video_understanding": 1.0,
        "hook_detection": 1.0,
        "narrative_analysis": 0.5,
    })

    ml_blending_ratio: float = 0.3

    pattern_min_score: float = 70.0
    pattern_boost_max: float = 1.15
    pattern_dedup_threshold: float = 0.95

    narrative_min_score: float = 60.0
    narrative_top_k: int = 5
    narrative_top_fraction: float = 0.2

    joke_batch_size: int = 16
    clip_batch_size: int = 16
    clip_max_frames: int = 100
    clip_use_full_distribution: bool = False

    # Boundary refinement parameters (tunable)
    allowed_gap_punc: float = 0.5
    allowed_gap_no_punc: float = 2.0
    look_ahead_max_gap: float = 2.5
    max_expand_sec: float = 20.0
    max_clip_duration: float = 60.0
    max_gap_for_same_sentence: float = 3.0

    def __post_init__(self):
        self.validate()
        # Build profile multipliers if a profile is selected
        if self.content_profile is not None:
            weights = PROFILE_WEIGHTS.get(self.content_profile, {})
            # For all detection methods, set multiplier = weights.get(method, 1.0)
            self._profile_multipliers = {method: weights.get(method, 1.0) for method in DetectionMethod}
        if self.processing_mode == "fast":
            self.whisper_model_name = "tiny"
        elif self.processing_mode == "balanced": 
            self.whisper_model_name = "tiny"
        elif self.processing_mode == "full":
            self.whisper_model_name = "small"

    @classmethod
    def fast(cls, watch_folder: Path, output_folder: Path, **kwargs) -> "DetectionConfig":
        kwargs.setdefault("processing_mode", "fast")
        config = cls(watch_folder, output_folder, **kwargs)
        config.apply_mode()
        return config

    @classmethod
    def balanced(cls, watch_folder: Path, output_folder: Path, **kwargs) -> "DetectionConfig":
        kwargs.setdefault("processing_mode", "balanced")
        config = cls(watch_folder, output_folder, **kwargs)
        config.apply_mode()
        return config

    @classmethod
    def full(cls, watch_folder: Path, output_folder: Path, **kwargs) -> "DetectionConfig":
        kwargs.setdefault("processing_mode", "full")
        config = cls(watch_folder, output_folder, **kwargs)
        config.apply_mode()
        return config

    def apply_mode(self) -> None:
        if self.enabled_methods:
            return
        if self.processing_mode == "fast":
            self.enabled_methods = {DetectionMethod.AUDIO_ENERGY, DetectionMethod.MOTION, DetectionMethod.SCENE_CHANGES}
        elif self.processing_mode == "balanced":
            self.enabled_methods = {
                DetectionMethod.AUDIO_ENERGY, DetectionMethod.MOTION, DetectionMethod.SCENE_CHANGES,
                DetectionMethod.FACE_EMOTION, DetectionMethod.OBJECT_DETECTION,
                # DetectionMethod.JOKE_DETECTION, 
                DetectionMethod.ARGUMENT_DETECTION,
                DetectionMethod.EMOTIONAL_PHRASE, DetectionMethod.SHOCKING_STATEMENT,
                DetectionMethod.HOOK_DETECTION
            }
        elif self.processing_mode == "full": 
            self.enabled_methods = set(DetectionMethod)
        else:
            self.processing_mode = "balanced"
            self.apply_mode()

    def validate(self) -> None:
        threshold_attrs = ["audio_energy_threshold", "motion_threshold", "scene_change_threshold", "silence_threshold"]
        for attr in threshold_attrs:
            val = getattr(self, attr)
            if not (0 <= val <= 100):
                raise ValueError(f"{attr} must be between 0 and 100, got {val}")
        if not (0 <= self.tier_c_threshold < self.tier_b_threshold < self.tier_a_threshold <= 100):
            raise ValueError(
                f"Invalid tier thresholds: tier_c={self.tier_c_threshold}, "
                f"tier_b={self.tier_b_threshold}, tier_a={self.tier_a_threshold}. "
                "Must satisfy: 0 <= C < B < A <= 100"
            )  
        if self.min_moment_duration > self.max_moment_duration:
            raise ValueError(f"min_moment_duration ({self.min_moment_duration}) > max_moment_duration ({self.max_moment_duration})")