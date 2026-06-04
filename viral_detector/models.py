# viral_detector/models.py
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import uuid


from .config import DetectionMethod, ViralityTier

def _round_ts(ts: float) -> float:
    return round(ts, 2)
 
@dataclass
class DetectionResult:
    method: DetectionMethod 
    timestamp: float
    score: float
    confidence: float = 1.0

    def __post_init__(self):
        self.score = max(0.0, min(100.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.timestamp = _round_ts(float(self.timestamp))

@dataclass
class ViralMoment:
    start_time: float
    end_time: float
    duration: float 
    combined_score: float
    tier: ViralityTier
    method_scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    methods_count: int = 0
    peak_time: float = 0.0
    peak_score: float = 0.0
    explanations: List[str] = field(default_factory=list)

@dataclass 
class ProcessingResult:
    video_path: Path
    moments_detected: int
    clips_created: int
    tier_breakdown: Dict[str, int]
    processing_time: float
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())