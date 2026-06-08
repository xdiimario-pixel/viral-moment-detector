# viral_detector/analyzer.py
import time
import threading
import numpy as np
import cv2
import psutil
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DetectionConfig, DetectionMethod, ViralityTier
from .models import ViralMoment, DetectionResult
from .utils import (
    LoggerFactory, VideoMetrics, FeatureExtractor, ViralPatternMemory,
    MLModelManager, _round_ts, DetectorBase
)
from .detectors import (
    AudioEnergyDetector, MotionDetector, SceneChangeDetector, SilenceDetector,
    FaceEmotionDetector, ObjectDetector, JokeDetector, ArgumentDetector,
    EmotionalPhraseDetector, ShockingStatementDetector, VideoUnderstandingDetector,
    HookDetector, NarrativeAnalysisDetector, SpeechDetector
)

class MomentAnalyzer:
    def __init__(self, config: DetectionConfig):    
        self.config = config
        self.video_metrics = VideoMetrics(config.ffprobe_path)
        self.logger = LoggerFactory.create(__name__)
        self.detectors = self._setup_detectors()
        self.detector_errors = {}   # track detector failures
        self.feature_extractor = FeatureExtractor(config)
        self.ml_model = MLModelManager(config.ml_model_path or "viral_model.pkl")
        self.use_ml = self.ml_model.load() if config.ml_model_path else False
        if not self.use_ml:
            self.logger.info("ML model disabled, setting blending ratio to 0")
            self.config.ml_blending_ratio = 0.0
        pattern_storage = config.watch_folder / "viral_patterns.json"
        self.pattern_memory = ViralPatternMemory(pattern_storage, boost_factor=config.pattern_boost_max)
        self.narrative_detector = NarrativeAnalysisDetector(config) if DetectionMethod.NARRATIVE_ANALYSIS in config.enabled_methods else None
        self.logger.info("Viral Pattern Memory initialized")
        self._ready = True
    # Mapping from DetectionMethod to human‑readable explanation phrase
    EXPLANATION_MAP = {
        "audio_energy": "Sudden audio spike",
        "motion_intensity": "Rapid motion or action",
        "scene_changes": "Fast scene cuts",
        "silence_detection": "Dramatic pause or silence",
        "face_emotion": "Strong emotional reaction",
        "object_detection": "Viral object or person detected",
        "joke_detection": "Humorous moment",
        "argument_detection": "Argument or disagreement",
        "emotional_phrase": "Intense emotional language",
        "shocking_statement": "Shocking or surprising statement",
        "video_understanding": "Visually viral scene",
        "hook_detection": "Strong hook in opening seconds",
        "narrative_analysis": "Compelling story moment",
    }        

    def _cleanup_gpu_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_continuation_word(self, text: str) -> bool:
        continuation_words = {'and', 'but', 'or', 'because', 'so', 'for', 'nor', 'yet', 'with', 'from', 'at', 'about', 'the', 'a', 'an', 'of', 'to', 'by'}
        words = text.strip().split()
        if not words:
            return False
        last_word = words[-1].lower().strip('.,!?;:')
        return last_word in continuation_words
    
    def _generate_explanations(self, moment: ViralMoment) -> List[str]:
        """Return a list of human‑readable reasons for the moment's score."""
        # Filter detectors with score > 20
        candidates = [(det, score) for det, score in moment.method_scores.items() if score > 20.0]
        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Take up to 4 highest
        top = candidates[:4]
        # Map to phrases
        phrases = []
        for det, score in top:
            phrase = self.EXPLANATION_MAP.get(det)
            if phrase:
                phrases.append(phrase)
        return phrases
    
    def _label_sentences_with_spacy(self, transcript: List[Dict]) -> Optional[List[int]]:
        try:
            import spacy
        except ImportError: 
            self.logger.warning("spaCy not installed, falling back to dynamic gap method")
            return None
        nlp = spacy.load("en_core_web_sm")
        full_text = " ".join(seg['text'] for seg in transcript)
        doc = nlp(full_text)
        sent_ends = []
        pos = 0
        for sent in doc.sents:
            pos += len(sent.text)
            sent_ends.append(pos)
        seg_starts = [] 
        pos = 0
        for seg in transcript:
            seg_starts.append(pos)
            pos += len(seg['text']) + 1
        seg_sent_ids = []
        for seg_start in seg_starts:
            sent_id = 0
            while sent_id < len(sent_ends) and seg_start >= sent_ends[sent_id]:
                sent_id += 1
            seg_sent_ids.append(sent_id)
        return seg_sent_ids

    def _setup_detectors(self) -> List['DetectorBase']:
        detectors = []
        if DetectionMethod.AUDIO_ENERGY in self.config.enabled_methods:
            detectors.append(AudioEnergyDetector(self.config))
        if DetectionMethod.MOTION in self.config.enabled_methods:
            detectors.append(MotionDetector(self.config))
        if DetectionMethod.SCENE_CHANGES in self.config.enabled_methods:
            detectors.append(SceneChangeDetector(self.config))
        if DetectionMethod.SILENCE_DETECTION in self.config.enabled_methods:
            detectors.append(SilenceDetector(self.config))
        if DetectionMethod.FACE_EMOTION in self.config.enabled_methods:
            detectors.append(FaceEmotionDetector(self.config))
        if DetectionMethod.OBJECT_DETECTION in self.config.enabled_methods:
            detectors.append(ObjectDetector(self.config))
        if DetectionMethod.JOKE_DETECTION in self.config.enabled_methods:
            detectors.append(JokeDetector(self.config))
        if DetectionMethod.ARGUMENT_DETECTION in self.config.enabled_methods:
            detectors.append(ArgumentDetector(self.config))
        if DetectionMethod.EMOTIONAL_PHRASE in self.config.enabled_methods:
            detectors.append(EmotionalPhraseDetector(self.config))
        if DetectionMethod.SHOCKING_STATEMENT in self.config.enabled_methods:
            detectors.append(ShockingStatementDetector(self.config))
        if DetectionMethod.VIDEO_UNDERSTANDING in self.config.enabled_methods:
            detectors.append(VideoUnderstandingDetector(self.config))
        if DetectionMethod.HOOK_DETECTION in self.config.enabled_methods:
            detectors.append(HookDetector(self.config))
        return detectors

    def analyze_video(self, video_path: Path) -> Tuple[List[ViralMoment], Optional[List[Dict]]]:
        video_path = video_path.resolve()
        # Get duration using OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            cap.release()
        else: 
            duration = 0.0
            self.logger.error(f"Cannot open video: {video_path}")
            return [], None

        if duration < self.config.min_moment_duration:
            self.logger.warning(f"Video too short ({duration:.1f}s < {self.config.min_moment_duration}s), skipping")
            return [], None

        if psutil.virtual_memory().available < self.config.min_available_ram_gb * (1024**3):
            self.logger.warning(f"Low RAM (<{self.config.min_available_ram_gb}GB). Analysis may be slow or fail.")

        frame_detectors = [d for d in self.detectors if d.method != DetectionMethod.NARRATIVE_ANALYSIS]
        print(f"[DEBUG] Detectors in frame_detectors: {[d.method.value for d in frame_detectors]}", flush=True)
        all_results = defaultdict(list)
        lock = threading.Lock()
        detector_times = {}
        times_lock = threading.Lock()
        max_workers = min(self.config.max_detector_workers, len(frame_detectors))
        # --- FrameProvider for visual detectors ---
        from .frame_provider import FrameProvider
        # Get fps and total frames
        cap_temp = cv2.VideoCapture(str(video_path))
        if not cap_temp.isOpened():
            self.logger.error(f"Cannot open video for FrameProvider: {video_path}")
            return [], None
        fps = cap_temp.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()

        provider = FrameProvider(video_path, fps, total_frames)

        # Register consumers
        motion_sample_rate = max(10, int(fps // 2))
        def motion_preprocess(frame, idx, ts):
            # Resize to target_width=480, convert to grayscale
            h, w = frame.shape[:2]
            scale = 480 / w
            new_h = int(h * scale)
            resized = cv2.resize(frame, (480, new_h))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            return (gray, idx, ts)
        motion_queue = provider.register_consumer('motion', motion_sample_rate, motion_preprocess, queue_size=5000)

        scene_sample_rate = max(5, int(fps // 3))
        scene_queue = provider.register_consumer('scene', scene_sample_rate, lambda frame, idx, ts: (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), idx, ts), queue_size=5000)

        face_sample_rate = max(30, int(fps * 2))
        def face_preprocess(frame, idx, ts):
            h, w = frame.shape[:2]
            if h > 480:
                scale = 480 / h
                new_w, new_h = int(w * scale), 480
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
            # Return RGB frame (DeepFace expects RGB) and also the grayscale for detection?
            # We'll return the resized BGR; detector will convert as needed.
            return (frame_resized, idx, ts)
        face_queue = provider.register_consumer('face', face_sample_rate, face_preprocess, queue_size=5000)

        object_sample_rate = max(30, int(fps))
        def object_preprocess(frame, idx, ts):
            h, w = frame.shape[:2]
            scale = 640 / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(frame, (new_w, new_h))
            return (resized, idx, ts)
        object_queue = provider.register_consumer('object', object_sample_rate, object_preprocess, queue_size=5000)

        # Start the provider thread
        provider.start()

        # Map detector methods to queues
        queue_map = {
            DetectionMethod.MOTION: motion_queue,
            DetectionMethod.SCENE_CHANGES: scene_queue,
            DetectionMethod.FACE_EMOTION: face_queue,
            DetectionMethod.OBJECT_DETECTION: object_queue,
        }
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def process_detector(detector):
            det_name = detector.method.value
            try:
                start = time.perf_counter()
                # If this detector has a queue, pass it
                queue = queue_map.get(detector.method)
                if queue is not None:
                    results = detector.analyze(video_path, frame_queue=queue)
                else:
                    results = detector.analyze(video_path)
                elapsed = time.perf_counter() - start
                if self.config.enable_profiling:
                    with times_lock:
                        detector_times[det_name] = elapsed
                with lock:
                    for r in results:
                        r.timestamp = _round_ts(r.timestamp)
                        all_results[r.timestamp].append(r)
            except Exception as e:
                self.logger.error(f"Detector {det_name} failed: {e}")
                self.detector_errors[det_name] = str(e)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_detector, d): d for d in frame_detectors}
            for future in as_completed(futures):
                future.result()
        print("[DEBUG] Executor completed – all detectors finished", flush=True)


        if self.config.enable_profiling and detector_times:
            self.logger.info("=== Detector timings ===")
            for name, dur in sorted(detector_times.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {name}: {dur:.3f}s")

        speech_detectors = [d for d in frame_detectors if isinstance(d, SpeechDetector)]
        transcript = None
        if speech_detectors:
            first = speech_detectors[0]
            cache_key = f"{video_path.resolve()}_{video_path.stat().st_mtime}"
            with SpeechDetector._cache_lock:
                if cache_key in SpeechDetector.transcription_cache:
                    transcript = SpeechDetector.transcription_cache[cache_key]
        if transcript is None:
            print("[DEBUG] Calling transcribe_audio...", flush=True)
            transcript = first.transcribe_audio(video_path)
            print(f"[DEBUG] Transcript segments: {len(transcript) if transcript else 0}", flush=True)
            if transcript:
               print(f"[DEBUG] Transcript last end: {transcript[-1]['end']:.2f}", flush=True)

        # --- Semantic detectors (if enabled) ---
        if self.config.use_semantic_detectors and transcript:
            from .detectors import compute_semantic_scores
            sem_scores = compute_semantic_scores(transcript)
            for i, seg in enumerate(transcript):
                ts = _round_ts(seg['start'])
                for det_name, scores in sem_scores.items():
                    score = scores[i]
                    if score > 20.0:
                        method = DetectionMethod[det_name.upper()]
                        all_results[ts].append(DetectionResult(method, ts, score))

        # Hook detection
        hook = next((d for d in self.detectors if isinstance(d, HookDetector)), None)
        if hook:
            hook_results = hook.analyze(video_path, dict(all_results), transcript)
            for r in hook_results:
                all_results[_round_ts(r.timestamp)].append(r)
        print("[DEBUG] Hook detector finished", flush=True)

   
        self._cleanup_gpu_memory()
        if not all_results:
            self.logger.warning("No detections")
            return [], None 
 
        features_df = self.feature_extractor.extract(all_results)
        if self.use_ml and len(features_df) > 0:
            ml_scores = self.ml_model.predict_proba(features_df)
        else:
            ml_scores = np.full(len(features_df), 50.0)
        if transcript:
           print(f"[DEBUG] Passing transcript to combine: len={len(transcript)}, last_end={transcript[-1]['end']:.2f}", flush=True)
        else:
           print("[DEBUG] Passing transcript to combine: None", flush=True)    
        raw_moments = self._combine_detections(video_path, all_results, transcript)
        moments = self._rank_moments_with_ml(video_path, raw_moments, ml_scores, features_df, all_results)

        if self.narrative_detector and transcript and moments:
            n = max(1, int(len(moments) * self.config.narrative_top_fraction))
            n = min(n, self.config.narrative_top_k)
            candidates = moments[:n]
            for moment in candidates:
                if moment.combined_score >= self.config.narrative_min_score:
                    segment_texts = []
                    for seg in transcript:
                        if seg['end'] >= moment.start_time and seg['start'] <= moment.end_time:
                            segment_texts.append(seg['text'])
                    if segment_texts:
                        combined_text = " ".join(segment_texts)[:1000]
                        narrative_score = self.narrative_detector.analyze_moment(combined_text)
                        if narrative_score > 0:
                            moment.combined_score = moment.combined_score * 0.8 + narrative_score * 0.2
                            moment.method_scores['narrative_analysis'] = narrative_score

        self.logger.info(f"Found {len(moments)} viral moments")
        return moments, transcript

    def _combine_detections(self, video_path: Path, results: Dict[float, List[DetectionResult]], transcript: Optional[List[Dict]] = None, video_duration: Optional[float] = None) -> List[ViralMoment]:
        print(f"[DEBUG] _combine_detections received transcript: len={len(transcript) if transcript else 0}", flush=True)
        if transcript:
            print(f"[DEBUG] _combine_detections transcript last_end: {transcript[-1]['end']:.2f}", flush=True)        
        moments = []
        if video_duration is None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = frame_count / fps if fps > 0 else 0.0
                cap.release()
            else:
                video_duration = 0.0
        
        # ---- First pass: collect raw scores for each timestamp ----
        timestamps = []
        raw_scores = []
        for ts in sorted(results.keys()):
            dets = results[ts]
            weighted_sum = 0.0
            total_weight = 0.0
            for d in dets:
                base_w = self.config.detector_weights.get(d.method.value, 1.0)
                profile_mult = self.config._profile_multipliers.get(d.method, 1.0)
                effective_w = base_w * profile_mult
                weighted_sum += d.score * effective_w
                total_weight += effective_w
            avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
            timestamps.append(ts)
            raw_scores.append(avg_score)
        
        if not timestamps:
            return []
        
        # ---- Apply temporal smoothing (moving average) ----
        window = self.config.smoothing_window
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(raw_scores, pad, mode='edge')
        smoothed = np.convolve(padded, np.ones(window)/window, mode='valid')
        for i in range(min(10, len(timestamps))):
            print(f"    {timestamps[i]:.2f}: raw={raw_scores[i]:.2f} smoothed={smoothed[i]:.2f}", flush=True)
        if len(timestamps) > 10:
            for i in range(max(0, len(timestamps)-10), len(timestamps)):
                print(f"    {timestamps[i]:.2f}: raw={raw_scores[i]:.2f} smoothed={smoothed[i]:.2f}", flush=True)        
        
        # ---- Second pass: use smoothed scores for moment merging ----
        current = None
        min_dur = self.config.min_moment_duration
        gap_tol = self.config.gap_tolerance
        last_high_ts = None
        weights = self.config.detector_weights
        
        for idx, ts in enumerate(timestamps):
            avg_score = smoothed[idx]
            is_above = avg_score > self.config.moment_threshold
            
            if is_above:
                if current is None:
                    current = {'start': ts, 'timestamps': [ts], 'scores': [avg_score], 'methods': defaultdict(list)}
                    dets = results[ts]
                    for d in dets:
                        current['methods'][d.method.value].append(d.score)
                else:
                    print(f"[DEBUG] extending moment, current start={current['start']:.2f}", flush=True)
                    current['timestamps'].append(ts)
                    current['scores'].append(avg_score)
                    dets = results[ts]
                    for d in dets:
                        current['methods'][d.method.value].append(d.score)
                last_high_ts = ts
            else:
                if current is not None and last_high_ts is not None and ts - last_high_ts > gap_tol:
                    print(f"[DEBUG] ending moment at {last_high_ts:.2f}, current start={current['start']:.2f}, duration={current['end'] - current['start']:.2f}", flush=True)
                    current['end'] = last_high_ts
                    current['duration'] = current['end'] - current['start']
                    if current['duration'] >= min_dur:
                        peak_ts = current['start']
                        peak_score = current['scores'][0]
                        for t, sc in zip(current['timestamps'], current['scores']):
                            if sc > peak_score:
                                peak_score = sc
                                peak_ts = t
                        moment = ViralMoment(
                            start_time=current['start'], end_time=current['end'], duration=current['duration'],
                            combined_score=np.mean(current['scores']), tier=ViralityTier.TIER_A,
                            method_scores={k: np.mean(v) for k, v in current['methods'].items()},
                            peak_time=peak_ts,
                            peak_score=peak_score
                        )
                        print(f"[DEBUG] moment finalised: start={current['start']:.2f} end={current['end']:.2f} dur={current['duration']:.2f} score={np.mean(current['scores']):.2f}", flush=True)
                        if transcript and self.config.smart_boundaries:
                            moment = self._refine_moment_boundaries(moment, transcript, video_path, video_duration)
                        moments.append(moment)
                    current = None
                # else: if current exists but gap not exceeded, do nothing (still inside moment)
        
        for i, m in enumerate(moments):
            print(f"  {i}: {m.start_time:.2f}-{m.end_time:.2f} score={m.combined_score:.2f} peak={m.peak_time:.2f}", flush=True)
        
        # Handle final moment
        if current is not None:
            current['duration'] = video_duration - current['start']
            if current['duration'] >= min_dur:
                peak_ts = current['start']
                peak_score = current['scores'][0]
                for t, sc in zip(current['timestamps'], current['scores']):
                    if sc > peak_score:
                        peak_score = sc
                        peak_ts = t
                moment = ViralMoment(
                    start_time=current['start'], end_time=video_duration, duration=current['duration'],
                    combined_score=np.mean(current['scores']), tier=ViralityTier.TIER_A,
                    method_scores={k: np.mean(v) for k, v in current['methods'].items()},
                    peak_time=peak_ts,
                    peak_score=peak_score
                )
                if transcript and self.config.smart_boundaries:
                    print(f"[DEBUG] Before refine: transcript len={len(transcript)}, last_end={transcript[-1]['end']:.2f}, moment start={moment.start_time}, end={moment.end_time}", flush=True)
                    moment = self._refine_moment_boundaries(moment, transcript, video_path, video_duration)
                moments.append(moment)
                print(f"[DEBUG] Final moment added: start={moment.start_time:.2f} end={moment.end_time:.2f} duration={moment.duration:.2f}", flush=True)

        print(f"[DEBUG] Before dynamic splitter: moments count={len(moments)}, max_clip_duration={self.config.max_clip_duration}, transcript exists={transcript is not None}")
        # --- Split long moments using natural boundaries (transcript gaps) ---
        if self.config.max_clip_duration > 0 and transcript:
            new_moments = []
            for m in moments:
                if m.duration <= self.config.max_clip_duration:
                    new_moments.append(m)
                else:
                    print(f"[DEBUG] Splitting long moment {m.start_time:.1f}-{m.end_time:.1f} dur={m.duration:.1f}")
                    split_parts = self._split_long_moment(m, transcript)
                    new_moments.extend(split_parts)
            moments = new_moments
        # -----------------------------------------        
        return moments

    def _refine_moment_boundaries(self, moment: ViralMoment, transcript: List[Dict], video_path: Path, video_duration: float) -> ViralMoment:
        if moment.duration > self.config.max_clip_duration:
           print(f"[REFINE] Skipping refinement for moment longer than max_clip_duration ({moment.duration:.1f}s > {self.config.max_clip_duration}s)")
           return moment        
        if not transcript or not self.config.smart_boundaries:
            return moment

        sent_ids = None
        try:
            sent_ids = self._label_sentences_with_spacy(transcript)
        except Exception as e:
            print(f"spaCy failed: {e}, using dynamic gap fallback")

        # Find nearest segment to peak
        best_idx = min(enumerate(transcript), key=lambda x: min(abs(x[1]['start']-moment.peak_time), abs(x[1]['end']-moment.peak_time)))[0]
        peak_idx = best_idx

        # Forward expansion
        end_idx = peak_idx
        while end_idx < len(transcript) - 1:
            curr = transcript[end_idx]
            nxt = transcript[end_idx + 1]
            gap = nxt['start'] - curr['end']
            ends_with_punc = curr['text'].strip().endswith(('.', '!', '?'))
            allowed = self.config.allowed_gap_punc if ends_with_punc else self.config.allowed_gap_no_punc

            merge = False
            if sent_ids is not None:
                if sent_ids[end_idx] == sent_ids[end_idx+1] and gap < self.config.max_gap_for_same_sentence:
                    merge = True
            if not merge:
                next_starts_lower = nxt['text'][0].islower() if nxt['text'] else False
                next_short = (nxt['end'] - nxt['start']) < 0.5
                if gap < allowed or (gap < self.config.look_ahead_max_gap and (not ends_with_punc or next_short) and next_starts_lower):
                    merge = True
                elif not ends_with_punc and self._is_continuation_word(curr['text']):
                    merge = True

            if merge:
                end_idx += 1
            else:
                break
            if transcript[end_idx]['end'] - moment.peak_time > self.config.max_expand_sec:
                break

        # Backward expansion
        start_idx = peak_idx
        while start_idx > 0:
            curr = transcript[start_idx]
            prev = transcript[start_idx - 1]
            gap = curr['start'] - prev['end']
            ends_with_punc = prev['text'].strip().endswith(('.', '!', '?'))
            allowed = self.config.allowed_gap_punc if ends_with_punc else self.config.allowed_gap_no_punc

            merge = False
            if sent_ids is not None:
                if sent_ids[start_idx-1] == sent_ids[start_idx] and gap < self.config.max_gap_for_same_sentence:
                    merge = True
            if not merge:
                curr_starts_lower = curr['text'][0].islower() if curr['text'] else False
                if gap < allowed or (gap < self.config.look_ahead_max_gap and not ends_with_punc and curr_starts_lower):
                    merge = True
                elif not ends_with_punc and self._is_continuation_word(prev['text']):
                    merge = True

            if merge:
                start_idx -= 1
            else:
                break
            if moment.peak_time - transcript[start_idx]['start'] > self.config.max_expand_sec:
                break

        new_start = max(0.0, transcript[start_idx]['start'])
        new_end = min(video_duration, transcript[end_idx]['end'])

        if new_end - new_start > self.config.max_clip_duration:
            half = self.config.max_clip_duration / 2
            new_start = max(0.0, moment.peak_time - half)
            new_end = min(video_duration, moment.peak_time + half)

        moment.start_time = new_start
        moment.end_time = new_end
        moment.duration = new_end - new_start
        # Debug: print overlapping transcript segments for this moment
        overlapping = [seg for seg in transcript if seg['end'] > moment.start_time and seg['start'] < moment.end_time]
        print(f"[BOUNDARY DEBUG] Moment {moment.start_time:.1f}-{moment.end_time:.1f} (peak {moment.peak_time:.1f})")
        for seg in overlapping:
            print(f"    {seg['start']:.2f} -> {seg['end']:.2f}: {seg['text'][:80]}")        
        return moment
    def _split_long_moment(self, moment: ViralMoment, transcript: List[Dict]) -> List[ViralMoment]:
        print(f"[SPLIT] Entering _split_long_moment for moment {moment.start_time:.1f}-{moment.end_time:.1f} (duration {moment.duration:.1f})")
        """Split a moment into smaller pieces at the largest transcript gaps, each ≤ max_clip_duration."""
        max_dur = self.config.max_clip_duration
        if moment.duration <= max_dur:
            return [moment]

        # Collect all transcript segments that overlap with this moment
        overlapping = []
        for seg in transcript:
            if seg['end'] > moment.start_time and seg['start'] < moment.end_time:
                overlapping.append(seg)

        if len(overlapping) < 2:
            # No split points – just cap the clip to max_dur centered on peak
            half = max_dur / 2
            new_start = max(0.0, moment.peak_time - half)
            new_end = min(moment.end_time, new_start + max_dur)
            if new_end - new_start < max_dur:
                new_start = max(0.0, new_end - max_dur)
            moment.start_time = new_start
            moment.end_time = new_end
            moment.duration = new_end - new_start
            return [moment]

        # Find the largest gap between consecutive overlapping segments
        best_gap = 0.0
        best_idx = -1
        for i in range(len(overlapping) - 1):
            gap = overlapping[i+1]['start'] - overlapping[i]['end']
            if gap > best_gap:
                best_gap = gap
                best_idx = i

        if best_gap <= 0.0 or best_idx == -1:
            # No meaningful gap – fallback to fixed split
            return self._split_fixed(moment, max_dur)

        # Split at the middle of the largest gap
        split_time = (overlapping[best_idx]['end'] + overlapping[best_idx+1]['start']) / 2
        # Create left and right moments
        left = ViralMoment(
            start_time=moment.start_time, end_time=split_time, duration=split_time - moment.start_time,
            combined_score=moment.combined_score, tier=moment.tier, method_scores=moment.method_scores,
            confidence=moment.confidence, methods_count=moment.methods_count,
            peak_time=min(moment.peak_time, split_time), peak_score=moment.peak_score,
            explanations=moment.explanations.copy()
        )
        right = ViralMoment(
            start_time=split_time, end_time=moment.end_time, duration=moment.end_time - split_time,
            combined_score=moment.combined_score, tier=moment.tier, method_scores=moment.method_scores,
            confidence=moment.confidence, methods_count=moment.methods_count,
            peak_time=max(moment.peak_time, split_time), peak_score=moment.peak_score,
            explanations=moment.explanations.copy()
        )
        # Recursively split each part if still too long
        result = []
        result.extend(self._split_long_moment(left, transcript))
        result.extend(self._split_long_moment(right, transcript))
        return result

    def _split_fixed(self, moment: ViralMoment, max_dur: float) -> List[ViralMoment]:
        """Fallback: split into fixed‑length chunks (if natural splits fail)."""
        result = []
        start = moment.start_time
        while start < moment.end_time:
            end = min(start + max_dur, moment.end_time)
            sub = ViralMoment(
                start_time=start, end_time=end, duration=end - start,
                combined_score=moment.combined_score, tier=moment.tier,
                method_scores=moment.method_scores.copy(), confidence=moment.confidence,
                methods_count=moment.methods_count, peak_time=start + (end - start)/2,
                peak_score=moment.peak_score, explanations=moment.explanations.copy()
            )
            result.append(sub)
            start = end
        return result
    
    def _rank_moments_with_ml(self, video_path: Path, raw_moments, ml_scores, features_df, all_results):
        moments = []
        hook_result = None
        for dets in all_results.values():
            for r in dets:
                if r.method == DetectionMethod.HOOK_DETECTION:
                    hook_result = r
                    break
            if hook_result:
                break
        for moment in raw_moments:
            start_bin = _round_ts(moment.start_time)
            end_bin = _round_ts(moment.end_time)
            mask = (features_df['timestamp'].round(2) >= start_bin) & (features_df['timestamp'].round(2) <= end_bin)
            overlapping = ml_scores[mask]
            avg_ml = np.mean(overlapping) if len(overlapping) > 0 else 50.0
            blended = (1.0 - self.config.ml_blending_ratio) * moment.combined_score + self.config.ml_blending_ratio * avg_ml
            if self.config.fusion_enabled:
                high_count = 0
                for ts in np.unique(features_df['timestamp'].round(2)[mask]):
                    dets = all_results.get(float(ts), [])
                    for d in dets:
                        if d.score >= self.config.fusion_high_threshold:
                            high_count += 1
                if high_count >= 2:
                    blended = min(blended * self.config.fusion_boost_factor, 100.0)
            if self.pattern_memory:
                moment_features = []
                if mask.any():
                    feat_df = features_df.loc[mask, [c for c in features_df.columns if c != 'timestamp']]
                    if len(feat_df) > 0:
                        moment_features = feat_df.select_dtypes(include=[np.number]).mean().tolist()
                if moment_features:
                    boost = self.pattern_memory.find_similar_boost(moment_features)
                    blended *= boost
            if moment.start_time < 3.0 and hook_result and hook_result.score < self.config.hook_penalty_threshold:
                blended *= 0.7
            if blended >= self.config.tier_a_threshold:
                tier = ViralityTier.TIER_A
            elif blended >= self.config.tier_b_threshold:
                tier = ViralityTier.TIER_B
            else:
                tier = ViralityTier.TIER_C
            moment.combined_score = blended
            moment.tier = tier
            moment.explanations = self._generate_explanations(moment) 
            moments.append(moment)
        if self.pattern_memory:
            video_name = video_path.stem
            for moment in moments:
                if moment.combined_score >= self.config.pattern_min_score:
                    start_bin = _round_ts(moment.start_time)
                    end_bin = _round_ts(moment.end_time)
                    mask = (features_df['timestamp'].round(2) >= start_bin) & (features_df['timestamp'].round(2) <= end_bin)
                    if mask.any():
                        feat_df = features_df.loc[mask, [c for c in features_df.columns if c != 'timestamp']]
                        if len(feat_df) > 0:
                            feat_vec = feat_df.select_dtypes(include=[np.number]).mean().tolist()
                            self.pattern_memory.add_pattern(feat_vec, moment.combined_score, video_name, moment.start_time, self.config.pattern_min_score)
        moments.sort(key=lambda x: x.combined_score, reverse=True)
        return moments