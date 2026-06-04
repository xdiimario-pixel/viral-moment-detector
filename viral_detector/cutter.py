# viral_detector/cutter.py
import json
import subprocess
import bisect
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DetectionConfig, ViralityTier
from .models import ViralMoment
from .utils import LoggerFactory, VideoMetrics

class VideoCutter:
    _keyframe_cache = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = LoggerFactory.create(__name__)

    def _get_keyframes(self, video_path: Path) -> List[float]:
        video_path = video_path.resolve()
        cache_key = str(video_path)
        with self._cache_lock:
            if cache_key in self._keyframe_cache:
                return self._keyframe_cache[cache_key]
        cmd = [str(self.config.ffprobe_path), '-v', 'error', '-select_streams', 'v:0',
               '-show_frames', '-show_entries', 'frame=pkt_pts_time,key_frame', '-of', 'csv', str(video_path)]
        keyframes = []
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True)
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 3 and parts[1] == '1':
                    try:
                        keyframes.append(float(parts[2]))
                    except ValueError:
                        continue
            keyframes.sort()
            with self._cache_lock:
                self._keyframe_cache[cache_key] = keyframes
        except Exception as e:
            self.logger.error(f"Keyframe extraction failed: {e}")
            # Fallback: every 2 seconds
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0.0
                cap.release()
            else:
                duration = 0.0
            keyframes = [float(i * 2.0) for i in range(int(duration // 2) + 1)]
            with self._cache_lock:
                self._keyframe_cache[cache_key] = keyframes
        return keyframes

    def _find_nearest_keyframe(self, keyframes: List[float], timestamp: float) -> float:
        if not keyframes:
            return timestamp
        idx = bisect.bisect_left(keyframes, timestamp)
        if idx == 0:
            return keyframes[0]
        if idx == len(keyframes):
            return keyframes[-1]
        before = keyframes[idx-1]
        after = keyframes[idx]
        return before if (timestamp - before) <= (after - timestamp) else after

    def _build_ffmpeg_command(self, video_path: Path, clip_start: float, clip_end: float,
                              output_path: Path, keyframes: List[float], tolerance: float = 0.1) -> List[str]:
        nearest = self._find_nearest_keyframe(keyframes, clip_start)
        distance = abs(clip_start - nearest)
        ffmpeg = str(self.config.ffmpeg_path)
        duration = clip_end - clip_start
        if distance <= tolerance:
            return [ffmpeg, '-ss', str(clip_start), '-i', str(video_path), '-t', str(duration),
                    '-c', 'copy', '-avoid_negative_ts', 'make_zero', str(output_path)]
        else:
            offset = clip_start - nearest
            return [ffmpeg, '-ss', str(nearest), '-i', str(video_path), '-ss', str(offset), '-t', str(duration),
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k',
                    '-avoid_negative_ts', 'make_zero', str(output_path)]

    def cut_moments(self, video_path: Path, moments: List[ViralMoment], transcript: Optional[List[Dict]] = None, video_duration: Optional[float] = None, precise_cut: bool = False) -> List[Path]:
        import json
        from datetime import datetime
        from concurrent.futures import ThreadPoolExecutor, as_completed
        video_path = video_path.resolve()
        output_paths = []
        video_name = video_path.stem
 
        # Get duration using OpenCV 
        if video_duration is None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = frame_count / fps if fps > 0 else 0.0
                cap.release()
            else:
                video_duration = 0.0

        keyframes = self._get_keyframes(video_path)
        tier_durations = {ViralityTier.TIER_A: 30.0, ViralityTier.TIER_B: 22.0, ViralityTier.TIER_C: 15.0}
        base_dir = self.config.output_folder / video_name
        for t in [ViralityTier.TIER_A, ViralityTier.TIER_B, ViralityTier.TIER_C]:
            (base_dir / t.value).mkdir(parents=True, exist_ok=True)

        # List to store info for caption burning (only for burned subtitles, not SRT)
        caption_tasks = []  # (output_path, shift, overlapping_segments)

        tasks = []
        for idx, moment in enumerate(moments, 1):
            # ---------- Clipping logic: respect refined boundaries when smart_boundaries is enabled ----------
            if self.config.smart_boundaries and moment.duration > 0:
                # Start with the refined boundaries
                clip_start = moment.start_time
                clip_end = moment.end_time
                # Ensure within video bounds
                clip_start = max(0.0, clip_start)
                clip_end = min(video_duration, clip_end)
                
                # Optionally, if the refined clip is shorter than the target duration, expand a bit
                target_dur = tier_durations.get(moment.tier, 15.0)
                current_dur = clip_end - clip_start
                if current_dur < target_dur:
                    # Expand symmetrically, but keep the original peak in the middle
                    delta = target_dur - current_dur
                    # Limit expansion to avoid excessive length (e.g., no more than 30% of target)
                    delta = min(delta, target_dur * 0.3)
                    clip_start = max(0.0, clip_start - delta/2)
                    clip_end = min(video_duration, clip_end + delta/2)
            else:
                # Original fallback for when smart boundaries are disabled
                target_dur = tier_durations.get(moment.tier, 15.0)
                if self.config.smart_boundaries and moment.duration <= target_dur + 2.0:
                    clip_start = max(0.0, moment.start_time)
                    clip_end = min(video_duration, moment.end_time)
                    if clip_end - clip_start < target_dur:
                        delta = target_dur - (clip_end - clip_start)
                        clip_start = max(0.0, clip_start - delta/2)
                        clip_end = min(video_duration, clip_end + delta/2)
                else:
                    buffer = 2.0
                    start_buf = max(0, moment.start_time - buffer)
                    end_buf = min(video_duration, moment.end_time + buffer)
                    buf_dur = end_buf - start_buf
                    clip_start = max(0, start_buf + (buf_dur - target_dur)/2)
                    clip_end = min(video_duration, clip_start + target_dur)
                clip_start = max(0.0, clip_start)
                clip_end = min(video_duration, clip_end)
                if clip_end - clip_start < 0.5:
                    peak = moment.start_time + moment.duration/2
                    clip_start = max(0.0, peak - 0.5)
                    clip_end = min(video_duration, peak + 0.5)

            # Safety: ensure positive duration
            if clip_end - clip_start < 0.1:
                clip_start = max(0.0, moment.start_time)
                clip_end = min(video_duration, moment.end_time)
            print(f"[CUTTER] Moment {idx}: start={moment.start_time:.2f}, end={moment.end_time:.2f} -> clip {clip_start:.2f} to {clip_end:.2f}", flush=True)
            # ---------- Filename and metadata ----------
            filename = f"{video_name}_{moment.tier.value}_{idx:03d}.mp4"
            output_path = base_dir / moment.tier.value / filename
            metadata = {"video": video_name, "clip": filename, "tier": moment.tier.value,
                        "score": moment.combined_score, "start_time": moment.start_time,
                        "end_time": moment.end_time, "duration": moment.duration,
                        "method_scores": moment.method_scores, "timestamp": datetime.now().isoformat()}
            with open(output_path.with_suffix(".json"), 'w') as f:
                json.dump(metadata, f, indent=2)

            # ---------- Caption preparation ----------
            overlap = None
            if self.config.add_captions and transcript:
                overlapping = [seg for seg in transcript if seg['end'] > clip_start and seg['start'] < clip_end]
                if overlapping:
                    overlap = overlapping
                    if self.config.caption_export_srt:
                        # Write external SRT file immediately (does not depend on video)
                        srt_content = self._generate_srt(overlapping, shift=clip_start)
                        srt_path = output_path.with_suffix(".srt")
                        srt_path.write_text(srt_content, encoding='utf-8')
                    else:
                        # Store for later burning after ffmpeg creates the video
                        caption_tasks.append((output_path, clip_start, overlapping))
            # ---------- Build ffmpeg command ----------
            cmd = self._build_ffmpeg_command(video_path, clip_start, clip_end, output_path, keyframes)
            tasks.append((cmd, output_path, idx, moment))

        # ---------- Execute ffmpeg tasks in parallel (cutting clips) ----------
        with ThreadPoolExecutor(max_workers=self.config.max_cutter_workers) as ex:
            futures = {ex.submit(self._run_ffmpeg_cmd, cmd, out, idx, moment): out for cmd, out, idx, moment in tasks}
            for f in as_completed(futures):
                out = f.result()
                if out:
                    output_paths.append(out)
                    # If vertical export enabled, crop to square
                    if self.config.vertical_export:
                        square_path = self._crop_to_square(out)
                        if square_path:
                            output_paths.append(square_path)

        # ---------- Burn captions sequentially (for burned‑in subtitles) ----------
        if not self.config.caption_export_srt and caption_tasks:
            for out_path, shift, segments in caption_tasks:
                if out_path in output_paths:  # only if the clip was successfully created
                    self._add_captions_to_clip(out_path, segments, shift=shift)

        return output_paths

    def _run_ffmpeg_cmd(self, cmd: List[str], output_path: Path, idx: int, moment: ViralMoment) -> Optional[Path]:
        import subprocess
        # Parse original command to get input, start, duration
        input_file = None
        start_time = None
        duration = None
        for i, arg in enumerate(cmd):
            if arg == '-i' and i+1 < len(cmd):
                input_file = cmd[i+1]
            if arg == '-ss' and i+1 < len(cmd):
                start_time = cmd[i+1]
            if arg == '-t' and i+1 < len(cmd):
                duration = cmd[i+1]
        if not input_file or start_time is None or duration is None:
            self.logger.error(f"Could not parse ffmpeg command: {cmd}")
            return None
        new_cmd = [
            str(self.config.ffmpeg_path),
            '-i', input_file,
            '-ss', start_time,
            '-t', duration,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            str(output_path)
        ]
        self.logger.info(f"Running ffmpeg (re-encode): {' '.join(new_cmd)}")
        process = subprocess.Popen(new_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = process.communicate(timeout=300)
            if process.returncode == 0:
                self.logger.info(f"Created: {output_path.name}")
                return output_path
            else:
                self.logger.error(f"FFmpeg failed (code {process.returncode}): {stderr}")
                return None
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            self.logger.error(f"FFmpeg timeout after 300s for {output_path.name}")
            return None

    def _has_subtitle_stream(self, video_path: Path) -> bool:
        cmd = [str(self.config.ffprobe_path), "-v", "error", "-select_streams", "s",
               "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
            return bool(result.stdout.strip())
        except:
            return False

    def _generate_ass(self, segments: List[Dict], shift: float = 0.0) -> str:
        style = self.config.caption_style
        ass = f"""[Script Info]
Title: ViralDetector Captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.get('font','Arial')},{style.get('fontsize','24')},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{style.get('borderw','1')},0,{style.get('alignment','2')},10,10,10,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        def ass_time(t):
            h = int(t//3600)
            m = int((t%3600)//60)
            s = t%60
            return f"{h}:{m:02d}:{s:06.2f}"
        for seg in segments:
            start = seg['start'] - shift
            end = seg['end'] - shift
            if start < 0: start = 0
            if end < 0: end = 0
            text = seg['text'].strip()
            if not text:
                continue
            ass += f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{text}\n"
        return ass

    def _generate_srt(self, segments: List[Dict], shift: float = 0.0) -> str:
        """Generate SRT subtitle string with timestamps shifted by `-shift`."""
        def fmt_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            ms = int((seconds - int(seconds)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        lines = []
        for i, seg in enumerate(segments, 1):
            start = seg['start'] - shift
            end = seg['end'] - shift
            if start < 0: start = 0
            if end < 0: end = 0
            lines.append(str(i))
            lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
            lines.append(seg['text'].strip())
            lines.append("")
        return "\n".join(lines)

    def _add_captions_to_clip(self, video_path: Path, segments: List[Dict], shift: float = 0.0) -> Path:
        """Burn captions into video using shifted ASS file."""
        ass_content = self._generate_ass(segments, shift=shift)
        ass_path = video_path.with_suffix(".ass")
        ass_path.write_text(ass_content, encoding='utf-8')
        temp_path = video_path.with_stem(video_path.stem + "_temp")
        cmd = [str(self.config.ffmpeg_path), "-i", str(video_path),
               "-vf", f"subtitles={ass_path.name}", "-c:a", "copy", "-y", str(temp_path)]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            video_path.unlink()
            temp_path.rename(video_path)
            self.logger.info(f"Added captions to {video_path.name}")
        except Exception as e:
            self.logger.error(f"Caption burn failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
        finally:
            if ass_path.exists():
                ass_path.unlink()
        return video_path 

    def _validate_clip(self, path: Path) -> bool:
        """Return True if file exists, size > 0, and duration > 0.1s."""
        if not path.exists():
            return False 
        if path.stat().st_size == 0:
            return False
        # Check duration via OpenCV
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        return duration > 0.1    
    
    def _crop_to_square(self, video_path: Path) -> Optional[Path]:
        """Crop the video to 1:1 square (center crop). Returns path to new square video or None if failed."""
        import subprocess
        import cv2
        # Get video dimensions
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.logger.error(f"Cannot open {video_path} for cropping")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Determine crop size (square = min(width, height))
        crop_size = min(width, height)
        x_offset = (width - crop_size) // 2
        y_offset = (height - crop_size) // 2

        output_path = video_path.with_stem(video_path.stem + "_square")
        cmd = [
            str(self.config.ffmpeg_path),
            "-i", str(video_path),
            "-vf", f"crop={crop_size}:{crop_size}:{x_offset}:{y_offset}",
            "-c:a", "copy",
            "-y", str(output_path)
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            self.logger.info(f"Created square version: {output_path.name}")
            return output_path
        except Exception as e:
            self.logger.error(f"Square crop failed: {e}")
            return None    