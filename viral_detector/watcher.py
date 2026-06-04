# viral_detector/watcher.py
import logging
from .config import DetectionMethod, ViralityTier
import os
import sys
import time
import json
import queue
import threading
import signal
from pathlib import Path
from typing import List, Dict
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .config import DetectionConfig, ContentProfile
from .models import ProcessingResult
from .utils import LoggerFactory, DetectorBase
from .analyzer import MomentAnalyzer
from .cutter import VideoCutter
 
class ProcessedVideosStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self._data = {}
        self.logger = LoggerFactory.create(__name__)
        self.load()

    def load(self):
        if self.store_path.exists():
            try:
                with open(self.store_path) as f:
                    self._data = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load state: {e}")

    def save(self):
        try:
            with open(self.store_path, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Could not save state: {e}")

    def is_processed(self, video_path: Path) -> bool:
        return str(video_path.resolve()) in self._data

    def mark_processed(self, result: ProcessingResult) -> None:
        resolved = result.video_path.resolve()
        self._data[str(resolved)] = result.__dict__
        self.save()

class VideoWatcher(FileSystemEventHandler):
    def __init__(self, analyzer: MomentAnalyzer, cutter: VideoCutter,
                 config: DetectionConfig, store: ProcessedVideosStore):
        self.analyzer = analyzer
        self.cutter = cutter
        self.config = config
        self.store = store
        self.logger = LoggerFactory.create(__name__)
        self.video_queue = queue.Queue(maxsize=10)
        self.worker_thread = None
        self.running = False
        self.processing = set()
        self.processing_lock = threading.Lock()
        self.start_worker()
        self._process_existing()

    def _process_existing(self):
        videos = self._find_videos(self.config.watch_folder, recursive=True)
        if videos:
            for v in sorted(videos):
                try:
                    self.video_queue.put_nowait(v)
                except queue.Full:
                    self.logger.warning(f"Queue full, skipping {v.name}")
        else:
            self.logger.info("No existing videos")

    def _find_videos(self, folder: Path, recursive: bool = False) -> List[Path]:
        videos = []
        if recursive:
            for ext in self.config.supported_formats:
                videos.extend(folder.rglob(f"*{ext}"))
                videos.extend(folder.rglob(f"*{ext.upper()}"))
        else:
            for ext in self.config.supported_formats:
                videos.extend(folder.glob(f"*{ext}"))
                videos.extend(folder.glob(f"*{ext.upper()}"))
        output_names = {'Cut', 'ViralClips', 'Converted', 'temp', '.temp'}
        filtered = []
        for v in videos:
            try:
                v.relative_to(self.config.output_folder)
                continue
            except ValueError:
                pass
            if any(p.name in output_names for p in v.parents):
                continue
            filtered.append(v)
        return filtered

    def on_created(self, event):
        if event.is_directory:
            new_folder = Path(str(event.src_path))
            self.logger.info(f"New folder: {new_folder.name}")
            videos = self._find_videos(new_folder, recursive=False)
            for v in videos:
                try:
                    self.video_queue.put_nowait(v)
                except queue.Full:
                    self.logger.warning(f"Queue full, skipping folder video")
            return
        file_path = Path(str(event.src_path))
        if file_path.suffix.lower() in self.config.supported_formats:
            self.logger.info(f"New video: {file_path.name}")
            try:
                self.video_queue.put_nowait(file_path)
            except queue.Full:
                self.logger.warning(f"Queue full, skipping {file_path.name}")

    def on_moved(self, event):
        if event.is_directory:
            return
        file_path = Path(str(event.dest_path))
        if file_path.suffix.lower() in self.config.supported_formats:
            self.logger.info(f"Moved video: {file_path.name}")
            try:
                self.video_queue.put_nowait(file_path)
            except queue.Full:
                self.logger.warning(f"Queue full, skipping moved {file_path.name}")

    def on_modified(self, event):
        if event.is_directory:
            return
        file_path = Path(str(event.src_path))
        if file_path.suffix.lower() in self.config.supported_formats:
            with self.processing_lock:
                already = str(file_path) in self.processing
            if not already:
                self.logger.info(f"Modified video: {file_path.name}")
                try:
                    self.video_queue.put_nowait(file_path)
                except queue.Full:
                    self.logger.warning(f"Queue full, skipping modified {file_path.name}")

    def _worker(self):
        self.logger.info("Worker started")
        while self.running:
            try:
                video_path = self.video_queue.get(timeout=1)
                self.logger.info(f"Processing: {video_path.name}")
                time.sleep(2)
                initial = video_path.stat().st_size
                time.sleep(1)
                final = video_path.stat().st_size
                wait = 0
                while initial != final and wait < 10:
                    time.sleep(1)
                    final = video_path.stat().st_size
                    wait += 1
                self._process_video(video_path)
                self.video_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                if 'video_path' in locals():
                    self.video_queue.task_done()

    def start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def _process_video(self, video_path: Path):
        video_path = video_path.resolve()
        with self.processing_lock:
            if str(video_path) in self.processing:
                return
            self.processing.add(str(video_path))
        try:
            if self.store.is_processed(video_path):
                self.logger.info(f"Already processed: {video_path.name}")
                return
            start = time.time()
            self.logger.info("="*80)
            self.logger.info(f"Processing: {video_path.name} (mode: {self.config.processing_mode})")
            self.logger.info("="*80)
            moments, transcript = self.analyzer.analyze_video(video_path)
            tier_breakdown = {
                'A': sum(1 for m in moments if m.tier == ViralityTier.TIER_A),
                'B': sum(1 for m in moments if m.tier == ViralityTier.TIER_B),
                'C': sum(1 for m in moments if m.tier == ViralityTier.TIER_C),
            }
            clips = self.cutter.cut_moments(video_path, moments, transcript=transcript, video_duration=None)
            elapsed = time.time() - start
            self.logger.info(f"Results: A:{tier_breakdown['A']} B:{tier_breakdown['B']} C:{tier_breakdown['C']}")
            self.logger.info(f"Complete: {len(clips)} clips in {elapsed:.1f}s")
            result = ProcessingResult(video_path=video_path, moments_detected=sum(tier_breakdown.values()),
                                      clips_created=len(clips), tier_breakdown=tier_breakdown,
                                      processing_time=elapsed)
            self.store.mark_processed(result)
            DetectorBase.clear_audio_cache()
        except Exception as e:
            self.logger.error(f"Processing failed: {e}", exc_info=True)
        finally:
            with self.processing_lock:
                self.processing.discard(str(video_path))

class ViralDetectorApp:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._setup_directories()
        self.logger = LoggerFactory.create(__name__)
        self.analyzer = MomentAnalyzer(config)
        self.cutter = VideoCutter(config)
        self.store = ProcessedVideosStore(config.watch_folder / config.state_file)
        self.observer = None
        self._shutdown_event = threading.Event()
        self._print_banner()
        self._setup_signal_handlers()
        self._warmup()

    def _setup_directories(self):
        self.config.watch_folder.mkdir(parents=True, exist_ok=True)
        self.config.output_folder.mkdir(parents=True, exist_ok=True)

    def _print_banner(self):
        banner = r"""
╔════════════════════════════════════════════════════════════════╗
║  VIRAL MOMENT DETECTOR - AI Video Analysis Engine           ║
║     Detects & ranks viral moments into A, B, C tiers        ║
╚════════════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info(f"Watch: {self.config.watch_folder}")
        self.logger.info(f"Output: {self.config.output_folder}")
        self.logger.info(f"Clip duration: {self.config.clip_duration}s")
        print("-"*68)

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()
        self.stop()

    def _warmup(self):
        def warmup():
            time.sleep(1)
            self.logger.debug("Warm‑up: pre‑loading models if any")
        threading.Thread(target=warmup, daemon=True).start()

    def start(self, watch_mode: bool = True):
        if watch_mode:
            self._watch_folder()
        else:
            self._process_once()

    def _watch_folder(self):
        self.watcher = VideoWatcher(self.analyzer, self.cutter, self.config, self.store)
        self.observer = Observer()
        self.observer.schedule(self.watcher, str(self.config.watch_folder), recursive=True)
        self.observer.start()
        self.logger.info("Watcher started. Press Ctrl+C to stop.")
        try:
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self._shutdown_event.set()
        finally:
            self.stop()

    def _process_once(self):
        import sys
        self.logger.info("Processing all videos once...")
        output_resolved = self.config.output_folder.resolve()
        videos = []
        for ext in self.config.supported_formats:
            for p in self.config.watch_folder.rglob(f"*{ext}"):
                try:
                    if p.resolve().is_relative_to(output_resolved): 
                        continue
                except ValueError:
                    if str(p.resolve()).startswith(str(output_resolved)):
                        continue
                videos.append(p)
            for p in self.config.watch_folder.rglob(f"*{ext.upper()}"):
                try:
                    if p.resolve().is_relative_to(output_resolved):
                        continue
                except ValueError:
                    if str(p.resolve()).startswith(str(output_resolved)):
                        continue
                videos.append(p)
        for vpath in videos:
            try:
                self.logger.info(f"Processing {vpath.name}")
                moments, transcript = self.analyzer.analyze_video(vpath)
                self.logger.info(f"Analysis done, found {len(moments)} moments")
                if moments:
                    self.logger.info("Calling cutter.cut_moments...")
                    clips = self.cutter.cut_moments(vpath, moments, transcript=transcript, video_duration=None)
                    self.logger.info(f"Done: {len(moments)} moments, {len(clips)} clips created")
                else:
                    self.logger.info("No moments, skipping cut")
            except Exception as e:
                self.logger.error(f"Failed {vpath.name}: {e}")
        self.logger.info("Processing complete!")
        self._shutdown_event.set()
        sys.exit(0)

    def stop(self):
        if hasattr(self, 'watcher') and self.watcher:
            self.watcher.stop()
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.logger.info("Stopped")

def get_default_paths():
    home = Path.home()
    default_watch = home / "Videos" / "WatchFolder"
    default_output = home / "Videos" / "ViralClips"
    ffmpeg = shutil.which("ffmpeg") or r"C:\Users\Mario\Desktop\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
    ffprobe = shutil.which("ffprobe") or r"C:\Users\Mario\Desktop\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"
    return default_watch, default_output, Path(ffmpeg), Path(ffprobe)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Viral Moment Detector")
    parser.add_argument("--watch", type=str, help="Watch folder")
    parser.add_argument("--output", type=str, help="Output folder")
    parser.add_argument("--duration", type=int, default=15, help="Clip duration (s)")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    parser.add_argument("--methods", nargs='+', choices=['audio','motion','scene','silence','face','object',
                         'joke','argument','emotional','shocking','video','narrative','hook'], help="Methods to enable")
    parser.add_argument("--ml-model", type=str, help="Trained ML model path")
    parser.add_argument("--ffmpeg-path", type=str, help="FFmpeg path")
    parser.add_argument("--ffprobe-path", type=str, help="FFprobe path")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument("--mode", choices=['fast','balanced'], default='balanced', help="Processing mode")
    parser.add_argument("--profile", choices=["podcast", "gaming", "reaction"], help="Content profile for scoring")
    parser.add_argument("--content-profile", choices=["podcast", "gaming", "reaction"], help="Content profile for scoring")
    parser.add_argument("--add-hook", type=str, help="Add hook keyword")
    parser.add_argument("--vertical-export", action="store_true", help="Create square (1:1) clips")
    args = parser.parse_args()

    config_file = Path("config.json") 
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    default_watch, default_output, default_ffmpeg, default_ffprobe = get_default_paths()

    if args.watch:
        watch_path = Path(args.watch)
        print(f"Using CLI watch folder: {watch_path}")
    elif config_data.get('watch_folder'):
        watch_path = Path(config_data['watch_folder'])
        print(f"Using config.json watch folder: {watch_path}")
    else:
        watch_path = default_watch
        print(f"Using default watch folder: {watch_path}")

    if args.output:
        output_path = Path(args.output)
        print(f"Using CLI output folder: {output_path}")
    elif config_data.get('output_folder'):
        output_path = Path(config_data['output_folder'])
        print(f"Using config.json output folder: {output_path}")
    else:
        output_path = default_output
        print(f"Using default output folder: {output_path}")

    if args.ffmpeg_path:
        ffmpeg_path = Path(args.ffmpeg_path)
    elif config_data.get('ffmpeg_path'):
        ffmpeg_path = Path(config_data['ffmpeg_path'])
    else:
        ffmpeg_path = default_ffmpeg

    if args.ffprobe_path:
        ffprobe_path = Path(args.ffprobe_path)
    elif config_data.get('ffprobe_path'):
        ffprobe_path = Path(config_data['ffprobe_path'])
    else:
        ffprobe_path = default_ffprobe

    LoggerFactory.configure_root(Path("viral_detector.log"))
    # --- Content profile handling ---
    profile_enum = None
    if args.content_profile == "podcast":
        profile_enum = ContentProfile.PODCAST
    elif args.content_profile == "gaming":
        profile_enum = ContentProfile.GAMING
    elif args.content_profile == "reaction":
        profile_enum = ContentProfile.REACTION

    config = DetectionConfig(
        watch_folder=watch_path,
        output_folder=output_path,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        clip_duration=args.duration or config_data.get('clip_duration', 15),
        ml_model_path=args.ml_model or config_data.get('ml_model_path'),
        use_gpu=args.use_gpu or config_data.get('use_gpu', False),
        enable_profiling=args.profile,   # original boolean flag
        processing_mode=args.mode or config_data.get('processing_mode', 'balanced'),
        tier_a_threshold=config_data.get('tier_a_threshold', 80.0),
        tier_b_threshold=config_data.get('tier_b_threshold', 50.0),
        tier_c_threshold=config_data.get('tier_c_threshold', 20.0),
        content_profile=profile_enum,
        vertical_export=args.vertical_export 
    )
    config.apply_mode()
    config.validate()
    if args.methods:
        config.enabled_methods = set()
        method_map = {
            'audio': DetectionMethod.AUDIO_ENERGY, 'motion': DetectionMethod.MOTION, 'scene': DetectionMethod.SCENE_CHANGES,
            'silence': DetectionMethod.SILENCE_DETECTION, 'face': DetectionMethod.FACE_EMOTION, 'object': DetectionMethod.OBJECT_DETECTION,
            'joke': DetectionMethod.JOKE_DETECTION, 'argument': DetectionMethod.ARGUMENT_DETECTION,
            'emotional': DetectionMethod.EMOTIONAL_PHRASE, 'shocking': DetectionMethod.SHOCKING_STATEMENT,
            'video': DetectionMethod.VIDEO_UNDERSTANDING, 'narrative': DetectionMethod.NARRATIVE_ANALYSIS,
            'hook': DetectionMethod.HOOK_DETECTION
        }
        for m in args.methods:
            config.enabled_methods.add(method_map[m])
    else:
        logger = logging.getLogger("main")
        logger.info(f"Mode: {config.processing_mode}, methods: {len(config.enabled_methods)}")

    if args.add_hook:
        hooks_file = Path("viral_hooks.json")
        try:
            hooks = json.loads(hooks_file.read_text()) if hooks_file.exists() else []
            if args.add_hook not in hooks:
                hooks.append(args.add_hook.lower())
                hooks_file.write_text(json.dumps(hooks, indent=2))
                print(f"Added hook '{args.add_hook}'")
            else:
                print(f"Hook already exists")
        except Exception as e:
            print(f"Error: {e}")
        return

    app = ViralDetectorApp(config)
    app.start(watch_mode=not args.once)

if __name__ == "__main__":
    main()