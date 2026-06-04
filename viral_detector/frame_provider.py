import threading
import queue
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Any, Optional

class FrameProvider:
    def __init__(self, video_path: Path, fps: float, total_frames: int):
        self.video_path = video_path
        self.fps = fps
        self.total_frames = total_frames
        self.consumers = []
        self._stop = threading.Event()
        self._thread = None

    def register_consumer(self, name: str, sample_rate: int, preprocess_fn: Callable, queue_size: int = 2000):
        """Register a detector consumer.
        name: unique name (e.g., 'motion', 'face', 'object', 'scene')
        sample_rate: process every Nth frame (1 = every frame)
        preprocess_fn: callable(frame, frame_idx, timestamp) -> any (the data to send to detector)
        Returns a queue that the detector can read from.
        """
        q = queue.Queue(maxsize=queue_size)
        self.consumers.append({
            'name': name,
            'sample_rate': sample_rate,
            'preprocess_fn': preprocess_fn,
            'queue': q
        })
        return q

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            for consumer in self.consumers:
                consumer['queue'].put(None)
            return
        frame_idx = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / self.fps
            for consumer in self.consumers:
                if frame_idx % consumer['sample_rate'] == 0:
                    data = consumer['preprocess_fn'](frame, frame_idx, timestamp)
                    if data is not None:
                        # Block until space is available – never skip frames
                        consumer['queue'].put(data)
                        if frame_idx % 100 == 0:
                            print(f"[FrameProvider] Sent frame {frame_idx} to {consumer['name']}", flush=True)
            frame_idx += 1
        print(f"[FrameProvider] Total frames processed: {frame_idx}", flush=True)
        for consumer in self.consumers:
            consumer['queue'].put(None)
        cap.release()