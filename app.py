# app.py
import streamlit as st
from pathlib import Path
import sys
import time
import numpy as np
import cv2 
import os
import shutil
from datetime import datetime
from viral_detector import DetectionConfig, MomentAnalyzer, VideoCutter, ContentProfile
from typing import Dict, List, Optional, Tuple

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from viral_detector import DetectionConfig, MomentAnalyzer, VideoCutter
from viral_detector.utils import LoggerFactory

def get_ffmpeg_path():
    # 1. Check environment variable
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path))
    # 2. Check system PATH
    which_path = shutil.which("ffmpeg")
    if which_path:
        return which_path
    # 3. Fallback to common install locations (optional)
    return "ffmpeg"  # hope it's in PATH

def get_ffprobe_path():
    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and Path(env_path).exists():
        return str(Path(env_path))
    which_path = shutil.which("ffprobe")
    if which_path:
        return which_path
    return "ffprobe"


def show_srt_captions(srt_path: Path):
    """Display SRT captions as formatted text."""
    if not srt_path.exists():
        return
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    captions = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, numeric indexes, and timestamps
        if line and not line[0].isdigit() and '-->' not in line:
            captions.append(line)
    if captions:
        with st.expander("📝 Captions (click to expand)"):
            for cap in captions:
                st.markdown(f"• {cap}")

st.set_page_config(page_title="Viral Detector Pro", layout="wide", page_icon="🔥")



st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin-bottom: 2rem;
    border: 1px solid #334155;
}
.hero h1 {
    color: #f1f5f9 !important;
    margin: 0;
    font-size: 2.2rem;
}
.hero p {
    color: #cbd5e1 !important;
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
</style>

<div class="hero">
    <h1>🔥 Viral Moment Detector Pro</h1>
    <p>AI‑powered video analysis – find the moments that go viral</p>
    <p style="font-size: 0.9rem;">Podcasts · Interviews · Gaming · Reactions</p>
</div>
""", unsafe_allow_html=True)

# Feature cards (improved: lighter background, border, subtle shadow)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
        <div style='background: #1e293b; border: 1px solid #4a5568; border-radius: 0.75rem; padding: 1.25rem; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
            <h3 style='color: #f1f5f9; margin: 0 0 0.5rem 0;'>🎯 9+ Detectors</h3>
            <p style='color: #cbd5e1; margin: 0;'>Audio, motion, faces, objects, speech emotions.</p>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
        <div style='background: #1e293b; border: 1px solid #4a5568; border-radius: 0.75rem; padding: 1.25rem; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
            <h3 style='color: #f1f5f9; margin: 0 0 0.5rem 0;'>✂️ Smart Clips</h3>
            <p style='color: #cbd5e1; margin: 0;'>≤60s clips split naturally at pauses.</p>
        </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
        <div style='background: #1e293b; border: 1px solid #4a5568; border-radius: 0.75rem; padding: 1.25rem; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);'>
            <h3 style='color: #f1f5f9; margin: 0 0 0.5rem 0;'>📝 Auto Captions</h3>
            <p style='color: #cbd5e1; margin: 0;'>External .SRT subtitles.</p>
        </div>
    """, unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'last_run_video' not in st.session_state:
    st.session_state.last_run_video = None
if 'last_run_time' not in st.session_state:
    st.session_state.last_run_time = None
if 'last_summary' not in st.session_state:
    st.session_state.last_summary = None

# ==================== SIDEBAR ====================
with st.sidebar:
    watch_folder = st.text_input("Watch Folder", value=r"C:\Users\Mario\Desktop\viral_in")
    output_folder = st.text_input("Output Folder", value=r"C:\Users\Mario\Desktop\viral_out")
    mode = st.selectbox("Mode", ["fast", "balanced"], index=1)
    content_profile = st.selectbox("Content Profile", ["None", "podcast", "gaming", "reaction"], index=0)
    use_gpu = st.checkbox("Use GPU", False)

    # Caption settings (only for balanced/full)
    if mode in ("balanced", "full"):
        with st.expander("🎬 Caption Settings"):
            add_captions = st.checkbox("Add external SRT captions", value=False)
            st.caption("Creates .srt subtitle file next to each clip.")
    else:
        add_captions = False

    # Clip settings (max duration)
    with st.expander("✂️ Clip Settings"):
        max_clip_duration = st.slider("Maximum clip length (seconds)", min_value=10, max_value=300, value=60, step=5)
        st.caption("Clips are split naturally at transcript gaps, never exceeding this length.")
    with st.expander("📱 Vertical Export (TikTok/Reels/Shorts)"):
        vertical_export = st.checkbox("Create square (1:1) clips", value=False)
        if vertical_export:
            st.caption("Original clips will be center‑cropped to square.")        



# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["📹 Select Videos", "🎬 Results", "📊 Status & Summary"])

# ==================== TAB 1: SELECT VIDEOS ====================
with tab1:
    st.subheader("Video Selection")
    watch_path = Path(watch_folder)
    video_names = []
    if watch_path.exists():
        for f in watch_path.iterdir():
            if f.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"] and f.is_file():
                video_names.append(f.name)
        if not video_names:
            st.warning("No video files found.")
    else:
        st.error(f"Watch folder does not exist: {watch_folder}")

    if video_names:
        selected_name = st.selectbox("Choose a video", video_names, index=0)
        if selected_name:
            selected_path = str(watch_path / selected_name)
            st.caption(f"Selected: {selected_name}")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    start_analysis = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

if start_analysis:
    # Create a status container for live logging
    status_container = st.status("Analyzing video...", expanded=True)   
    progress_bar = st.progress(0, text="Starting...")
    def log(msg):
        status_container.write(msg)
    log("🚀 Starting analysis")
    start_time = time.time()
    log(f"Mode: {mode} | GPU: {use_gpu}")
    try:
        progress_bar.progress(5, text="Creating configuration...")
        log("Creating configuration...")
        profile_map = {"podcast": ContentProfile.PODCAST, "gaming": ContentProfile.GAMING, "reaction": ContentProfile.REACTION}
        prof = profile_map.get(content_profile) if content_profile != "None" else None
        config = DetectionConfig(
            watch_folder=Path(watch_folder),
            output_folder=Path(output_folder),
            ffmpeg_path=Path(get_ffmpeg_path()),
            ffprobe_path=Path(get_ffprobe_path()),
            processing_mode=mode,
            use_gpu=use_gpu,
            content_profile=prof,
            add_captions=add_captions,
            caption_export_srt=add_captions,   # always export SRT if captions enabled
            vertical_export=vertical_export,
        )
        config.apply_mode()
        config.max_clip_duration = max_clip_duration
        config.validate()
        progress_bar.progress(15, text="Config ready")
        log(f"✅ Config ready. {len(config.enabled_methods)} detectors enabled")
        
        progress_bar.progress(25, text="Loading models...")
        log("Initializing MomentAnalyzer (loading models may take a while)...")
        analyzer = MomentAnalyzer(config)
        cutter = VideoCutter(config)
        log("✅ Analyzer initialized")
        
        progress_bar.progress(45, text="Running detectors...")
        log(f"🔍 Running detection on {selected_name}...")
        moments, transcript = analyzer.analyze_video(Path(selected_path))
        log(f"✅ Detection complete! Found {len(moments)} moments")
        
        progress_bar.progress(75, text="Cutting clips...")
        log("✂️ Cutting clips...")
        clips = cutter.cut_moments(Path(selected_path), moments, transcript=transcript, video_duration=None)
        log(f"✅ Created {len(clips)} clips in {output_folder}")
        
        # --- Processing summary ---
        progress_bar.progress(90, text="Creating summary...")
        valid_clips = 0
        failed_clips = 0
        for clip_path in clips:
            if cutter._validate_clip(Path(clip_path)):
                valid_clips += 1
            else:
                failed_clips += 1
        avg_score = np.mean([m.combined_score for m in moments]) if moments else 0
        summary = {
            "processed_video": selected_name,
            "clips_created": valid_clips,
            "failed_clips": failed_clips,
            "average_score": avg_score,
            "processing_time_sec": time.time() - start_time,
            "detector_errors": analyzer.detector_errors if hasattr(analyzer, 'detector_errors') else {}
        }
        st.session_state.last_summary = summary
        st.success(f"✅ Done! Found {len(moments)} moments. Clips saved to {output_folder}")

        # Store results in session state
        st.session_state.analysis_results = {
            "video_name": selected_name,
            "video_path": selected_path,
            "moments": moments,
            "clips": clips,
            "output_folder": output_folder,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.last_run_video = selected_name
        st.session_state.last_run_time = time.time()
        
        progress_bar.progress(100, text="All done!")
        # Update status to complete
        status_container.update(label="Analysis complete!", state="complete", expanded=False)

    except Exception as e:
        progress_bar.progress(100, text="Failed")
        status_container.update(label="Analysis failed", state="error")
        log(f"❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        st.error(f"Analysis failed: {e}")

# ==================== TAB 2: RESULTS ====================
with tab2:
    st.subheader("Analysis Results")
    if st.session_state.analysis_results is None:
        st.info("No analysis results yet. Select a video and click 'Start Analysis' in the Videos tab.")
    else:
        res = st.session_state.analysis_results
        st.success(f"Results from video: **{res['video_name']}** (analysed at {res['timestamp']})")
        st.caption(f"Output folder: {res['output_folder']}")
        
        moments = res['moments']
        clips = res['clips']
        
        if not moments:
            st.warning("No viral moments detected.")
        else:
            st.metric("Total moments found", len(moments))
            for i, m in enumerate(moments):
                with st.expander(f"Moment {i+1} – Tier {m.tier.value} • Score {m.combined_score:.1f}", expanded=i==0):
                    # Border container around each moment's content
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            # Thumbnail from the moment start time
                            cap = cv2.VideoCapture(res['video_path'])
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                frame_idx = int(m.start_time * fps) if fps > 0 else 0
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame = cap.read()
                                if ret:
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    st.image(frame_rgb, caption=f"{m.start_time:.1f}s", width=200, use_container_width=False)
                                cap.release()
                        with col2:
                            st.markdown(f"**Time:** {m.start_time:.1f}s – {m.end_time:.1f}s  (duration {m.duration:.1f}s)")
                            st.progress(m.combined_score/100, text=f"Score {m.combined_score:.1f}")
                            # Show explanations
                            if hasattr(m, 'explanations') and m.explanations: 
                                st.markdown("💡 **Why this moment is viral:**")
                                for reason in m.explanations:
                                    st.markdown(f"- {reason}")
                        # Video and captions section (outside the columns)
                        if i < len(clips) and clips[i] and Path(clips[i]).exists():
                            st.video(clips[i])
                            # Display SRT captions if available
                            srt_path = Path(clips[i]).with_suffix('.srt')
                            if srt_path.exists():
                                with st.expander("📝 Captions (click to expand)"):
                                    with open(srt_path, 'r', encoding='utf-8') as f:
                                        caption_lines = f.read().strip().split('\n')
                                    # Simple display: skip numeric indexes and timestamp lines
                                    for line in caption_lines:
                                        line = line.strip()
                                        if line and not line[0].isdigit() and '-->' not in line:
                                            st.markdown(f"• {line}")
                            # Download button to copy clip path
                            clip_path_str = str(Path(clips[i]))
                            st.download_button(
                                label="📋 Copy clip path",
                                data=clip_path_str,
                                file_name=f"clip_{i+1}_path.txt",
                                mime="text/plain",
                                key=f"copy_btn_{i}"
                            )
                        else:
                            st.caption("(Clip not generated – check output folder permissions)")
# ==================== TAB 3: STATUS & SUMMARY ====================
with tab3:
    st.subheader("Processing Summary")
    if st.session_state.last_summary is None:
        st.info("No analysis performed yet. Run analysis in the Videos tab.")
    else:
        summary = st.session_state.last_summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Clips Created", summary['clips_created'])
        col2.metric("Failed Clips", summary['failed_clips'])
        col3.metric("Average Score", f"{summary['average_score']:.2f}")
        st.metric("Processing Time", f"{summary['processing_time_sec']:.1f} seconds")
        st.metric("Video Processed", summary['processed_video'])
        
        if summary['detector_errors']:
            st.subheader("⚠️ Detector Errors")
            for det, err in summary['detector_errors'].items():
                st.error(f"**{det}**: {err}")
        else:
            st.success("All detectors ran successfully.")
        
        # Optional: download button for JSON
        if st.button("💾 Save Summary as JSON"):
            import json
            json_str = json.dumps(summary, indent=2)
            st.download_button("Download JSON", json_str, "processing_summary.json", "application/json")                    