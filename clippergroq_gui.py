import streamlit as st
import os
import subprocess
import tempfile
import json
import re
import shutil
from pathlib import Path

import cv2

from groq import Groq
from dotenv import load_dotenv, set_key

# ====================== CONFIG ======================
st.set_page_config(page_title="RANTONB Clipper 1.0", layout="wide", page_icon="🎬")

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #FF6B35; }
    .step-badge {
        display: inline-block; background: #FF6B35; color: white;
        border-radius: 50%; width: 28px; height: 28px;
        text-align: center; line-height: 28px; font-weight: bold;
        margin-right: 8px; font-size: 14px;
    }
    .step-header { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
    .clip-card {
        background: #1e1e2e; border: 1px solid #444; border-radius: 10px;
        padding: 14px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 RANTONB Clipper 1.0</div>', unsafe_allow_html=True)
st.caption("Auto-clip • Vertical 9:16 • Face Detection • AI Subtitle")

# ====================== SESSION STATE ======================
SESSION_DEFAULTS = {
    "video_path": None,
    "video_source": None,
    "analyzed_clips": [],
    "processed_clips": [],
    "processing_done": False,
    "deleted_clips": [],
}
for key, default in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Preset disimpan terpisah agar tidak ikut reset
if "subtitle_presets" not in st.session_state:
    st.session_state.subtitle_presets = {}

# ====================== SIDEBAR ======================
ENV_FILE = ".env"

def load_api_key():
    load_dotenv(ENV_FILE)
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    return key

def save_api_key(api_key: str):
    set_key(ENV_FILE, "GROQ_API_KEY", api_key)
    os.environ["GROQ_API_KEY"] = api_key

with st.sidebar:
    st.header("🔑 Groq API Key")
    current_key = load_api_key()
    if current_key:
        st.success("✅ API Key aktif")
        masked = current_key[:8] + "..." + current_key[-4:]
        st.caption(f"Key: `{masked}`")
        if st.button("🗑️ Hapus API Key"):
            if Path(ENV_FILE).exists():
                Path(ENV_FILE).unlink()
            os.environ.pop("GROQ_API_KEY", None)
            st.rerun()
    else:
        with st.form("api_form"):
            new_key = st.text_input("Masukkan Groq API Key:", type="password")
            if st.form_submit_button("💾 Simpan"):
                if new_key.strip():
                    save_api_key(new_key.strip())
                    st.success("✅ API Key disimpan!")
                    st.rerun()

    st.divider()
    st.markdown("**🔄 Reset Project**")
    st.caption("Reset video & clip. Preset subtitle tidak ikut direset.")
    if st.button("🔄 Reset Project", type="secondary", use_container_width=True):
        for key, default in SESSION_DEFAULTS.items():
            st.session_state[key] = default if not isinstance(default, list) else []
        st.success("✅ Project direset!")
        st.rerun()

    st.divider()
    st.markdown("**ℹ️ Tentang App**")
    st.caption("Clipper Groq menggunakan Whisper & LLaMA dari Groq untuk analisis & transkripsi otomatis.")

api_key = load_api_key()
if not api_key:
    st.error("⚠️ Masukkan Groq API Key di sidebar terlebih dahulu.")
    st.stop()

client = Groq(api_key=api_key)

# ====================== HELPER FUNCTIONS ======================

def run_ffmpeg(cmd: list, label: str = "") -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0 and label:
        st.warning(f"⚠️ ffmpeg ({label}):\n`{result.stderr[-500:]}`")
    return result


def download_youtube(url: str) -> str | None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name
    with st.spinner("⬇️ Mendownload video dari YouTube..."):
        result = subprocess.run(
            ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
             "--merge-output-format", "mp4", "-o", out_path, url],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
    if result.returncode != 0 or not Path(out_path).exists():
        st.error(f"❌ Gagal download:\n`{result.stderr[-400:]}`")
        return None
    return out_path


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def extract_audio(video_path: str, start: float = None, end: float = None) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [audio_path, "-y"]
    run_ffmpeg(cmd, "extract_audio")
    return audio_path


def sec_to_srt(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02}:{m:02}:{sc:02},{ms:03}"


def transcribe_audio(audio_path: str) -> tuple[str, str]:
    """Transcribe audio file. Returns (srt_content, detected_lang)."""
    try:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="verbose_json",
                temperature=0.0
            )
        detected_lang = getattr(result, "language", "id") or "id"
        segments = result.segments if hasattr(result, "segments") and result.segments else []
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            seg_start = float(seg.get("start", 0))
            seg_end   = float(seg.get("end", 0))
            text      = seg.get("text", "").strip()
            if not text:
                continue
            srt_lines.append(str(i))
            srt_lines.append(f"{sec_to_srt(seg_start)} --> {sec_to_srt(seg_end)}")
            srt_lines.append(text)
            srt_lines.append("")
        return "\n".join(srt_lines), detected_lang
    except Exception as e:
        st.warning(f"⚠️ Transkripsi gagal: {e}")
        return "", "id"


def analyze_clips_with_ai(video_path: str, num_clips: int, clip_duration: int) -> list[dict]:
    total_duration = get_video_duration(video_path)
    if total_duration == 0:
        st.error("❌ Tidak bisa membaca durasi video.")
        return []

    CHUNK_SIZE = 600
    transcript_text = ""
    num_chunks = max(1, int(total_duration // CHUNK_SIZE) + (1 if total_duration % CHUNK_SIZE > 0 else 0))
    status_box = st.empty()

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end   = min(chunk_start + CHUNK_SIZE, total_duration)
        status_box.info(f"🎙️ Mentranskrip audio... bagian {chunk_idx+1}/{num_chunks}")
        audio_path = extract_audio(video_path, chunk_start, chunk_end)
        try:
            if not Path(audio_path).exists() or Path(audio_path).stat().st_size < 1000:
                continue
            srt, _ = transcribe_audio(audio_path)
            # Ambil teks saja dari SRT untuk prompt
            for line in srt.split("\n"):
                if line and not line.strip().isdigit() and "-->" not in line:
                    transcript_text += line + " "
        except Exception as ex:
            st.warning(f"⚠️ Chunk {chunk_idx+1} gagal: {ex}")
        finally:
            Path(audio_path).unlink(missing_ok=True)

    status_box.success(f"✅ Transkripsi selesai ({num_chunks} bagian)")

    if not transcript_text.strip():
        transcript_text = "(Tidak ada transkripsi tersedia)"

    example_end    = min(float(clip_duration), total_duration)
    example_start2 = min(float(clip_duration) + 10, total_duration - float(clip_duration))
    example_end2   = min(example_start2 + float(clip_duration), total_duration)

    prompt = (
        f"Kamu adalah editor video viral profesional.\n\n"
        f"TUGAS: Pilih TEPAT {num_clips} segmen terbaik dari video ini untuk dijadikan short clip viral.\n\n"
        f"DATA VIDEO:\n"
        f"- Durasi total: {total_duration:.1f} detik\n"
        f"- Jumlah clip yang HARUS dikembalikan: {num_clips} (WAJIB)\n"
        f"- Durasi setiap clip: TEPAT {clip_duration} detik (end - start = {clip_duration})\n\n"
        f"TRANSKRIPSI VIDEO:\n{transcript_text[:6000]}\n\n"
        f"ATURAN WAJIB:\n"
        f"1. Kembalikan TEPAT {num_clips} objek dalam array JSON\n"
        f"2. Setiap clip: end - start = TEPAT {clip_duration} detik\n"
        f"3. Tidak boleh ada clip yang overlap\n"
        f"4. start >= 0 dan end <= {total_duration:.1f}\n"
        f"5. Pilih momen paling menarik: hook, insight, emosi, tips, punchline\n\n"
        f"FORMAT RESPONS (HANYA JSON mentah, tanpa teks lain, tanpa backtick):\n"
        f"[\n"
        f'  {{"start": 0.0, "end": {example_end:.1f}, "title": "Judul clip 1", "reason": "Alasan"}},\n'
        f'  {{"start": {example_start2:.1f}, "end": {example_end2:.1f}, "title": "Judul clip 2", "reason": "Alasan"}}\n'
        f"]\n\n"
        f"Sekarang kembalikan TEPAT {num_clips} clip:"
    )

    with st.spinner("🤖 AI menganalisis momen-momen terbaik dari video..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000
        )
    raw = response.choices[0].message.content.strip()

    with st.expander("🔍 Raw output AI (debugging)", expanded=False):
        st.code(raw, language="json")

    raw_clean  = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    json_match = re.search(r"\[.*\]", raw_clean, re.DOTALL)
    if not json_match:
        st.error("❌ AI tidak mengembalikan format JSON yang valid.")
        return []

    try:
        clips = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Gagal parse JSON: {e}")
        return []

    if not isinstance(clips, list):
        return []

    valid_clips = []
    for i, c in enumerate(clips):
        try:
            start = max(0.0, float(c.get("start", 0)))
            end   = float(c.get("end", start + clip_duration))
            if end - start < 5 or end - start > clip_duration * 1.5:
                end = start + clip_duration
            if end > total_duration:
                end   = total_duration
                start = max(0.0, end - clip_duration)
            if end - start < 5:
                continue
            valid_clips.append({
                "start":  round(start, 2),
                "end":    round(end, 2),
                "title":  str(c.get("title", f"Clip {i+1}")),
                "reason": str(c.get("reason", ""))
            })
        except (ValueError, TypeError):
            continue

    if len(valid_clips) < num_clips:
        st.warning(f"⚠️ AI memberikan {len(valid_clips)} clip, mengisi sisa otomatis...")
        segment = total_duration / num_clips
        for i in range(num_clips):
            if len(valid_clips) >= num_clips:
                break
            auto_start = round(i * segment, 2)
            auto_end   = round(min(auto_start + clip_duration, total_duration), 2)
            overlaps   = any(not (auto_end <= c["start"] or auto_start >= c["end"]) for c in valid_clips)
            if not overlaps and auto_end - auto_start >= 5:
                valid_clips.append({
                    "start": auto_start, "end": auto_end,
                    "title": f"Clip {i+1}", "reason": "Dipilih otomatis"
                })

    valid_clips.sort(key=lambda x: x["start"])
    return valid_clips[:num_clips]


# ====================== REFRAME ======================
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def reframe_clip(input_path: str, output_path: str):
    cap    = cv2.VideoCapture(input_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(h * 9 / 16)
    if crop_w > w:
        crop_w = w

    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1920))
    smooth_x = float((w - crop_w) // 2)
    target_x = smooth_x
    alpha    = 0.04
    DETECT_EVERY = 8
    frame_count  = 0
    last_faces   = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % DETECT_EVERY == 0:
            small = cv2.resize(frame, (w // 2, h // 2))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = _face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=6, minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                last_faces = [(x*2, y*2, fw*2, fh*2) for (x, y, fw, fh) in faces]

        if len(last_faces) > 0:
            biggest    = max(last_faces, key=lambda f: f[2] * f[3])
            x_f, _, w_f, _ = biggest
            face_center = x_f + w_f // 2
            new_target  = float(max(0, min(face_center - crop_w // 2, w - crop_w)))
            if abs(new_target - target_x) > crop_w * 0.05:
                target_x = new_target

        smooth_x = smooth_x * (1 - alpha) + target_x * alpha
        smooth_x = max(0, min(smooth_x, w - crop_w))
        cropped  = frame[:, int(smooth_x):int(smooth_x) + crop_w]
        resized  = cv2.resize(cropped, (1080, 1920))
        out.write(resized)
        frame_count += 1

    cap.release()
    out.release()


# ====================== SUBTITLE ======================

def hex_to_ass(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&"

def build_subtitle_style(cfg: dict) -> str:
    return (
        f"Fontname={cfg.get('font','Arial')},"
        f"Fontsize={cfg.get('size', 16)},"
        f"PrimaryColour={hex_to_ass(cfg.get('color','#FFFFFF'))},"
        f"BackColour=&H80000000&,"
        f"OutlineColour=&H00000000&,"
        f"BorderStyle=3,"
        f"Outline={cfg.get('outline', 1)},"
        f"Shadow=0,"
        f"Bold={1 if cfg.get('bold') else 0},"
        f"Italic={1 if cfg.get('italic') else 0},"
        f"Alignment={cfg.get('alignment', 2)},"
        f"MarginV={cfg.get('margin_v', 80)}"
    )

def burn_subtitle(input_video: str, srt_content: str, output_video: str, subtitle_cfg: dict) -> str | None:
    srt_path = Path(output_video).parent / f"_sub_{Path(output_video).stem}.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    srt_str     = str(srt_path).replace("\\", "/")
    srt_escaped = re.sub(r"(?<=[A-Za-z]):", r"\\:", srt_str)
    style       = build_subtitle_style(subtitle_cfg)

    cmd = [
        "ffmpeg", "-i", input_video,
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-crf", "18",
        output_video, "-y"
    ]
    result = run_ffmpeg(cmd, "burn_subtitle")
    srt_path.unlink(missing_ok=True)
    if result.returncode != 0:
        st.error(f"❌ Gagal burn subtitle:\n```\n{result.stderr[-600:]}\n```")
        return None
    return output_video


# ====================== PROCESS CLIP ======================

def process_single_clip(video_path: str, clip: dict, idx: int,
                        output_dir: Path, subtitle_cfg: dict) -> dict | None:
    start    = clip["start"]
    end      = clip["end"]
    duration = end - start

    # Step 1: Cut
    horiz_path = str(output_dir / f"_horiz_{idx}.mp4")
    run_ffmpeg([
        "ffmpeg", "-ss", str(start), "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-crf", "18",
        "-avoid_negative_ts", "make_zero", "-reset_timestamps", "1",
        horiz_path, "-y"
    ], "cut_clip")

    if not Path(horiz_path).exists() or Path(horiz_path).stat().st_size < 10000:
        st.error(f"❌ Clip {idx}: Gagal memotong video")
        return None

    actual_duration = get_video_duration(horiz_path)

    # Step 2: Reframe
    reframed_path = str(output_dir / f"_reframed_{idx}.mp4")
    reframe_clip(horiz_path, reframed_path)

    if not Path(reframed_path).exists() or Path(reframed_path).stat().st_size < 10000:
        st.error(f"❌ Clip {idx}: Gagal reframe")
        Path(horiz_path).unlink(missing_ok=True)
        return None

    # Step 3: Mux audio
    reframed_audio = str(output_dir / f"_ra_{idx}.mp4")
    run_ffmpeg([
        "ffmpeg", "-i", reframed_path, "-i", horiz_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-c:a", "aac",
        "-preset", "fast", "-shortest",
        reframed_audio, "-y"
    ], "mux_audio")

    video_for_sub = reframed_audio if Path(reframed_audio).exists() else reframed_path

    # Step 4: Transcribe
    audio_path   = extract_audio(horiz_path, 0, actual_duration)
    srt_content, lang = "", "id"
    try:
        srt_content, lang = transcribe_audio(audio_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    # Step 5: Burn subtitle
    final_path = str(output_dir / f"clip_{idx:02d}.mp4")
    burned     = burn_subtitle(video_for_sub, srt_content, final_path, subtitle_cfg)

    for p in [horiz_path, reframed_path, reframed_audio]:
        Path(p).unlink(missing_ok=True)

    if burned and Path(final_path).exists():
        return {**clip, "output_path": final_path, "language": lang, "index": idx}
    st.error(f"❌ Clip {idx}: Gagal burn subtitle")
    return None


# ============================================================
# MAIN UI
# ============================================================

# ─── STEP 1: SUMBER VIDEO ────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-badge">1</span>Sumber Video</div>', unsafe_allow_html=True)

tab_local, tab_yt = st.tabs(["📁 Upload dari Lokal", "▶️ Download dari YouTube"])

with tab_local:
    uploaded = st.file_uploader("Pilih file video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded and st.button("✅ Gunakan video ini", key="use_upload"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            st.session_state.video_path      = tmp.name
            st.session_state.video_source    = uploaded.name
            st.session_state.analyzed_clips  = []
            st.session_state.processed_clips = []
            st.session_state.processing_done = False
            st.session_state.deleted_clips   = []
        st.rerun()

with tab_yt:
    yt_url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("⬇️ Download Video", key="yt_dl"):
        if not yt_url.strip():
            st.warning("Masukkan URL YouTube.")
        else:
            path = download_youtube(yt_url.strip())
            if path:
                st.session_state.video_path      = path
                st.session_state.video_source    = yt_url
                st.session_state.analyzed_clips  = []
                st.session_state.processed_clips = []
                st.session_state.processing_done = False
                st.session_state.deleted_clips   = []
                st.rerun()

if st.session_state.video_path and Path(st.session_state.video_path).exists():
    dur = get_video_duration(st.session_state.video_path)
    st.info(f"📹 **Video aktif:** `{st.session_state.video_source}` — Durasi: **{dur:.1f}s** ({dur/60:.1f} menit)")
    with st.expander("👁️ Preview"):
        st.video(st.session_state.video_path)
else:
    st.warning("Belum ada video. Upload atau download video terlebih dahulu.")
    st.stop()

st.divider()

# ─── STEP 2: ANALISIS AI ─────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-badge">2</span>Analisis Clip oleh AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    num_clips = st.slider("🎬 Jumlah clip", 1, 10, 3)
with col2:
    clip_duration = st.selectbox("⏱️ Durasi per clip (detik)", [15, 30, 45, 60, 90], index=2)

if st.button("🤖 Analisis & Tentukan Clip Terbaik", type="primary"):
    clips = analyze_clips_with_ai(st.session_state.video_path, num_clips, clip_duration)
    if clips:
        st.session_state.analyzed_clips  = clips
        st.session_state.processed_clips = []
        st.session_state.processing_done = False
        st.session_state.deleted_clips   = []
        st.success(f"✅ AI menemukan {len(clips)} clip terbaik!")

st.divider()

# ─── STEP 3: KELOLA HASIL CLIP ───────────────────────────────
if st.session_state.analyzed_clips:
    st.markdown('<div class="step-header"><span class="step-badge">3</span>Kelola Hasil Clip</div>', unsafe_allow_html=True)
    st.caption("Klik ✕ untuk menghapus clip yang tidak diinginkan sebelum diproses.")

    active_clips = [
        (i, clip) for i, clip in enumerate(st.session_state.analyzed_clips)
        if i not in st.session_state.deleted_clips
    ]

    if not active_clips:
        st.warning("⚠️ Semua clip dihapus. Klik Analisis ulang untuk memulai lagi.")
    else:
        for i, clip in active_clips:
            dur = clip["end"] - clip["start"]
            col_info, col_btn = st.columns([11, 1])
            with col_info:
                st.markdown(
                    f'<div class="clip-card">'
                    f'<b>Clip {i+1}: {clip["title"]}</b><br>'
                    f'⏰ {clip["start"]:.1f}s → {clip["end"]:.1f}s &nbsp;|&nbsp; ⏱ {dur:.1f}s<br>'
                    f'<small>💡 {clip["reason"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_{i}", help="Hapus clip ini"):
                    st.session_state.deleted_clips.append(i)
                    st.rerun()

        st.info(f"📋 **{len(active_clips)} clip** akan diproses.")

    st.divider()

    # ─── STEP 4: KUSTOMISASI SUBTITLE ────────────────────────
    st.markdown('<div class="step-header"><span class="step-badge">4</span>Kustomisasi Subtitle</div>', unsafe_allow_html=True)

    FONT_OPTIONS = [
        "Arial", "Arial Black", "Verdana", "Tahoma", "Trebuchet MS",
        "Times New Roman", "Georgia", "Courier New",
        "Impact", "Comic Sans MS", "Calibri", "Segoe UI"
    ]
    POSITION_OPTIONS = {
        "Bawah Tengah": 2,
        "Bawah Kiri":   1,
        "Bawah Kanan":  3,
        "Tengah":        5,
        "Atas Tengah":   8,
    }

    # ── Load Preset ──
    preset_names = list(st.session_state.subtitle_presets.keys())
    selected_preset_cfg = {}

    col_pload, col_pname, col_psave = st.columns([3, 3, 2])
    with col_pload:
        if preset_names:
            chosen = st.selectbox("📂 Load Preset", ["— Pilih Preset —"] + preset_names)
            if chosen != "— Pilih Preset —":
                selected_preset_cfg = st.session_state.subtitle_presets[chosen]
        else:
            st.caption("Belum ada preset tersimpan.")

    def pv(key, default):
        return selected_preset_cfg.get(key, default)

    # ── Pengaturan ──
    st.markdown("**⚙️ Pengaturan Subtitle:**")
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        font_idx = FONT_OPTIONS.index(pv("font", "Arial")) if pv("font", "Arial") in FONT_OPTIONS else 0
        sub_font    = st.selectbox("🔤 Font", FONT_OPTIONS, index=font_idx)
        sub_size    = st.slider("📏 Ukuran Font", 10, 48, int(pv("size", 16)))
        sub_color   = st.color_picker("🎨 Warna Teks", pv("color", "#FFFFFF"))

    with sc2:
        sub_bold    = st.checkbox("**B** Bold",   value=bool(pv("bold", False)))
        sub_italic  = st.checkbox("*I* Italic",   value=bool(pv("italic", False)))
        sub_outline = st.slider("🔲 Tebal Outline", 0, 4, int(pv("outline", 1)))

    with sc3:
        pos_keys  = list(POSITION_OPTIONS.keys())
        pos_vals  = list(POSITION_OPTIONS.values())
        saved_align = pv("alignment", 2)
        pos_idx   = pos_vals.index(saved_align) if saved_align in pos_vals else 0
        pos_label   = st.selectbox("📍 Posisi", pos_keys, index=pos_idx)
        sub_margin_v = st.slider("↕️ Margin Vertikal", 0, 200, int(pv("margin_v", 80)))

    subtitle_cfg = {
        "font":      sub_font,
        "size":      sub_size,
        "color":     sub_color,
        "bold":      sub_bold,
        "italic":    sub_italic,
        "outline":   sub_outline,
        "alignment": POSITION_OPTIONS[pos_label],
        "margin_v":  sub_margin_v,
    }

    # ── Preview ──
    preview_style = (
        f"font-family:{sub_font}; font-size:{min(sub_size, 22)}px; color:{sub_color}; "
        f"{'font-weight:bold;' if sub_bold else ''}"
        f"{'font-style:italic;' if sub_italic else ''}"
        f"background:rgba(0,0,0,0.6); padding:4px 14px; border-radius:4px; display:inline-block;"
    )
    st.markdown(
        f'<div style="text-align:center;margin:10px 0;">'
        f'<span style="{preview_style}">✨ Preview Subtitle Kamu ✨</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Simpan Preset ──
    with col_pname:
        preset_name_input = st.text_input("💾 Nama Preset Baru", placeholder="contoh: Style Viral")
    with col_psave:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Simpan Preset", use_container_width=True):
            if preset_name_input.strip():
                st.session_state.subtitle_presets[preset_name_input.strip()] = subtitle_cfg.copy()
                st.success(f"✅ Preset '{preset_name_input.strip()}' disimpan!")
                st.rerun()
            else:
                st.warning("Isi nama preset terlebih dahulu.")

    # ── Kelola Preset ──
    if preset_names:
        with st.expander("🗂️ Kelola Preset Tersimpan"):
            for pname in list(preset_names):
                pcfg = st.session_state.subtitle_presets.get(pname, {})
                col_pi, col_pd = st.columns([9, 1])
                with col_pi:
                    st.caption(
                        f"**{pname}** — {pcfg.get('font','?')} {pcfg.get('size','?')}px "
                        f"| Bold:{pcfg.get('bold',False)} | Italic:{pcfg.get('italic',False)} "
                        f"| Warna:{pcfg.get('color','?')}"
                    )
                with col_pd:
                    if st.button("✕", key=f"del_preset_{pname}"):
                        del st.session_state.subtitle_presets[pname]
                        st.rerun()

    st.divider()

    # ─── STEP 5: PROSES ──────────────────────────────────────
    st.markdown('<div class="step-header"><span class="step-badge">5</span>Proses: Reframe + Subtitle</div>', unsafe_allow_html=True)

    active_for_process = [
        clip for i, clip in enumerate(st.session_state.analyzed_clips)
        if i not in st.session_state.deleted_clips
    ]

    if not active_for_process:
        st.warning("⚠️ Tidak ada clip untuk diproses.")
    elif not st.session_state.processing_done:
        if st.button("🚀 Proses Semua Clip", type="primary"):
            temp_output_dir = Path(tempfile.mkdtemp(prefix="clippergroq_"))
            processed = []
            total     = len(active_for_process)
            progress  = st.progress(0, text="Memulai proses...")

            for idx, clip in enumerate(active_for_process):
                progress.progress(idx / total, text=f"⚙️ Clip {idx+1}/{total}: {clip['title']}...")
                with st.spinner(f"Clip {idx+1}: Cut → Reframe → Transkripsi → Subtitle..."):
                    result = process_single_clip(
                        st.session_state.video_path,
                        clip, idx + 1, temp_output_dir, subtitle_cfg
                    )
                    if result:
                        processed.append(result)
                        st.success(f"✅ Clip {idx+1} selesai — Bahasa: `{result['language']}`")
                    else:
                        st.error(f"❌ Clip {idx+1} gagal.")

            progress.progress(1.0, text="✅ Semua clip selesai!")
            st.session_state.processed_clips = processed
            st.session_state.processing_done = True
            st.rerun()

# ─── STEP 6: SIMPAN HASIL ────────────────────────────────────
if st.session_state.processing_done and st.session_state.processed_clips:
    import zipfile, io

    st.divider()
    st.markdown('<div class="step-header"><span class="step-badge">6</span>Simpan Hasil</div>', unsafe_allow_html=True)

    # Prefix nama file
    filename_prefix = st.text_input("✏️ Prefix nama file", value="clip",
                                    help="Contoh: 'viral' → viral_01_Judul.mp4")

    def make_fname(i, clip):
        safe = re.sub(r'[^\w\s-]', '', clip['title']).strip().replace(' ', '_')[:30]
        return f"{filename_prefix}_{i:02d}_{safe}.mp4"

    st.markdown("**📋 Clip siap didownload:**")

    # ── Download per clip ──
    for i, clip in enumerate(st.session_state.processed_clips, 1):
        src  = Path(clip["output_path"])
        dur  = clip['end'] - clip['start']
        lang = clip.get('language', '?')
        fname = make_fname(i, clip)

        col_info, col_btn = st.columns([8, 2])
        with col_info:
            st.markdown(
                f'<div class="clip-card" style="padding:10px;">'
                f'<b>Clip {i}: {clip["title"]}</b><br>'
                f'<small>⏱ {dur:.1f}s &nbsp;|&nbsp; 🌐 {lang} &nbsp;|&nbsp; 📄 {fname}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if src.exists():
                with open(src, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download",
                        data=f,
                        file_name=fname,
                        mime="video/mp4",
                        key=f"dl_{i}",
                        use_container_width=True
                    )
            else:
                st.caption("❌ File tidak ada")

    st.divider()

    # ── Download semua sebagai ZIP ──
    st.markdown("**📦 Download Semua Sekaligus (ZIP):**")

    existing_clips = [
        (i, clip) for i, clip in enumerate(st.session_state.processed_clips, 1)
        if Path(clip["output_path"]).exists()
    ]

    if existing_clips:
        with st.spinner("📦 Menyiapkan file ZIP..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_STORED) as zf:
                for i, clip in existing_clips:
                    fname = make_fname(i, clip)
                    zf.write(clip["output_path"], arcname=fname)
            zip_buffer.seek(0)

        st.download_button(
            label=f"📦 Download Semua ({len(existing_clips)} clip) sebagai ZIP",
            data=zip_buffer,
            file_name=f"{filename_prefix}_semua_clip.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True
        )
        st.caption("💡 Setelah download ZIP, extract dan semua clip langsung tersedia.")
    else:
        st.warning("⚠️ Tidak ada file clip yang tersedia.")

st.divider()
st.caption("🎬 RANTONB Clipper 1.0 • Whisper Large v3 Turbo + LLaMA 3.3 70B • Face Detection by OpenCV")
