import streamlit as st
import os
import subprocess
import tempfile
import json
import re
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
        background: #1e1e2e; border: 1px solid #333; border-radius: 10px;
        padding: 16px; margin-bottom: 12px;
    }
    .status-ok { color: #4CAF50; font-weight: 600; }
    .status-err { color: #f44336; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 RANTONB Clipper 1.0</div>', unsafe_allow_html=True)
st.caption("Auto-clip • Vertical 9:16 • Face Detection • AI Subtitle")

# ====================== SESSION STATE ======================
for key, default in {
    "video_path": None,
    "video_source": None,
    "analyzed_clips": [],
    "processed_clips": [],
    "processing_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ====================== SIDEBAR: API KEY ======================
ENV_FILE = ".env"

def load_api_key():
    load_dotenv(ENV_FILE)
    return os.getenv("GROQ_API_KEY")

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
    st.markdown("**ℹ️ Tentang App**")
    st.caption("Clipper Groq menggunakan Whisper & LLaMA dari Groq untuk analisis & transkripsi otomatis.")

api_key = load_api_key()
if not api_key:
    st.error("⚠️ Masukkan Groq API Key di sidebar terlebih dahulu.")
    st.stop()

client = Groq(api_key=api_key)

# ====================== HELPER FUNCTIONS ======================

def run_ffmpeg(cmd: list, label: str = "") -> subprocess.CompletedProcess:
    """Run ffmpeg command and return result."""
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0 and label:
        st.warning(f"⚠️ ffmpeg ({label}) warning:\n`{result.stderr[-600:]}`")
    return result


def download_youtube(url: str) -> str | None:
    """Download video from YouTube using yt-dlp, return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        out_path = tmp.name

    with st.spinner("⬇️ Mendownload video dari YouTube..."):
        result = subprocess.run(
            ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
             "--merge-output-format", "mp4", "-o", out_path, url],
            capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
    if result.returncode != 0 or not Path(out_path).exists():
        st.error(f"❌ Gagal download video:\n`{result.stderr[-600:]}`")
        return None
    return out_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
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
    """Extract audio as WAV for transcription."""
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


def transcribe_clip(video_path: str, start: float, end: float) -> tuple[str, str]:
    """
    Transcribe a clip using Whisper via Groq.
    Returns (srt_content, detected_language_code).
    """
    audio_path = extract_audio(video_path, start, end)
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
            seg_end = float(seg.get("end", 0))
            text = seg.get("text", "").strip()
            if not text:
                continue

            def sec_to_srt(s):
                h = int(s // 3600)
                m = int((s % 3600) // 60)
                sc = int(s % 60)
                ms = int((s - int(s)) * 1000)
                return f"{h:02}:{m:02}:{sc:02},{ms:03}"

            srt_lines.append(str(i))
            srt_lines.append(f"{sec_to_srt(seg_start)} --> {sec_to_srt(seg_end)}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines), detected_lang
    finally:
        Path(audio_path).unlink(missing_ok=True)


def analyze_clips_with_ai(video_path: str, num_clips: int, clip_duration: int) -> list[dict]:
    """
    Use Groq LLM + audio transcription to find the best viral clip segments.
    Returns list of dicts: [{start, end, title, reason}, ...]
    """
    total_duration = get_video_duration(video_path)
    if total_duration == 0:
        st.error("❌ Tidak bisa membaca durasi video.")
        return []

    # Transcribe full audio in chunks (Groq limit ~25MB / ~10 menit per request)
    CHUNK_SIZE = 600  # 10 menit per chunk
    transcript_text = ""

    num_chunks = max(1, int(total_duration // CHUNK_SIZE) + (1 if total_duration % CHUNK_SIZE > 0 else 0))
    status_box = st.empty()

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = min(chunk_start + CHUNK_SIZE, total_duration)
        status_box.info(f"🎙️ Mentranskrip audio... bagian {chunk_idx + 1}/{num_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)")

        audio_path = extract_audio(video_path, chunk_start, chunk_end)
        try:
            # Cek ukuran file, skip jika kosong
            if not Path(audio_path).exists() or Path(audio_path).stat().st_size < 1000:
                continue

            with open(audio_path, "rb") as f:
                chunk_result = client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=f,
                    response_format="verbose_json",
                    temperature=0.0
                )
            chunk_segments = chunk_result.segments if hasattr(chunk_result, "segments") and chunk_result.segments else []
            for seg in chunk_segments:
                # Offset timestamp sesuai posisi chunk dalam video
                s = float(seg.get("start", 0)) + chunk_start
                e = float(seg.get("end", 0)) + chunk_start
                t = seg.get("text", "").strip()
                if t:
                    transcript_text += f"[{s:.1f}s-{e:.1f}s] {t}\n"
        except Exception as ex:
            st.warning(f"⚠️ Chunk {chunk_idx + 1} gagal ditranskrip: {ex}")
        finally:
            Path(audio_path).unlink(missing_ok=True)

    status_box.success(f"✅ Transkripsi selesai ({num_chunks} bagian)")

    if not transcript_text.strip():
        transcript_text = "(Tidak ada transkripsi tersedia — analisis berdasarkan durasi saja)"

    # Buat contoh eksplisit agar LLM tidak salah format
    example_end = min(float(clip_duration), total_duration)
    example_start2 = min(float(clip_duration) + 10, total_duration - float(clip_duration))
    example_end2 = min(example_start2 + float(clip_duration), total_duration)

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

    # Tampilkan raw output AI untuk debugging
    with st.expander("🔍 Raw output AI (untuk debugging)", expanded=False):
        st.code(raw, language="json")

    # Bersihkan markdown fence jika ada
    raw_clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Extract JSON array
    json_match = re.search(r"\[.*\]", raw_clean, re.DOTALL)
    if not json_match:
        st.error("❌ AI tidak mengembalikan format JSON yang valid.")
        st.code(raw[:800])
        return []

    try:
        clips = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        st.error(f"❌ Gagal parse JSON dari AI: {e}")
        st.code(raw_clean[:800])
        return []

    if not isinstance(clips, list):
        st.error("❌ JSON dari AI bukan array/list.")
        return []

    # Validasi dan perbaiki setiap clip
    valid_clips = []
    for i, c in enumerate(clips):
        try:
            start = max(0.0, float(c.get("start", 0)))
            end = float(c.get("end", start + clip_duration))

            # Paksa durasi tepat jika AI tidak patuh
            actual_dur = end - start
            if actual_dur < 5 or actual_dur > clip_duration * 1.5:
                end = start + clip_duration

            # Clamp agar tidak melebihi durasi video
            if end > total_duration:
                end = total_duration
                start = max(0.0, end - clip_duration)

            if end - start < 5:
                st.warning(f"⚠️ Clip {i+1} dilewati: durasi terlalu pendek ({end-start:.1f}s)")
                continue

            valid_clips.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "title": str(c.get("title", f"Clip {i+1}")),
                "reason": str(c.get("reason", ""))
            })
        except (ValueError, TypeError) as e:
            st.warning(f"⚠️ Clip {i+1} dari AI tidak valid: {e}")
            continue

    # Jika AI kurang dari yang diminta, isi sisa dengan distribusi merata
    if len(valid_clips) < num_clips:
        st.warning(f"⚠️ AI hanya memberikan {len(valid_clips)} clip valid dari {num_clips} yang diminta. Mengisi sisa otomatis...")
        segment = total_duration / num_clips
        for i in range(num_clips):
            if len(valid_clips) >= num_clips:
                break
            auto_start = round(i * segment, 2)
            auto_end = round(min(auto_start + clip_duration, total_duration), 2)
            overlaps = any(
                not (auto_end <= c["start"] or auto_start >= c["end"])
                for c in valid_clips
            )
            if not overlaps and auto_end - auto_start >= 5:
                valid_clips.append({
                    "start": auto_start,
                    "end": auto_end,
                    "title": f"Clip {i+1}",
                    "reason": "Dipilih otomatis (AI tidak memberikan cukup clip)"
                })

    # Urutkan berdasarkan waktu mulai
    valid_clips.sort(key=lambda x: x["start"])
    st.info(f"✅ Total clip valid: {len(valid_clips)} dari {num_clips} yang diminta")
    return valid_clips[:num_clips]


# ====================== REFRAME WITH FACE DETECTION ======================
# OpenCV Haar Cascade — ringan dan kompatibel di semua platform
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def reframe_clip(input_path: str, output_path: str):
    """Reframe video ke 9:16 dengan smooth face-tracking crop."""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w = int(h * 9 / 16)
    if crop_w > w:
        crop_w = w

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1080, 1920))

    # Posisi awal di tengah
    smooth_x = float((w - crop_w) // 2)
    target_x = smooth_x

    # Smoothing: semakin kecil alpha, semakin lambat & mulus gerakannya
    # 0.02 = sangat halus (lambat), 0.1 = cukup responsif
    alpha = 0.04

    # Deteksi wajah setiap N frame untuk performa
    DETECT_EVERY = 8
    frame_count = 0
    last_faces = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah hanya setiap DETECT_EVERY frame
        if frame_count % DETECT_EVERY == 0:
            small = cv2.resize(frame, (w // 2, h // 2))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # normalisasi pencahayaan
            faces = _face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Skala balik ke resolusi asli
            if len(faces) > 0:
                last_faces = [(x*2, y*2, fw*2, fh*2) for (x, y, fw, fh) in faces]
            # Jika tidak ada wajah, pertahankan last_faces (jangan reset ke kosong)
            # sehingga kamera tidak loncat ke tengah tiba-tiba

        # Hitung target_x dari wajah terakhir yang diketahui
        if len(last_faces) > 0:
            # Ambil wajah terbesar
            biggest = max(last_faces, key=lambda f: f[2] * f[3])
            x_f, y_f, w_f, h_f = biggest
            face_center = x_f + w_f // 2
            # Offset sedikit ke atas agar wajah tidak terlalu di bawah frame
            new_target = face_center - crop_w // 2
            new_target = max(0, min(new_target, w - crop_w))
            # Update target hanya jika perubahan cukup signifikan (>5% lebar crop)
            # Ini mencegah micro-jitter dari deteksi yang tidak stabil
            if abs(new_target - target_x) > crop_w * 0.05:
                target_x = float(new_target)

        # Exponential moving average untuk gerakan kamera yang sangat mulus
        smooth_x = smooth_x * (1 - alpha) + target_x * alpha
        smooth_x = max(0, min(smooth_x, w - crop_w))

        cropped = frame[:, int(smooth_x):int(smooth_x) + crop_w]
        resized = cv2.resize(cropped, (1080, 1920))
        out.write(resized)
        frame_count += 1

    cap.release()
    out.release()

def burn_subtitle(input_video: str, srt_content: str, output_video: str) -> str | None:
    """Burn SRT subtitle into video."""
    srt_path = Path(output_video).parent / f"_sub_{Path(output_video).stem}.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    # Windows: ffmpeg subtitles filter butuh forward slash
    # dan drive letter colon di-escape: C:/ → C\:/
    srt_str = str(srt_path).replace("\\", "/")
    srt_escaped = re.sub(r'''(?<=[A-Za-z]):''', r'''\\:''', srt_str)

    subtitle_style = (
        "Fontsize=16,"
        "PrimaryColour=&H00FFFFFF&,"
        "OutlineColour=&H00000000&,"
        "BackColour=&H80000000&,"
        "BorderStyle=3,"
        "Outline=1,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=80"
    )

    cmd = [
        "ffmpeg", "-i", input_video,
        "-vf", f"subtitles='{srt_escaped}':force_style='{subtitle_style}'",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",
        output_video, "-y"
    ]
    result = run_ffmpeg(cmd, "burn_subtitle")
    srt_path.unlink(missing_ok=True)

    if result.returncode != 0:
        import streamlit as _st
        _st.error(f"❌ burn_subtitle error:\n```\n{result.stderr[-800:]}\n```")
        return None
    return output_video


def process_single_clip(video_path: str, clip: dict, idx: int, output_dir: Path) -> dict | None:
    """Full pipeline: cut → reframe → transcribe → burn subtitle."""
    start = clip["start"]
    end = clip["end"]
    duration = end - start

    # Step 1: Cut raw horizontal clip
    # Gunakan -ss SEBELUM -i agar seeking akurat (fast seek),
    # lalu -t untuk durasi eksak, dan re-encode agar tidak terpotong di keyframe
    horiz_path = str(output_dir / f"_horiz_{idx}.mp4")
    cut_result = run_ffmpeg([
        "ffmpeg",
        "-ss", str(start),       # fast seek sebelum input
        "-i", video_path,
        "-t", str(duration),     # durasi eksak
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-crf", "18",
        "-avoid_negative_ts", "make_zero",
        "-reset_timestamps", "1",
        horiz_path, "-y"
    ], "cut_clip")

    if not Path(horiz_path).exists() or Path(horiz_path).stat().st_size < 10000:
        st.error(f"❌ Clip {idx}: Gagal memotong video (start={start}s, end={end}s)")
        return None

    actual_duration = get_video_duration(horiz_path)
    st.caption(f"  ✂️ Clip {idx} terpotong: {actual_duration:.1f}s (target: {duration:.1f}s)")

    # Step 2: Reframe to 9:16 with face detection
    reframed_path = str(output_dir / f"_reframed_{idx}.mp4")
    reframe_clip(horiz_path, reframed_path)

    if not Path(reframed_path).exists() or Path(reframed_path).stat().st_size < 10000:
        st.error(f"❌ Clip {idx}: Gagal reframe video")
        Path(horiz_path).unlink(missing_ok=True)
        return None

    # Step 3: Re-mux — gabungkan video dari reframe + audio dari horiz
    reframed_audio_path = str(output_dir / f"_reframed_audio_{idx}.mp4")
    mux_result = run_ffmpeg([
        "ffmpeg",
        "-i", reframed_path,
        "-i", horiz_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        "-shortest",
        reframed_audio_path, "-y"
    ], "mux_audio")

    video_for_sub = reframed_audio_path if Path(reframed_audio_path).exists() else reframed_path

    # Step 4: Transcribe audio dari clip yang sudah dipotong (horiz)
    srt_content, lang = transcribe_clip(horiz_path, 0, actual_duration)

    # Step 5: Burn subtitle
    final_path = str(output_dir / f"clip_{idx:02d}.mp4")
    burned = burn_subtitle(video_for_sub, srt_content, final_path)

    # Cleanup temp files
    for p in [horiz_path, reframed_path, reframed_audio_path]:
        Path(p).unlink(missing_ok=True)

    if burned and Path(final_path).exists():
        return {
            **clip,
            "output_path": final_path,
            "language": lang,
            "index": idx
        }
    st.error(f"❌ Clip {idx}: Gagal burn subtitle")
    return None


# ====================== MAIN UI ======================

# ─── STEP 1: Source Video ───────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-badge">1</span>Sumber Video</div>', unsafe_allow_html=True)

source_tab1, source_tab2 = st.tabs(["📁 Upload dari Lokal", "▶️ Download dari YouTube"])

with source_tab1:
    uploaded = st.file_uploader("Pilih file video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if uploaded and st.button("✅ Gunakan video ini", key="use_upload"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            st.session_state.video_path = tmp.name
            st.session_state.video_source = uploaded.name
            st.session_state.analyzed_clips = []
            st.session_state.processed_clips = []
            st.session_state.processing_done = False
        st.success(f"✅ Video `{uploaded.name}` siap diproses!")
        st.rerun()

with source_tab2:
    yt_url = st.text_input("URL YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("⬇️ Download Video", key="yt_download"):
        if not yt_url.strip():
            st.warning("Masukkan URL YouTube terlebih dahulu.")
        else:
            path = download_youtube(yt_url.strip())
            if path:
                st.session_state.video_path = path
                st.session_state.video_source = yt_url
                st.session_state.analyzed_clips = []
                st.session_state.processed_clips = []
                st.session_state.processing_done = False
                st.success("✅ Video berhasil didownload!")
                st.rerun()

if st.session_state.video_path and Path(st.session_state.video_path).exists():
    duration = get_video_duration(st.session_state.video_path)
    st.info(f"📹 **Video aktif:** `{st.session_state.video_source}` — Durasi: **{duration:.1f} detik** ({duration/60:.1f} menit)")
    with st.expander("👁️ Preview video"):
        st.video(st.session_state.video_path)
else:
    st.warning("Belum ada video. Upload atau download video terlebih dahulu.")
    st.stop()

st.divider()

# ─── STEP 2: Pengaturan & Analisis AI ──────────────────────────
st.markdown('<div class="step-header"><span class="step-badge">2</span>Analisis Clip oleh AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    num_clips = st.slider("🎬 Jumlah clip yang diinginkan", 1, 10, 3)
with col2:
    clip_duration = st.selectbox("⏱️ Durasi maksimal per clip (detik)", [15, 30, 45, 60, 90], index=2)

if st.button("🤖 Analisis & Tentukan Clip Terbaik", type="primary"):
    clips = analyze_clips_with_ai(st.session_state.video_path, num_clips, clip_duration)
    if clips:
        st.session_state.analyzed_clips = clips
        st.session_state.processed_clips = []
        st.session_state.processing_done = False
        st.success(f"✅ AI menemukan {len(clips)} clip terbaik!")
    else:
        st.error("❌ Analisis AI gagal. Coba lagi.")

if st.session_state.analyzed_clips:
    st.markdown("**📋 Hasil Analisis AI:**")
    for i, clip in enumerate(st.session_state.analyzed_clips, 1):
        dur = clip["end"] - clip["start"]
        st.markdown(
            f'<div class="clip-card">'
            f'<b>Clip {i}: {clip["title"]}</b><br>'
            f'⏰ {clip["start"]:.1f}s → {clip["end"]:.1f}s &nbsp;|&nbsp; '
            f'⏱ {dur:.1f} detik<br>'
            f'<small>💡 {clip["reason"]}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ─── STEP 3: Proses Semua Clip ──────────────────────────────
    st.markdown('<div class="step-header"><span class="step-badge">3</span>Proses: Reframe + Subtitle</div>', unsafe_allow_html=True)
    st.caption("Setiap clip akan dipotong, direframe ke 9:16 dengan face detection, lalu ditambahkan subtitle otomatis sesuai bahasa video.")

    if not st.session_state.processing_done:
        if st.button("🚀 Proses Semua Clip", type="primary"):
            temp_output_dir = Path(tempfile.mkdtemp(prefix="clippergroq_"))
            processed = []
            progress = st.progress(0, text="Memulai proses...")
            total = len(st.session_state.analyzed_clips)

            for i, clip in enumerate(st.session_state.analyzed_clips):
                progress.progress((i) / total, text=f"⚙️ Memproses Clip {i+1}/{total}: {clip['title']}...")
                with st.spinner(f"Clip {i+1}: Cut → Reframe → Transkripsi → Subtitle..."):
                    result = process_single_clip(
                        st.session_state.video_path,
                        clip, i + 1, temp_output_dir
                    )
                    if result:
                        processed.append(result)
                        st.success(f"✅ Clip {i+1} selesai — Bahasa terdeteksi: `{result['language']}`")
                    else:
                        st.error(f"❌ Clip {i+1} gagal diproses.")

            progress.progress(1.0, text="✅ Semua clip selesai!")
            st.session_state.processed_clips = processed
            st.session_state.processing_done = True
            st.rerun()

if st.session_state.processing_done and st.session_state.processed_clips:
    st.divider()

    # ─── STEP 4: Simpan Hasil ───────────────────────────────────
    st.markdown('<div class="step-header"><span class="step-badge">4</span>Simpan Hasil</div>', unsafe_allow_html=True)

    col_dir, col_prefix = st.columns(2)
    with col_dir:
        save_folder = st.text_input(
            "📁 Folder penyimpanan",
            value=str(Path.home() / "clippergroq_output"),
            help="Masukkan path folder. Folder akan dibuat otomatis jika belum ada."
        )
    with col_prefix:
        filename_prefix = st.text_input(
            "✏️ Prefix nama file",
            value="clip",
            help="Contoh: 'viral' → file menjadi viral_01.mp4, viral_02.mp4, dst."
        )

    st.markdown("**📋 Preview penamaan file:**")
    for i, clip in enumerate(st.session_state.processed_clips, 1):
        safe_title = re.sub(r'[^\w\s-]', '', clip['title']).strip().replace(' ', '_')[:30]
        fname = f"{filename_prefix}_{i:02d}_{safe_title}.mp4"
        dur = clip['end'] - clip['start']
        st.caption(f"`{fname}` — {dur:.1f}s — bahasa: {clip.get('language', '?')}")

    col_save, col_dl = st.columns(2)

    with col_save:
        if st.button("💾 Simpan ke Folder", type="primary"):
            out_dir = Path(save_folder)
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.error(f"❌ Gagal membuat folder: {e}")
                st.stop()

            saved_count = 0
            for i, clip in enumerate(st.session_state.processed_clips, 1):
                src = Path(clip["output_path"])
                if not src.exists():
                    st.warning(f"⚠️ File Clip {i} tidak ditemukan, skip.")
                    continue
                safe_title = re.sub(r'[^\w\s-]', '', clip['title']).strip().replace(' ', '_')[:30]
                dst = out_dir / f"{filename_prefix}_{i:02d}_{safe_title}.mp4"
                import shutil
                shutil.copy2(str(src), str(dst))
                saved_count += 1
                st.success(f"✅ Disimpan: `{dst}`")

            if saved_count > 0:
                st.balloons()
                st.success(f"🎉 {saved_count} clip berhasil disimpan di `{save_folder}`")

    with col_dl:
        st.markdown("**⬇️ Download Satu per Satu:**")
        for i, clip in enumerate(st.session_state.processed_clips, 1):
            src = Path(clip["output_path"])
            if src.exists():
                safe_title = re.sub(r'[^\w\s-]', '', clip['title']).strip().replace(' ', '_')[:30]
                fname = f"{filename_prefix}_{i:02d}_{safe_title}.mp4"
                with open(src, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download Clip {i}: {clip['title'][:30]}",
                        data=f,
                        file_name=fname,
                        mime="video/mp4",
                        key=f"dl_{i}"
                    )
            else:
                st.caption(f"Clip {i} tidak tersedia.")

st.divider()
st.caption("🎬 Clipper Groq • Whisper Large v3 Turbo + LLaMA 3.3 70B • Face Detection by MediaPipe")
