import os
import cv2
import numpy as np
import mediapipe as mp
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from scipy.signal import find_peaks
import tempfile
import asyncio
import logging
from flask import Flask
import threading

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== HEALTH CHECK SERVER UNTUK RAILWAY ==========
health_app = Flask(__name__)

@health_app.route('/')
def health():
    return "Bot is running", 200

def run_health_server():
    port = int(os.environ.get('PORT', 8080))
    health_app.run(host='0.0.0.0', port=port)

threading.Thread(target=run_health_server, daemon=True).start()

# ========== INISIALISASI MEDIAPIPE ==========
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=35,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indeks landmark mata MediaPipe 478 titik
LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]

# ========== FUNGSI ANALISIS ==========
def eye_aspect_ratio(landmarks, eye_indices, h, w):
    """Menghitung Eye Aspect Ratio (EAR)"""
    points = []
    for idx in eye_indices:
        x = landmarks[idx].x * w
        y = landmarks[idx].y * h
        points.append((x, y))
    
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    h1 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    
    ear = (v1 + v2) / (2.0 * h1) if h1 > 0 else 0
    return ear

def classify_blink_state(ear):
    """Klasifikasi berdasarkan nilai EAR"""
    if ear > 0.25:
        return "OPEN"
    elif ear > 0.20:
        return "PARTIAL"
    else:
        return "CLOSED"

def analyze_video(video_path):
    """Analisis video untuk menghitung kedipan"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 5
    
    logger.info(f"Video: {total_frames} frames, {fps:.2f} fps, durasi {duration:.2f} detik")
    
    eye_history = {}
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                if face_id >= 35:
                    continue
                
                ear_left = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_INDICES, h, w)
                ear_right = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_INDICES, h, w)
                
                eye_history[(face_id, "left")] = eye_history.get((face_id, "left"), []) + [ear_left]
                eye_history[(face_id, "right")] = eye_history.get((face_id, "right"), []) + [ear_right]
        
        if frame_count % 30 == 0:
            logger.info(f"Proses: {frame_count}/{total_frames} frame")
    
    cap.release()
    
    if not eye_history:
        return None, "Tidak ada wajah terdeteksi dalam video"
    
    perfect_blinks = 0
    partial_events = 0
    eyes_analyzed = 0
    results_per_eye = []
    
    for eye_key, ear_list in eye_history.items():
        if len(ear_list) < 10:
            continue
        
        eyes_analyzed += 1
        face_id, side = eye_key
        
        states = [classify_blink_state(ear) for ear in ear_list]
        
        perfect_count = 0
        partial_count = 0
        i = 0
        while i < len(states) - 2:
            if states[i] == "OPEN" and states[i+1] == "CLOSED":
                j = i + 2
                while j < len(states) and states[j] == "CLOSED":
                    j += 1
                if j < len(states) and states[j] == "OPEN":
                    perfect_count += 1
                    i = j
                    continue
            if states[i] == "OPEN" and states[i+1] == "PARTIAL":
                j = i + 2
                while j < len(states) and states[j] == "PARTIAL":
                    j += 1
                if j < len(states) and states[j] == "OPEN":
                    partial_count += 1
                    i = j
                    continue
            i += 1
        
        perfect_blinks += perfect_count
        partial_events += partial_count
        results_per_eye.append({
            "eye": f"Wajah {face_id+1} - Mata {'Kiri' if side == 'left' else 'Kanan'}",
            "perfect": perfect_count,
            "partial": partial_count
        })
    
    total_mata = eyes_analyzed
    if total_mata == 0:
        return None, "Tidak ada mata yang dapat dianalisis"
    
    return {
        "total_eyes": total_mata,
        "perfect_blinks_total": perfect_blinks,
        "partial_blinks_total": partial_events,
        "avg_perfect_per_eye": perfect_blinks / total_mata,
        "avg_partial_per_eye": partial_events / total_mata,
        "duration_sec": duration,
        "details": results_per_eye[:10]
    }, None

# ========== HANDLER TELEGRAM ==========
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk video yang dikirim ke bot"""
    await update.message.reply_text("📹 Menerima video... Sedang menganalisis, mohon tunggu 30-60 detik")
    
    try:
        video_file = await update.message.video.get_file()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp_path = tmp.name
        
        await video_file.download_to_drive(tmp_path)
        logger.info(f"Video downloaded: {tmp_path}")
        
        result, error = analyze_video(tmp_path)
        
        os.unlink(tmp_path)
        
        if error:
            await update.message.reply_text(f"❌ {error}")
            return
        
        message = (
            f"📊 **HASIL ANALISIS KEDIPAN MATA**\n\n"
            f"⏱️ Durasi video: {result['duration_sec']:.1f} detik\n"
            f"👁️ Total mata terdeteksi: {result['total_eyes']}\n"
            f"🎯 **Total kedip sempurna**: {result['perfect_blinks_total']}\n"
            f"⚠️ **Total kedip sebagian**: {result['partial_blinks_total']}\n\n"
            f"📈 **Rata-rata per mata**:\n"
            f"   • Kedip sempurna: {result['avg_perfect_per_eye']:.1f}\n"
            f"   • Kedip sebagian: {result['avg_partial_per_eye']:.1f}\n\n"
            f"💡 **Definisi**:\n"
            f"   • Kedip sempurna = mata tertutup penuh\n"
            f"   • Kedip sebagian = mata setengah tertutup\n"
        )
        
        if result['details']:
            message += "\n🔍 **Detail 5 mata pertama**:\n"
            for i, d in enumerate(result['details'][:5]):
                message += f"   {i+1}. {d['eye']}: sempurna={d['perfect']}, sebagian={d['partial']}\n"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"❌ Terjadi error: {str(e)[:200]}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👁️ **Bot Analisis Kedipan Mata**\n\n"
        "Kirimkan video (maks 50 MB) yang berisi orang-orang dengan mata terlihat jelas.\n"
        "Bot akan menganalisis berapa kali mata berkedip sempurna vs sebagian.\n\n"
        "Format yang didukung: MP4\n"
        "Durasi optimal: 5-10 detik"
    )

def main():
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.error("BOT_TOKEN environment variable not set")
        return
    
    app = Application.builder().token(token).build()
    
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, start))
    
    logger.info("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
