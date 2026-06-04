import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from database import init_db, insert_log

app = FastAPI(title="SAMAYA Edge AI API")

# Konfigurasi CORS untuk Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi SQLite Database
init_db()

# Load Model MobileNetV2 FER-2013
model = tf.keras.models.load_model('samaya_rafdb_advanced.keras', compile=False)
EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Logika Engagement Scoring
def get_engagement_score(emotion):
    if emotion in ['happy', 'surprise']: 
        return 1.0  # High Engagement / Antusias
    elif emotion == 'neutral': 
        return 0.8  # Focused / Memperhatikan
    else: 
        return 0.2  # Cognitive Overload / Bingung / Frustrasi

@app.get("/")
async def read_index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict_emotion(
    file: UploadFile = File(...),
    session_id: str = Form("anonymous_session")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Gambar rusak atau tidak valid"}

    # Karena Edge/Browser sudah memotong wajah, kita langsung ke preprocessing model
    roi_resized = cv2.resize(frame, (224, 224))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    # Model memiliki layer rescaling (1./255) internal, jadi gunakan data RGB berskala [0, 255]
    roi_expanded = np.expand_dims(roi_rgb, axis=0)

    # Inferensi Model
    prediction = model.predict(roi_expanded, verbose=0)
    max_index = int(np.argmax(prediction[0]))
    emotion = EMOTIONS[max_index]
    confidence = float(prediction[0][max_index])
    
    # Hitung Skor & Simpan ke DB
    score = get_engagement_score(emotion)
    insert_log(session_id, emotion, score)

    return {
        "emotion": emotion,
        "confidence": confidence,
        "engagement_score": score
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)