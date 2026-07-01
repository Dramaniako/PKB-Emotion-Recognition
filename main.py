import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
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

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "default-src 'self' https: data: blob: 'unsafe-inline' 'unsafe-eval'; connect-src 'self' https: wss:; worker-src 'self' blob:; frame-ancestors 'self';"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Permissions-Policy"] = "camera=*, microphone=*"
    return response

# Serve static files locally (MediaPipe, Chart.js, etc.)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inisialisasi SQLite Database
init_db()

# Load Model: Coba memuat model SOTA PyTorch jika ada, jika tidak fallback ke Keras
pytorch_model_path = 'samaya_rafdb_sota_pytorch_b2_adamw.pth'
is_pytorch = os.path.exists(pytorch_model_path)

if is_pytorch:
    import torch
    from train_sota_pytorch import build_model
    from torchvision import transforms
    
    print(f"Memuat model SOTA PyTorch (EfficientNet-B2): {pytorch_model_path}")
    model = build_model(pretrained_weights=pytorch_model_path)
    model.eval()
    
    py_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    import tensorflow as tf
    model_path = 'samaya_rafdb_sota.keras'
    if os.path.exists(model_path):
        print(f"Memuat model SOTA Keras: {model_path}")
    else:
        model_path = 'samaya_rafdb_advanced.keras'
        print(f"Model SOTA Keras belum ditemukan. Fallback menggunakan model advanced: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

EMOTIONS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

# Logika Engagement Scoring
def get_engagement_score(emotion):
    if emotion in ['happy', 'surprise']: 
        return 1.0  # High Engagement / Antusias
    elif emotion == 'neutral': 
        return 0.8  # Focused / Memperhatikan
    else: 
        return 0.2  # Cognitive Overload / Bingung / Frustrasi

@app.api_route("/", methods=["GET", "HEAD"])
async def read_index(request: Request):
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

    # Inferensi Model
    if is_pytorch:
        # Preprocessing untuk PyTorch
        roi_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = py_transform(roi_rgb).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.softmax(outputs, dim=-1).numpy()
            
        max_index = int(np.argmax(prediction[0]))
        emotion = EMOTIONS[max_index]
        confidence = float(prediction[0][max_index])
    else:
        # Preprocessing untuk Keras
        roi_resized = cv2.resize(frame, (224, 224))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        roi_expanded = np.expand_dims(roi_rgb, axis=0)
        
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