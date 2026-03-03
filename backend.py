from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import cv2
import numpy as np
import os

from skimage.feature import local_binary_pattern, hog
from scipy.fftpack import dct

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ── APP SETUP ───────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMG_SIZE = 96
LBP_POINTS = 24
LBP_RADIUS = 3

MODEL_PATH = "models/best_model.joblib"

# ── LOAD MODEL ─────────────────────────────────────────

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("No trained model found.")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]
threshold = bundle.get("threshold", 0.5)

# ── LOAD CNN ───────────────────────────────────────────

cnn_model = MobileNetV2(weights="imagenet",
                        include_top=False,
                        pooling="avg",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ── FEATURE FUNCTIONS ──────────────────────────────────

def classical_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0

    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_POINTS+2, range=(0, LBP_POINTS+2))
    hist = hist.astype("float32")
    hist /= hist.sum() + 1e-6

    hog_feat = hog(gray, orientations=9,
                   pixels_per_cell=(8,8),
                   cells_per_block=(2,2),
                   feature_vector=True).astype("float32")

    coeffs = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    lf = coeffs[:8,:8]
    dct_stats = np.array([lf.mean(), lf.std(), lf.min(), lf.max()], dtype="float32")

    edges = cv2.Canny((gray*255).astype("uint8"), 50,150)
    edge_stats = np.array([edges.mean(), edges.std()], dtype="float32")

    blocks = gray.reshape(IMG_SIZE//8,8,IMG_SIZE//8,8)
    block_var = blocks.var(axis=(1,3))
    jpeg_stats = np.array([block_var.mean(), block_var.std()], dtype="float32")

    lap_var = np.array([cv2.Laplacian(gray, cv2.CV_32F).var()], dtype="float32")

    return np.hstack([hist, hog_feat, dct_stats, edge_stats, jpeg_stats, lap_var])

def extract_features(img):

    # CNN features
    x = preprocess_input(img.astype("float32"))
    x = np.expand_dims(x, axis=0)
    cnn_feat = cnn_model.predict(x, verbose=0)[0]

    # Classical
    classical_feat = classical_features(img)

    return np.hstack([cnn_feat, classical_feat]).astype("float32")

# ── ENDPOINT ───────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    feat = extract_features(img).reshape(1, -1)

    prob = model.predict_proba(feat)[0]
    pred_class = (prob[1] > threshold).astype(int)
    pred = le.inverse_transform([pred_class])[0]

    return {
        "filename": file.filename,
        "prediction": pred,
        "confidence": float(max(prob)),
        "probabilities": dict(zip(le.classes_, prob.tolist()))
    }
# Run with: uvicorn backend:app --reload