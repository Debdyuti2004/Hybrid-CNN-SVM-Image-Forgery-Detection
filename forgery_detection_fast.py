import os
import logging
import warnings
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.stats import loguniform

from skimage.feature import local_binary_pattern, hog
from scipy.fftpack import dct

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────

DATASET_PATH = r"C:\Users\KIIT0001\Documents\DocumentForgeryDetection\Dataset"
IMG_SIZE = 96
LBP_POINTS = 24
LBP_RADIUS = 3
RANDOM_STATE = 42
N_CV_FOLDS = 3
N_ITER = 20

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ── LOAD CNN ─────────────────────────────────────────────

log.info("Loading MobileNetV2...")
cnn_model = MobileNetV2(weights="imagenet",
                        include_top=False,
                        pooling="avg",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))

# ── DATA LOADING ─────────────────────────────────────────

def load_images(dataset_path):
    images, labels = [], []

    for category in ["original", "forged"]:
        cat_path = os.path.join(dataset_path, category)

        for fname in os.listdir(cat_path):
            if os.path.splitext(fname)[1].lower() not in IMG_EXTS:
                continue

            img = cv2.imread(os.path.join(cat_path, fname))
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(category)

    log.info("Loaded %d images", len(images))
    return np.array(images), np.array(labels)

# ── CLASSICAL FEATURES ───────────────────────────────────

def classical_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0

    lbp = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=LBP_POINTS+2, range=(0, LBP_POINTS+2))
    hist = hist.astype("float32")
    hist /= hist.sum() + 1e-6

    hog_feat = hog(gray, orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   feature_vector=True).astype("float32")

    coeffs = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    lf = coeffs[:8, :8]
    dct_stats = np.array([lf.mean(), lf.std(), lf.min(), lf.max()], dtype="float32")

    edges = cv2.Canny((gray*255).astype("uint8"), 50, 150)
    edge_stats = np.array([edges.mean(), edges.std()], dtype="float32")

    blocks = gray.reshape(IMG_SIZE//8, 8, IMG_SIZE//8, 8)
    block_var = blocks.var(axis=(1,3))
    jpeg_stats = np.array([block_var.mean(), block_var.std()], dtype="float32")

    lap_var = np.array([cv2.Laplacian(gray, cv2.CV_32F).var()], dtype="float32")

    return np.hstack([hist, hog_feat, dct_stats, edge_stats, jpeg_stats, lap_var])

# ── FEATURE EXTRACTION ───────────────────────────────────

def extract_features(images):
    log.info("Extracting CNN + classical features...")
    all_features = []

    for img in images:
        # CNN
        x = preprocess_input(img.astype("float32"))
        x = np.expand_dims(x, axis=0)
        cnn_feat = cnn_model.predict(x, verbose=0)[0]

        # Classical
        classical_feat = classical_features(img)

        combined = np.hstack([cnn_feat, classical_feat])
        all_features.append(combined)

    return np.array(all_features, dtype=np.float32)

# ── PLOTTING ─────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.legend()
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("results/roc.png", dpi=150)
    plt.close()

    return auc

# ── MAIN ─────────────────────────────────────────────────

def main():

    X_raw, y_raw = load_images(DATASET_PATH)
    X = extract_features(X_raw)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=500, random_state=RANDOM_STATE)),
        ("clf", SVC(probability=True))
    ])

    param_dist = {
        "clf__C": loguniform(1e-2, 1e2),
        "clf__gamma": loguniform(1e-4, 1e-1)
    }

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=N_ITER,
        cv=StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    log.info("Training hybrid model...")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    y_prob = best_model.predict_proba(X_test)[:,1]

    # Automatic threshold tuning
    best_acc = 0
    best_t = 0.5
    for t in np.arange(0.35, 0.66, 0.01):
        preds = (y_prob > t).astype(int)
        acc = (preds == y_test).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t

    y_pred = (y_prob > best_t).astype(int)

    auc = plot_roc_curve(y_test, y_prob)

    log.info("Best Params: %s", search.best_params_)
    log.info("AUC: %.4f", auc)
    log.info("Optimal Threshold: %.2f", best_t)
    log.info("\n%s", classification_report(y_test, y_pred))

    joblib.dump({
        "model": best_model,
        "label_encoder": le,
        "threshold": best_t
    }, "models/best_model.joblib")

    log.info("Hybrid model saved.")

if __name__ == "__main__":
    main()