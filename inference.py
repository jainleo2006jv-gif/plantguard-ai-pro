"""
PlantGuard AI Pro - Inference Engine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy TF import so the rest of the app still loads if TF is absent
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    logger.warning("TensorFlow not available – running in demo mode.")


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    rank:         int
    class_index:  int
    class_id:     str
    display_name: str
    confidence:   float
    severity:     str = "Unknown"
    pathogen:     str = "Unknown"
    treatment:    str = ""
    prevention:   str = ""
    spread_risk:  str = ""
    recovery_time: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─── Model Registry (future multi-model support) ──────────────────────────────

_MODEL_REGISTRY: dict[str, object] = {}

def _get_model(model_path: str):
    """Load and cache a Keras model by path."""
    if not _TF_AVAILABLE:
        return None
    if model_path not in _MODEL_REGISTRY:
        try:
            model = tf.keras.models.load_model(model_path)
            _MODEL_REGISTRY[model_path] = model
            logger.info("Loaded model: %s", model_path)
        except Exception as exc:
            logger.error("Could not load model %s: %s", model_path, exc)
            _MODEL_REGISTRY[model_path] = None
    return _MODEL_REGISTRY[model_path]


# ─── Class Names ──────────────────────────────────────────────────────────────

PLANT_VILLAGE_CLASSES: List[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _class_id_to_display(class_id: str) -> str:
    parts = class_id.split("___")
    plant = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
    condition = parts[1].replace("_", " ").strip() if len(parts) > 1 else ""
    if "healthy" in condition.lower():
        return f"{plant} – Healthy"
    return f"{plant}: {condition}"


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    image_array: np.ndarray,   # shape (1, H, W, 3), normalised
    model_path: str,
    top_k: int,
    class_names: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> List[PredictionResult]:
    """
    Run model inference and return top-k PredictionResult objects.
    Falls back to demo predictions if model is unavailable.
    """
    names = class_names or PLANT_VILLAGE_CLASSES
    meta  = metadata or {}

    model = _get_model(model_path)

    if model is None:
        return _demo_predictions(top_k, names, meta)

    try:
        raw = model.predict(image_array, verbose=0)           # (1, num_classes)
        probs = raw[0]
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        return _demo_predictions(top_k, names, meta)

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        class_id = names[idx] if idx < len(names) else f"class_{idx}"
        conf     = float(probs[idx])
        m        = meta.get(class_id, {})
        results.append(PredictionResult(
            rank          = rank,
            class_index   = int(idx),
            class_id      = class_id,
            display_name  = m.get("display_name", _class_id_to_display(class_id)),
            confidence    = conf,
            severity      = m.get("severity", "Unknown"),
            pathogen      = m.get("pathogen", "Unknown"),
            treatment     = m.get("treatment", ""),
            prevention    = m.get("prevention", ""),
            spread_risk   = m.get("spread_risk", ""),
            recovery_time = m.get("recovery_time", ""),
        ))
    return results


def _demo_predictions(top_k: int, names: List[str], meta: dict) -> List[PredictionResult]:
    """Return deterministic demo predictions when no model is available."""
    sample_ids = [
        "Tomato___Early_blight",
        "Tomato___healthy",
        "Apple___Apple_scab",
        "Corn_(maize)___Common_rust_",
        "Potato___Late_blight",
    ]
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(len(sample_ids)) * 2)
    raw[::-1].sort()

    results = []
    for rank, (cid, conf) in enumerate(zip(sample_ids[:top_k], raw[:top_k]), start=1):
        m = meta.get(cid, {})
        results.append(PredictionResult(
            rank          = rank,
            class_index   = rank - 1,
            class_id      = cid,
            display_name  = m.get("display_name", _class_id_to_display(cid)),
            confidence    = float(conf),
            severity      = m.get("severity", "Unknown"),
            pathogen      = m.get("pathogen", "Unknown"),
            treatment     = m.get("treatment", "Consult a plant pathologist."),
            prevention    = m.get("prevention", "Maintain good cultural practices."),
            spread_risk   = m.get("spread_risk", "Unknown"),
            recovery_time = m.get("recovery_time", "Unknown"),
        ))
    return results


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def compute_gradcam(
    image_array: np.ndarray,
    model_path: str,
    class_index: int,
    layer_name: str,
    alpha: float = 0.4,
    original_size: tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Compute Grad-CAM heatmap overlay as a uint8 RGB numpy array.
    Returns None if computation fails.
    """
    if not _TF_AVAILABLE:
        return None

    model = _get_model(model_path)
    if model is None:
        return None

    try:
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            inputs        = tf.cast(image_array, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            loss          = predictions[:, class_index]

        grads   = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam     = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)
        cam     = tf.nn.relu(cam)

        # Normalise
        cam_np = cam.numpy()
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max - c_min < 1e-8:
            return None
        cam_np = (cam_np - c_min) / (c_max - c_min)

        # Resize to original image dimensions
        cam_resized = np.array(
            Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
                original_size, Image.LANCZOS
            ),
            dtype=np.float32,
        ) / 255.0

        # Colorise (jet-like: blue→cyan→green→yellow→red)
        heatmap = _apply_colormap(cam_resized)

        # Blend with original
        orig = (image_array[0] * 255).astype(np.uint8)
        orig_resized = np.array(
            Image.fromarray(orig).resize(original_size, Image.LANCZOS),
            dtype=np.float32,
        )
        blended = (1 - alpha) * orig_resized + alpha * heatmap * 255
        return np.clip(blended, 0, 255).astype(np.uint8)

    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)
        return None


def _apply_colormap(x: np.ndarray) -> np.ndarray:
    """Vectorised jet-like colourmap for a 2-D array in [0, 1]."""
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)
