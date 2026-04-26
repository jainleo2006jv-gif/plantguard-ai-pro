"""
PlantGuard AI Pro - Utilities
"""
from __future__ import annotations

import io
import json
import datetime
import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)


# ─── Image Validation & Quality ───────────────────────────────────────────────

def load_image_from_bytes(data: bytes) -> Optional[Image.Image]:
    """Safely load a PIL Image from raw bytes."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as exc:
        logger.warning("Failed to load image: %s", exc)
        return None


def validate_image(img: Image.Image, min_size: int = 32) -> Tuple[bool, str]:
    """
    Validate image meets minimum requirements.
    Returns (is_valid, message).
    """
    w, h = img.size
    if w < min_size or h < min_size:
        return False, f"Image too small ({w}×{h}). Minimum {min_size}×{min_size}."
    if img.mode not in ("RGB", "RGBA", "L"):
        return False, f"Unsupported colour mode: {img.mode}"
    return True, "OK"


def compute_image_quality(img: Image.Image) -> dict:
    """
    Score image quality across multiple dimensions.
    Returns a dict with individual scores and an overall [0–1] score.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    # Sharpness via Laplacian variance
    gray = img.convert("L")
    lap = np.array(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    sharpness = float(np.clip(np.var(lap) / 3000.0, 0.0, 1.0))

    # Brightness
    mean_brightness = float(arr.mean() / 255.0)
    brightness = 1.0 - abs(mean_brightness - 0.5) * 2.0
    brightness = float(np.clip(brightness, 0.0, 1.0))

    # Contrast (std of grayscale)
    stat = ImageStat.Stat(gray)
    contrast = float(np.clip(stat.stddev[0] / 80.0, 0.0, 1.0))

    # Resolution score
    w, h = img.size
    res_score = float(np.clip((w * h) / (1024 * 768), 0.0, 1.0))

    overall = sharpness * 0.40 + brightness * 0.25 + contrast * 0.25 + res_score * 0.10

    return {
        "sharpness":  round(sharpness, 3),
        "brightness": round(brightness, 3),
        "contrast":   round(contrast, 3),
        "resolution": round(res_score, 3),
        "overall":    round(float(np.clip(overall, 0.0, 1.0)), 3),
    }


def preprocess_image(img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize and normalise image for model inference."""
    img_resized = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Report Export ────────────────────────────────────────────────────────────

def build_report_payload(
    predictions: list[dict],
    quality: dict,
    threshold: float,
    top_k: int,
    expert_mode: bool,
    image_filename: str = "uploaded_image",
) -> dict:
    """Build a structured JSON-serialisable report payload."""
    return {
        "report_version": "2.0",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "image": image_filename,
        "settings": {
            "confidence_threshold": threshold,
            "top_k": top_k,
            "expert_mode": expert_mode,
        },
        "image_quality": quality,
        "predictions": [
            {
                "rank":         p["rank"],
                "class_id":     p["class_id"],
                "display_name": p["display_name"],
                "confidence":   round(float(p["confidence"]), 4),
                "severity":     p.get("severity", "Unknown"),
                "pathogen":     p.get("pathogen", "Unknown"),
                "treatment":    p.get("treatment", ""),
                "prevention":   p.get("prevention", ""),
            }
            for p in predictions
        ],
    }


def report_to_json_bytes(payload: dict) -> bytes:
    """Serialise report payload to UTF-8 JSON bytes for download."""
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


# ─── Misc Helpers ─────────────────────────────────────────────────────────────

def confidence_bar_color(confidence: float) -> str:
    """Return a hex colour for a confidence value [0–1]."""
    if confidence >= 0.8:
        return "#22c55e"
    if confidence >= 0.6:
        return "#84cc16"
    if confidence >= 0.4:
        return "#f59e0b"
    return "#ef4444"


def quality_label(score: float) -> Tuple[str, str]:
    """Return (label, colour) for an image quality score."""
    if score >= 0.75:
        return "Excellent", "#22c55e"
    if score >= 0.55:
        return "Good", "#84cc16"
    if score >= 0.35:
        return "Fair", "#f59e0b"
    return "Poor", "#ef4444"


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
