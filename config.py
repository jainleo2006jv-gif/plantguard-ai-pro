"""
PlantGuard AI Pro - Configuration
"""
from dataclasses import dataclass, field
from typing import Dict, Any

APP_TITLE = "PlantGuard AI Pro"
APP_VERSION = "2.0.0"
APP_ICON = "🌿"

MODEL_INPUT_SIZE = (224, 224)
MODEL_PATH = "plant_disease_model.h5"
TOP_K_DEFAULT = 5
CONFIDENCE_THRESHOLD_DEFAULT = 0.5
IMAGE_QUALITY_MIN = 0.3

GRAD_CAM_LAYER = "conv5_block3_out"
GRAD_CAM_ALPHA = 0.4

DISEASE_METADATA: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "display_name": "Apple Scab",
        "severity": "Moderate",
        "pathogen": "Venturia inaequalis",
        "symptoms": "Olive-green to brown lesions on leaves and fruit. Velvety appearance on undersides.",
        "treatment": "Apply fungicides (captan, myclobutanil) during early season. Remove infected debris.",
        "prevention": "Plant resistant varieties. Ensure good air circulation. Avoid overhead irrigation.",
        "recovery_time": "2–4 weeks with treatment",
        "spread_risk": "High in wet, cool conditions",
    },
    "Apple___Black_rot": {
        "display_name": "Apple Black Rot",
        "severity": "Severe",
        "pathogen": "Botryosphaeria obtusa",
        "symptoms": "Brown lesions with purple borders on leaves. Rotting fruit with black decay.",
        "treatment": "Prune and remove infected branches. Apply copper-based fungicides.",
        "prevention": "Remove mummified fruits. Prune dead wood. Maintain orchard sanitation.",
        "recovery_time": "Seasonal management required",
        "spread_risk": "High; spreads via wind and rain",
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Cedar Apple Rust",
        "severity": "Moderate",
        "pathogen": "Gymnosporangium juniperi-virginianae",
        "symptoms": "Bright orange-yellow spots on leaves; spore tubes on undersides.",
        "treatment": "Fungicides containing myclobutanil or propiconazole at bud break.",
        "prevention": "Remove nearby cedar/juniper hosts. Plant resistant apple varieties.",
        "recovery_time": "3–5 weeks",
        "spread_risk": "Moderate; requires two-host life cycle",
    },
    "Apple___healthy": {
        "display_name": "Healthy Apple",
        "severity": "None",
        "pathogen": "None",
        "symptoms": "No disease symptoms detected.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain regular monitoring and good cultural practices.",
        "recovery_time": "N/A",
        "spread_risk": "None",
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn Gray Leaf Spot",
        "severity": "Moderate to Severe",
        "pathogen": "Cercospora zeae-maydis",
        "symptoms": "Rectangular gray-tan lesions parallel to leaf veins.",
        "treatment": "Apply strobilurin or triazole fungicides at early tassel stage.",
        "prevention": "Use resistant hybrids. Rotate crops. Reduce residue.",
        "recovery_time": "Season-long management",
        "spread_risk": "High in warm, humid conditions",
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato Early Blight",
        "severity": "Moderate",
        "pathogen": "Alternaria solani",
        "symptoms": "Dark concentric ring lesions on lower leaves. Yellow halos around spots.",
        "treatment": "Apply chlorothalonil or copper fungicides. Remove affected leaves.",
        "prevention": "Mulch soil. Avoid wetting foliage. Rotate crops.",
        "recovery_time": "2–3 weeks with treatment",
        "spread_risk": "Moderate; spreads in warm, wet conditions",
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato",
        "severity": "None",
        "pathogen": "None",
        "symptoms": "No disease symptoms detected.",
        "treatment": "No treatment needed.",
        "prevention": "Maintain regular monitoring and good cultural practices.",
        "recovery_time": "N/A",
        "spread_risk": "None",
    },
}

SEVERITY_CONFIG = {
    "None":               {"color": "#22c55e", "icon": "✅", "label": "Healthy"},
    "Low":                {"color": "#84cc16", "icon": "🟡", "label": "Low Risk"},
    "Moderate":           {"color": "#f59e0b", "icon": "⚠️", "label": "Moderate"},
    "Moderate to Severe": {"color": "#f97316", "icon": "🔶", "label": "Mod-Severe"},
    "Severe":             {"color": "#ef4444", "icon": "🔴", "label": "Severe"},
    "Critical":           {"color": "#dc2626", "icon": "🚨", "label": "Critical"},
}

THEME_LIGHT = {
    "bg_primary":    "#f8faf5",
    "bg_secondary":  "#ffffff",
    "bg_card":       "#ffffff",
    "text_primary":  "#1a2e1a",
    "text_secondary":"#4a6741",
    "accent":        "#2d7a2d",
    "accent_light":  "#e8f5e9",
    "border":        "#c8e6c9",
    "shadow":        "rgba(45,122,45,0.12)",
}

THEME_DARK = {
    "bg_primary":    "#0d1a0d",
    "bg_secondary":  "#132013",
    "bg_card":       "#1a2e1a",
    "text_primary":  "#e8f5e9",
    "text_secondary":"#81c784",
    "accent":        "#4caf50",
    "accent_light":  "#1b3a1b",
    "border":        "#2d5a2d",
    "shadow":        "rgba(76,175,80,0.15)",
}
