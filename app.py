"""
PlantGuard AI Pro v2.0
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import datetime
import logging
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image

# ── Local modules ──────────────────────────────────────────────────────────────
from config import (
    APP_ICON,
    APP_TITLE,
    APP_VERSION,
    CONFIDENCE_THRESHOLD_DEFAULT,
    DISEASE_METADATA,
    GRAD_CAM_ALPHA,
    GRAD_CAM_LAYER,
    IMAGE_QUALITY_MIN,
    MODEL_INPUT_SIZE,
    MODEL_PATH,
    TOP_K_DEFAULT,
)
from inference import PLANT_VILLAGE_CLASSES, compute_gradcam, run_inference
from ui_components import (
    alert,
    inject_global_css,
    render_confidence_chart,
    render_disease_intelligence,
    render_download_button,
    render_empty_state,
    render_gradcam,
    render_hero,
    render_image_input,
    render_prediction_list,
    render_quality_panel,
    render_sidebar,
    render_top_prediction,
)
from utils import (
    build_report_payload,
    compute_image_quality,
    load_image_from_bytes,
    preprocess_image,
    report_to_json_bytes,
    validate_image,
)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    stream  = sys.stdout,
)
logger = logging.getLogger(__name__)


# ─── Streamlit page config (must be first st call) ────────────────────────────
st.set_page_config(
    page_title     = APP_TITLE,
    page_icon      = APP_ICON,
    layout         = "wide",
    initial_sidebar_state = "expanded",
)


# ─── Session State Initialisation ─────────────────────────────────────────────

def _init_session_state() -> None:
    defaults = {
        "last_predictions":   None,
        "last_quality":       None,
        "last_report_bytes":  None,
        "last_image":         None,
        "last_source_name":   None,
        "last_gradcam":       None,
        "analysis_complete":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Core Analysis Pipeline ───────────────────────────────────────────────────

def _run_analysis(
    raw_bytes:  bytes,
    source_name: str,
    settings:   dict,
) -> None:
    """
    Full analysis pipeline:
      1. Load & validate image
      2. Quality assessment
      3. Preprocess
      4. Model inference
      5. (Optional) Grad-CAM
      6. Build report
    Stores all results in st.session_state.
    """
    # 1. Load
    img = load_image_from_bytes(raw_bytes)
    if img is None:
        alert("Could not decode the uploaded image. Please try another file.", "error")
        return

    # 2. Validate
    valid, msg = validate_image(img)
    if not valid:
        alert(f"Image validation failed: {msg}", "error")
        return

    # 3. Quality
    quality = compute_image_quality(img)
    if quality["overall"] < IMAGE_QUALITY_MIN:
        alert(
            f"Image quality is very low ({int(quality['overall']*100)}%). "
            "Results may be unreliable. Try a sharper, better-lit photo.",
            "warning",
        )

    # 4. Preprocess & infer
    arr         = preprocess_image(img, MODEL_INPUT_SIZE)
    predictions = run_inference(
        image_array = arr,
        model_path  = MODEL_PATH,
        top_k       = settings["top_k"],
        class_names = PLANT_VILLAGE_CLASSES,
        metadata    = DISEASE_METADATA,
    )

    # 5. Grad-CAM (best effort)
    gradcam_overlay = None
    if settings.get("show_gradcam") and predictions:
        gradcam_overlay = compute_gradcam(
            image_array  = arr,
            model_path   = MODEL_PATH,
            class_index  = predictions[0].class_index,
            layer_name   = GRAD_CAM_LAYER,
            alpha        = GRAD_CAM_ALPHA,
            original_size= MODEL_INPUT_SIZE,
        )

    # 6. Report payload
    pred_dicts  = [p.to_dict() for p in predictions]
    report_blob = build_report_payload(
        predictions     = pred_dicts,
        quality         = quality,
        threshold       = settings["confidence_threshold"],
        top_k           = settings["top_k"],
        expert_mode     = settings["expert_mode"],
        image_filename  = source_name,
    )

    # Store in session state
    st.session_state.last_predictions  = predictions
    st.session_state.last_quality      = quality
    st.session_state.last_report_bytes = report_to_json_bytes(report_blob)
    st.session_state.last_image        = img
    st.session_state.last_source_name  = source_name
    st.session_state.last_gradcam      = gradcam_overlay
    st.session_state.analysis_complete = True

    logger.info(
        "Analysis complete: top=%s conf=%.2f quality=%.2f",
        predictions[0].display_name if predictions else "none",
        predictions[0].confidence   if predictions else 0,
        quality["overall"],
    )


# ─── Results Rendering ────────────────────────────────────────────────────────

def _render_results(settings: dict) -> None:
    """Render all result panels from session state."""
    predictions = st.session_state.last_predictions
    quality     = st.session_state.last_quality
    img         = st.session_state.last_image
    gradcam     = st.session_state.last_gradcam
    report_b    = st.session_state.last_report_bytes
    source_name = st.session_state.last_source_name or "image"

    if not predictions:
        alert("No predictions could be generated.", "error")
        return

    threshold = settings["confidence_threshold"]
    top_pred  = predictions[0]

    # ── Confidence gating ──────────────────────────────────────────────────
    if top_pred.confidence < threshold:
        alert(
            f"Top prediction confidence ({int(top_pred.confidence*100)}%) is below your "
            f"threshold ({int(threshold*100)}%). The image may not contain a recognisable "
            "plant leaf, or the model is uncertain.",
            "warning",
        )

    # ── Layout: two main columns ───────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        # Image preview — title as self-contained block, image rendered natively
        st.markdown(
            '<div class="pg-card"><div class="pg-section-title">📸 Analysed Image</div></div>',
            unsafe_allow_html=True,
        )
        st.image(img, use_container_width=True)

        # Quality
        if quality:
            render_quality_panel(quality)

        # Download — title as self-contained block, button rendered natively
        st.markdown(
            '<div class="pg-card"><div class="pg-section-title">📄 Export</div></div>',
            unsafe_allow_html=True,
        )
        render_download_button(
            report_b,
            filename=f"plantguard_{source_name.rsplit('.', 1)[0]}_{datetime.date.today()}.json",
        )

    with col_right:
        # Top prediction
        render_top_prediction(top_pred)

        # Prediction tabs
        tab_list, tab_chart = st.tabs(["📋 Ranked List", "📊 Chart"])
        with tab_list:
            render_prediction_list(predictions, threshold)
        with tab_chart:
            if settings.get("show_chart"):
                render_confidence_chart(predictions)
            else:
                alert("Enable 'Show Confidence Chart' in the sidebar.", "info")

    # ── Disease Intelligence (full width) ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    render_disease_intelligence(top_pred, settings["expert_mode"])

    # ── Grad-CAM (full width) ─────────────────────────────────────────────
    if settings.get("show_gradcam"):
        render_gradcam(gradcam, img)


# ─── Main Application ─────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()

    # Sidebar (controls)
    settings = render_sidebar()

    # CSS injection (depends on dark mode toggle)
    inject_global_css(dark=settings["dark_mode"])

    # Demo mode detection
    demo_mode = not os.path.isfile(MODEL_PATH)

    # Hero header
    render_hero(demo_mode=demo_mode)

    st.markdown("---")

    # ── Input area ────────────────────────────────────────────────────────
    raw_bytes, source_name = render_image_input()

    # Trigger analysis when new image is provided
    if raw_bytes is not None:
        # Only re-run if input changed (compare hash)
        new_hash = hash(raw_bytes)
        if st.session_state.get("_last_hash") != new_hash:
            st.session_state["_last_hash"] = new_hash
            st.session_state.analysis_complete = False

            with st.spinner("🌿 Analysing plant image…"):
                _run_analysis(raw_bytes, source_name, settings)

    # ── Results ───────────────────────────────────────────────────────────
    if st.session_state.analysis_complete and st.session_state.last_predictions:
        st.markdown("---")
        _render_results(settings)
    elif raw_bytes is None:
        render_empty_state()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center;padding:2rem 0 1rem;opacity:0.4;font-size:0.75rem;">
            PlantGuard AI Pro {APP_VERSION} &nbsp;·&nbsp;
            For research and educational purposes only.
            Not a substitute for professional agronomic advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()