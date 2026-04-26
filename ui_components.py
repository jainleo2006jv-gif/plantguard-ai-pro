"""
PlantGuard AI Pro - UI Components
All rendering is pure Streamlit + st.markdown with embedded CSS.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import SEVERITY_CONFIG, THEME_DARK, THEME_LIGHT
from utils import confidence_bar_color, quality_label


# ─── Theme Injection ───────────────────────────────────────────────────────────

def inject_global_css(dark: bool) -> None:
    """Inject global CSS using the active theme palette."""
    t = THEME_DARK if dark else THEME_LIGHT
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

        :root {{
            --bg-primary:    {t['bg_primary']};
            --bg-secondary:  {t['bg_secondary']};
            --bg-card:       {t['bg_card']};
            --text-primary:  {t['text_primary']};
            --text-secondary:{t['text_secondary']};
            --accent:        {t['accent']};
            --accent-light:  {t['accent_light']};
            --border:        {t['border']};
            --shadow:        {t['shadow']};
        }}

        html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif;
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }}

        /* Streamlit overrides */
        .main .block-container {{
            padding: 1.5rem 2rem;
            max-width: 1200px;
        }}
        .stApp {{
            background: var(--bg-primary) !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: var(--bg-secondary) !important;
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] * {{
            color: var(--text-primary) !important;
        }}

        /* Buttons */
        .stButton > button {{
            background: var(--accent) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.4rem !important;
            transition: opacity 0.2s, transform 0.1s;
        }}
        .stButton > button:hover {{
            opacity: 0.88;
            transform: translateY(-1px);
        }}

        /* Download button */
        .stDownloadButton > button {{
            background: var(--accent-light) !important;
            color: var(--accent) !important;
            border: 1.5px solid var(--accent) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}

        /* Sliders */
        .stSlider [data-baseweb="slider"] {{
            color: var(--accent) !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background: var(--bg-secondary) !important;
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 0.4rem 1rem;
            font-weight: 500;
            color: var(--text-secondary) !important;
        }}
        .stTabs [aria-selected="true"] {{
            background: var(--accent) !important;
            color: #ffffff !important;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-secondary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 6px; }}

        /* Cards */
        .pg-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 12px var(--shadow);
        }}
        .pg-card-accent {{
            border-left: 4px solid var(--accent);
        }}

        /* Metric chips */
        .pg-chip {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}

        /* Hero */
        .pg-hero {{
            text-align: center;
            padding: 2.5rem 1rem 1.5rem;
        }}
        .pg-hero h1 {{
            font-family: 'DM Serif Display', serif;
            font-size: 2.8rem;
            color: var(--accent);
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }}
        .pg-hero p {{
            color: var(--text-secondary);
            font-size: 1.05rem;
            max-width: 560px;
            margin: 0 auto;
        }}

        /* Section headings */
        .pg-section-title {{
            font-family: 'DM Serif Display', serif;
            font-size: 1.3rem;
            color: var(--text-primary);
            margin-bottom: 0.75rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--border);
        }}

        /* Top prediction badge */
        .pg-top-badge {{
            background: linear-gradient(135deg, var(--accent), #1b5e20);
            color: #ffffff !important;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .pg-top-badge .pg-top-name {{
            font-family: 'DM Serif Display', serif;
            font-size: 1.6rem;
        }}
        .pg-top-badge .pg-top-conf {{
            font-size: 0.9rem;
            opacity: 0.85;
        }}

        /* Prediction row */
        .pg-pred-row {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .pg-pred-row:last-child {{ border-bottom: none; }}
        .pg-pred-rank {{
            font-size: 0.75rem;
            font-weight: 700;
            color: var(--text-secondary);
            width: 1.5rem;
            text-align: center;
        }}
        .pg-pred-name {{
            flex: 1;
            font-weight: 500;
            font-size: 0.9rem;
        }}
        .pg-pred-pct {{
            font-weight: 700;
            font-size: 0.9rem;
            min-width: 3.5rem;
            text-align: right;
        }}

        /* Progress bar */
        .pg-progress-wrap {{
            background: var(--border);
            border-radius: 8px;
            height: 6px;
            width: 100%;
            overflow: hidden;
        }}
        .pg-progress-fill {{
            height: 100%;
            border-radius: 8px;
            transition: width 0.4s ease;
        }}

        /* Info grid */
        .pg-info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }}
        .pg-info-item {{
            background: var(--accent-light);
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}
        .pg-info-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        .pg-info-value {{
            font-size: 0.9rem;
            font-weight: 500;
            margin-top: 0.15rem;
        }}

        /* Quality gauge */
        .pg-quality-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.35rem 0;
        }}
        .pg-quality-label {{
            font-size: 0.82rem;
            color: var(--text-secondary);
            width: 90px;
        }}

        /* Alert banner */
        .pg-alert {{
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin: 0.5rem 0;
            font-size: 0.9rem;
        }}
        .pg-alert-warning {{
            background: #fff8e1;
            border-left: 4px solid #f59e0b;
            color: #92400e;
        }}
        .pg-alert-error {{
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            color: #7f1d1d;
        }}
        .pg-alert-success {{
            background: #f0fdf4;
            border-left: 4px solid #22c55e;
            color: #14532d;
        }}
        .pg-alert-info {{
            background: var(--accent-light);
            border-left: 4px solid var(--accent);
            color: var(--text-primary);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Hero Header ──────────────────────────────────────────────────────────────

def render_hero(demo_mode: bool = False) -> None:
    badge = (
        '<span class="pg-chip" style="background:#fff8e1;color:#92400e;">'
        "⚡ Demo Mode – No model loaded</span>"
        if demo_mode
        else ""
    )
    st.markdown(
        f"""
        <div class="pg-hero">
            <h1>🌿 PlantGuard AI Pro</h1>
            <p>Advanced plant disease detection powered by deep learning.
               Upload a leaf image for instant diagnosis and treatment guidance.</p>
            {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Alert Banners ────────────────────────────────────────────────────────────

def alert(message: str, kind: str = "info") -> None:
    """Render a styled alert. kind ∈ {info, warning, error, success}"""
    icons = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
    icon = icons.get(kind, "ℹ️")
    st.markdown(
        f'<div class="pg-alert pg-alert-{kind}">{icon}&nbsp; {message}</div>',
        unsafe_allow_html=True,
    )


# ─── Image Quality Panel ──────────────────────────────────────────────────────

def render_quality_panel(quality: dict) -> None:
    overall_score = quality.get("overall", 0.0)
    label, color  = quality_label(overall_score)

    dims = [
        ("Sharpness",  quality.get("sharpness",  0)),
        ("Brightness", quality.get("brightness", 0)),
        ("Contrast",   quality.get("contrast",   0)),
        ("Resolution", quality.get("resolution", 0)),
    ]
    rows_html = "".join(
        f"""<div class="pg-quality-row">
                <span class="pg-quality-label">{dl}</span>
                <div class="pg-progress-wrap" style="flex:1;margin:0 0.75rem;">
                    <div class="pg-progress-fill" style="width:{int(s*100)}%;background:{confidence_bar_color(s)};"></div>
                </div>
                <span style="font-size:0.78rem;font-weight:600;width:2.5rem;text-align:right;">{int(s*100)}%</span>
            </div>"""
        for dl, s in dims
    )

    st.markdown(
        f"""
        <div class="pg-card">
            <div class="pg-section-title">📊 Image Quality</div>
            <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.9rem;">
                <div style="font-size:2rem;font-weight:800;color:{color};">{int(overall_score*100)}</div>
                <div>
                    <div style="font-weight:700;color:{color};">{label}</div>
                    <div style="font-size:0.75rem;color:var(--text-secondary);">Overall quality score</div>
                </div>
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Prediction Results ───────────────────────────────────────────────────────

def render_top_prediction(pred: object) -> None:
    """Render the hero prediction badge for rank-1 result."""
    sev_cfg = SEVERITY_CONFIG.get(pred.severity, {"color": "#94a3b8", "icon": "❓", "label": pred.severity})
    pct     = int(pred.confidence * 100)
    st.markdown(
        f"""
        <div class="pg-top-badge">
            <div style="font-size:0.78rem;opacity:0.75;margin-bottom:0.3rem;">Top Diagnosis</div>
            <div class="pg-top-name">{pred.display_name}</div>
            <div class="pg-top-conf" style="margin-top:0.4rem;">
                Confidence: <strong>{pct}%</strong> &nbsp;|&nbsp;
                {sev_cfg['icon']} Severity: <strong>{sev_cfg['label']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_list(predictions: list, threshold: float) -> None:
    """Render the ranked prediction rows."""
    rows_html = "".join(
        f"""<div class="pg-pred-row" style="{'opacity:0.55;' if p.confidence < threshold else ''}">
                <span class="pg-pred-rank">#{p.rank}</span>
                <div style="flex:1;">
                    <div class="pg-pred-name">{p.display_name}</div>
                    <div class="pg-progress-wrap" style="margin-top:4px;">
                        <div class="pg-progress-fill" style="width:{int(p.confidence*100)}%;background:{confidence_bar_color(p.confidence)};"></div>
                    </div>
                </div>
                <span class="pg-pred-pct" style="color:{confidence_bar_color(p.confidence)};">{int(p.confidence*100)}%</span>
            </div>"""
        for p in predictions
    )
    st.markdown(
        f'<div class="pg-card"><div class="pg-section-title">🎯 All Predictions</div>{rows_html}</div>',
        unsafe_allow_html=True,
    )


def render_confidence_chart(predictions: list) -> None:
    """Render a Plotly horizontal bar chart for predictions."""
    names  = [p.display_name for p in predictions]
    confs  = [round(p.confidence * 100, 1) for p in predictions]
    colors = [confidence_bar_color(p.confidence) for p in predictions]

    fig = go.Figure(
        go.Bar(
            y           = names[::-1],
            x           = confs[::-1],
            orientation = "h",
            marker_color= colors[::-1],
            text        = [f"{c}%" for c in confs[::-1]],
            textposition= "outside",
        )
    )
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="DM Sans, sans-serif", size=12),
        xaxis         = dict(range=[0, 110], showgrid=False, zeroline=False, showticklabels=False),
        yaxis         = dict(showgrid=False),
        margin        = dict(l=10, r=50, t=10, b=10),
        height        = max(180, len(predictions) * 52),
        showlegend    = False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Disease Intelligence ─────────────────────────────────────────────────────

def render_disease_intelligence(pred: object, expert: bool) -> None:
    """Render disease detail card for the top prediction."""
    if not pred.treatment and not pred.pathogen:
        alert("No detailed disease intelligence available for this class.", "info")
        return

    sev_cfg = SEVERITY_CONFIG.get(pred.severity, {"color": "#94a3b8", "icon": "❓", "label": pred.severity})

    info_items = [
        ("Pathogen",      pred.pathogen or "N/A"),
        ("Severity",      f"{sev_cfg['icon']} {sev_cfg['label']}"),
        ("Spread Risk",   pred.spread_risk   or "N/A"),
        ("Recovery Time", pred.recovery_time or "N/A"),
    ]
    cells = "".join(
        f"""<div class="pg-info-item">
                <div class="pg-info-label">{lbl}</div>
                <div class="pg-info-value">{val}</div>
            </div>"""
        for lbl, val in info_items
    )

    # Render the header + info grid as ONE fully self-contained block
    st.markdown(
        f"""
        <div class="pg-card pg-card-accent">
            <div class="pg-section-title">🧬 Disease Intelligence: {pred.display_name}</div>
            <div class="pg-info-grid" style="margin-top:0.75rem;">{cells}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Treatment & Prevention — use native Streamlit columns (no split divs)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""<div class="pg-card" style="height:100%;">
                <div style="font-weight:700;margin-bottom:0.5rem;">💊 Treatment</div>
                <div style="font-size:0.9rem;line-height:1.6;">{pred.treatment or "No data available."}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="pg-card" style="height:100%;">
                <div style="font-weight:700;margin-bottom:0.5rem;">🛡️ Prevention</div>
                <div style="font-size:0.9rem;line-height:1.6;">{pred.prevention or "No data available."}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    if expert and pred.pathogen and pred.pathogen != "None":
        st.markdown(
            f"""<div class="pg-card" style="margin-top:0.5rem;">
                <div style="font-weight:700;margin-bottom:0.4rem;">🔬 Expert Notes</div>
                <div style="font-size:0.85rem;opacity:0.8;line-height:1.6;">
                    Causative organism: <em>{pred.pathogen}</em>.
                    Molecular confirmation via qPCR or culture is recommended for definitive diagnosis.
                    Resistance profiles may vary by region; consult local extension services before applying fungicides.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ─── Grad-CAM Display ─────────────────────────────────────────────────────────

def render_gradcam(heatmap_array: Optional[np.ndarray], original_img: Image.Image) -> None:
    st.markdown(
        '<div class="pg-card"><div class="pg-section-title">🔥 Grad-CAM Attention Map</div></div>',
        unsafe_allow_html=True,
    )
    if heatmap_array is None:
        alert("Grad-CAM could not be computed for this image or model.", "warning")
        col1, _ = st.columns([1, 1])
        with col1:
            st.image(original_img, caption="Original Image", use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="Original", use_container_width=True)
        with col2:
            st.image(heatmap_array, caption="Attention Heatmap", use_container_width=True)
        st.caption(
            "Warmer regions (red/yellow) indicate areas the model weighted most heavily in its prediction."
        )


# ─── Sidebar Controls ─────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Render all sidebar controls and return a settings dict.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 0.5rem;">
                <span style="font-size:2rem;">🌿</span>
                <div style="font-family:'DM Serif Display',serif;font-size:1.15rem;
                            font-weight:700;margin-top:0.25rem;">PlantGuard AI Pro</div>
                <div style="font-size:0.72rem;opacity:0.6;margin-top:0.1rem;">v2.0 · Powered by Deep Learning</div>
            </div>
            <hr style="border:none;border-top:1px solid var(--border);margin:0.5rem 0 1rem;">
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### ⚙️ Detection Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True, key="dark_mode")
        expert_mode = st.toggle("🔬 Expert Mode", value=False, key="expert_mode")

        st.divider()

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            format="%.0f%%",
            help="Predictions below this threshold are dimmed in results.",
        )

        top_k = st.slider(
            "Top-K Predictions",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Number of candidate diagnoses to display.",
        )

        show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
        show_chart   = st.checkbox("Show Confidence Chart",  value=True)

        st.divider()
        st.markdown(
            """
            <div style="font-size:0.75rem;opacity:0.6;line-height:1.6;">
            <strong>How to get best results</strong><br>
            • Use clear, well-lit leaf photos<br>
            • Fill the frame with the affected area<br>
            • Avoid blurry or overexposed images<br>
            • Single leaf works best
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "dark_mode":            dark_mode,
        "expert_mode":          expert_mode,
        "confidence_threshold": confidence_threshold,
        "top_k":                top_k,
        "show_gradcam":         show_gradcam,
        "show_chart":           show_chart,
    }


# ─── Image Input Panel ────────────────────────────────────────────────────────

def render_image_input() -> tuple[Optional[bytes], str]:
    """
    Render the upload / camera tabs.
    Returns (raw_bytes, source_name).
    """
    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    raw_bytes: Optional[bytes] = None
    source_name = "uploaded_image"

    with tab_upload:
        uploaded = st.file_uploader(
            "Drag & drop or browse a leaf image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            raw_bytes   = uploaded.read()
            source_name = uploaded.name

    with tab_camera:
        camera_photo = st.camera_input("Take a photo of the plant leaf")
        if camera_photo is not None:
            raw_bytes   = camera_photo.read()
            source_name = "camera_capture.jpg"

    return raw_bytes, source_name


# ─── Report Download ───────────────────────────────────────────────────────────

def render_download_button(json_bytes: bytes, filename: str = "plantguard_report.json") -> None:
    st.download_button(
        label     = "📥 Download JSON Report",
        data      = json_bytes,
        file_name = filename,
        mime      = "application/json",
        use_container_width=True,
    )


# ─── Empty State ──────────────────────────────────────────────────────────────

def render_empty_state() -> None:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;opacity:0.55;">
            <div style="font-size:4rem;">🌱</div>
            <div style="font-size:1.1rem;font-weight:600;margin-top:0.75rem;">
                No image loaded yet
            </div>
            <div style="font-size:0.875rem;margin-top:0.4rem;">
                Upload a leaf photo or use the camera to begin diagnosis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )