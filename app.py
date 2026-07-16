# DermAI - streamlit app for the skin lesion classifier
# loads the trained EfficientNetB0 model, asks a few quick questions, takes a photo, shows a prediction

import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image

from model import CLASS_NAMES, IMG_SIZE

# class display names, risk tier, and content for the glossary and condition card
CLASS_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "short_name": "Actinic Keratosis",  # the full name overflows the narrow glossary badge
        "risk": "Medium",
        "glossary": "Rough, scaly patches from sun damage that can sometimes progress to cancer if untreated.",
        "description": "A rough, scaly patch caused by years of sun exposure. It's precancerous - not yet invasive cancer, but it can develop into squamous cell carcinoma if left untreated.",
        "symptoms": [
            "Rough, dry, or scaly patch of skin",
            "Pink, red, or brown coloring",
            "Feels like sandpaper to the touch",
            "Usually on sun-exposed skin (face, ears, scalp, hands)",
        ],
        "see_doctor": "See a dermatologist if a rough patch doesn't go away, grows, or changes - it's simple and highly effective to treat early.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "risk": "High",
        "glossary": "The most common skin cancer - grows slowly and rarely spreads, but needs prompt treatment.",
        "description": "The most common type of skin cancer. It grows slowly and rarely spreads elsewhere in the body, but can damage surrounding tissue if left untreated.",
        "symptoms": [
            "A pearly or waxy bump",
            "A flat, flesh-colored or brown scar-like lesion",
            "A sore that bleeds, oozes, or crusts and won't fully heal",
            "Visible small blood vessels on the surface",
        ],
        "see_doctor": "See a doctor promptly for any sore that won't heal or a new, persistent bump - this is very treatable when caught early.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "risk": "Low",
        "glossary": "A harmless, non-cancerous growth, often related to aging or sun exposure.",
        "description": "A common, harmless skin growth (like a seborrheic keratosis). It's not cancerous and doesn't need treatment unless it's irritating or bothers you cosmetically.",
        "symptoms": [
            "Waxy, \"stuck-on\" appearance",
            "Brown, black, or tan coloring",
            "Slightly raised, rough texture",
            "Usually appears with age",
        ],
        "see_doctor": "Generally harmless - see a doctor if a spot changes rapidly or looks different from your other moles, just to rule anything else out.",
    },
    "df": {
        "name": "Dermatofibroma",
        "risk": "Low",
        "glossary": "A small, firm, benign nodule, often on the legs, sometimes after a minor injury.",
        "description": "A common, benign skin nodule made of fibrous tissue. Harmless, and often forms after a minor skin injury like a bug bite or small cut.",
        "symptoms": [
            "Small, firm bump under the skin",
            "Brown, pink, or reddish color",
            "Dimples inward when pinched from the sides",
            "Usually on the lower legs or arms",
        ],
        "see_doctor": "Almost always harmless - see a doctor if it grows quickly, bleeds, or becomes painful.",
    },
    "mel": {
        "name": "Melanoma",
        "risk": "High",
        "glossary": "The most serious skin cancer - can spread quickly if not caught early.",
        "description": "The most dangerous form of skin cancer. It can grow and spread quickly, but is highly treatable when caught early - which is why prompt evaluation matters.",
        "symptoms": [
            "A mole that's new or changing in size, shape, or color",
            "Asymmetrical shape or irregular, blurry border",
            "Multiple colors within one spot",
            "Diameter larger than a pencil eraser (>6mm)",
            "Evolving - changing over time",
        ],
        "see_doctor": "See a dermatologist as soon as possible for any mole that's new, changing, or matches the warning signs above. Don't wait.",
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "risk": "Low",
        "glossary": "A common mole - almost always harmless, but worth monitoring for changes.",
        "description": "The medical term for a common mole. The vast majority are completely harmless and just a normal part of skin - still worth keeping an eye on for changes.",
        "symptoms": [
            "Round or oval shape",
            "Even coloring (tan, brown, or black)",
            "Stable in size and appearance over time",
            "Smooth, well-defined border",
        ],
        "see_doctor": "See a doctor if a mole starts changing in size, shape, or color, or becomes itchy, painful, or starts bleeding.",
    },
    "vasc": {
        "name": "Vascular Lesion",
        "risk": "Low",
        "glossary": "A benign mark from blood vessels near the skin's surface, like a birthmark.",
        "description": "A benign mark caused by blood vessels close to the surface of the skin, like a cherry angioma or birthmark. Extremely common and typically harmless.",
        "symptoms": [
            "Red, purple, or blue coloring",
            "Flat or slightly raised",
            "May briefly lighten when pressed",
            "Usually stable over time",
        ],
        "see_doctor": "Usually harmless - see a doctor if a spot suddenly grows, bleeds easily, or changes noticeably.",
    },
}

RISK_LEVELS = ["Low", "Medium", "High"]

# sky blue brand palette. #38BDF8 is too light for text on white (2.1:1
# contrast), so ACCENT_TEXT is a deeper shade for text/labels
ACCENT = "#38BDF8"           # illustration linework, muted chart bars, dropzone
ACCENT_TEXT = "#0369A1"      # blue text sitting directly on white (headings, labels)
ACCENT_TINT = "#E0F2FE"      # light background wash (dropzone, callout banner)
ACCENT_MUTED = "#BAE6FD"     # non-highlighted chart bars

TEXT_PRIMARY = "#1E293B"     # charcoal
TEXT_SECONDARY = "#475569"   # softer slate for secondary copy
GRIDLINE = "#E2E8F0"
SURFACE = "#FFFFFF"

# risk badge colors for the results panel - kept separate from the brand color
# so risk always reads clearly
STATUS_COLORS = {
    "Low": {"color": "#086b08", "bg": "#e9f7e9", "icon": "✓"},
    "Medium": {"color": "#7a4d06", "bg": "#fdf3dc", "icon": "⚠"},
    "High": {"color": "#a02a2a", "bg": "#fbe9e9", "icon": "✕"},
}


# page setup + the white/sky-blue theme
st.set_page_config(
    page_title="DermAI",
    page_icon="\U0001FA79",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    /* hide the default streamlit chrome, keep it clean */
    #MainMenu, footer, header {{ visibility: hidden; }}

    .stApp {{
        background-color: {SURFACE};
        /* soft sky blue spotlight glow from the top, fading to white */
        background-image: radial-gradient(ellipse 1600px 900px at 50% -5%, {ACCENT_MUTED} 0%, rgba(224, 242, 254, 0.55) 35%, rgba(224, 242, 254, 0) 70%);
        background-attachment: fixed;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        color: {TEXT_PRIMARY};
    }}

    /* solid turquoise bar pinned to the very top of the page */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 40px;
        background: #0EA5E9;
        z-index: 1000;
    }}

    /* hero title - big logo-like wordmark */
    h1 {{
        color: {TEXT_PRIMARY};
        font-weight: 900;
        font-size: 5rem;
        letter-spacing: -0.03em;
    }}
    h1 .accent {{ color: {ACCENT_TEXT}; }}
    h1 .logo-icon {{ font-size: 0.75em; vertical-align: -0.05em; }}

    /* fade + slide up on load (opacity 0->1, translateY 30px->0). only
    applied on the first script run (session_state gate below) so it
    doesn't replay on every widget interaction */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    h1.fade-in-up {{ animation: fadeInUp 0.7s ease-out forwards; }}

    h2, h3, h4 {{ color: {TEXT_PRIMARY}; font-weight: 700; }}

    /* hero tagline/description/who-for text */
    .st-key-hero_copy p {{ color: {TEXT_SECONDARY}; margin-bottom: 0.4rem; }}
    .st-key-hero_copy p:first-child {{ font-size: 1.15rem; font-weight: 500; color: {TEXT_PRIMARY}; }}

    /* early-detection stat callout - subtle sky blue banner */
    .stat-callout {{
        background-color: {ACCENT_TINT};
        border-left: 4px solid {ACCENT};
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin: 1.25rem 0 0.5rem 0;
        color: {TEXT_PRIMARY};
        font-weight: 500;
    }}

    /* cards - white, rounded, soft shadow instead of a hard border. targeted
    by key (st-key-...) since border=True alone gives a plain 8px-radius
    border with no reliable selector to override per-card */
    .st-key-glossary_card, .st-key-questionnaire_card, .st-key-upload_card,
    .st-key-results_card, .st-key-condition_info_card {{
        border-radius: 20px !important;
        border: 1px solid rgba(30, 41, 59, 0.06) !important;
        box-shadow: 0 1px 2px rgba(30, 41, 59, 0.04), 0 8px 24px rgba(30, 41, 59, 0.06) !important;
    }}
    .section-title {{
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {ACCENT_TEXT};
        margin-bottom: 0.75rem;
    }}

    /* results card - sky blue background, dark charcoal text throughout */
    .st-key-results_card {{
        background-color: {ACCENT} !important;
    }}
    .st-key-results_card .section-title,
    .st-key-results_card h3,
    .st-key-results_card h4,
    .st-key-results_card p {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* analyze button - deeper blue than ACCENT so white text has contrast */
    .stButton > button {{
        background-color: {ACCENT_TEXT};
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        width: 100%;
        transition: background-color 0.15s ease;
    }}
    .stButton > button:hover {{
        background-color: #075985;
        color: #FFFFFF;
    }}

    /* drag-and-drop upload zone */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {ACCENT_TINT};
        border: 2px dashed {ACCENT};
        border-radius: 12px;
    }}

    /* label text color */
    .stSelectbox label {{
        color: {TEXT_SECONDARY};
        font-weight: 500;
    }}

    /* risk pill (results panel + condition glossary) - colored background,
    not just colored text, so risk reads at a glance */
    .risk-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1rem;
    }}
    .risk-badge-sm {{
        padding: 0.25rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }}

    /* how-it-works steps */
    .how-it-works-step {{
        text-align: center;
        padding: 1rem;
    }}
    .how-it-works-icon {{
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }}
    .how-it-works-step h4 {{
        margin: 0 0 0.3rem 0;
        font-size: 1.05rem;
    }}
    .how-it-works-step p {{
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
        margin: 0;
    }}

    /* right column flip card - skin health tips on the front, condition info
    on the back once there's a result. one raw HTML block since the 3D flip
    needs both faces absolutely positioned inside a shared perspective
    parent; flip animation is gated in python (see render_flip_card) */
    .flip-card {{
        perspective: 1500px;
        height: 640px;
        margin-bottom: 1rem;
    }}
    .flip-card-inner {{
        position: relative;
        width: 100%;
        height: 100%;
        transform-style: preserve-3d;
    }}
    .flip-card-inner.flipped {{ transform: rotateY(180deg); }}
    @keyframes flipForward {{ from {{ transform: rotateY(0deg); }} to {{ transform: rotateY(180deg); }} }}
    @keyframes flipBackward {{ from {{ transform: rotateY(180deg); }} to {{ transform: rotateY(0deg); }} }}
    .flip-card-inner.animate.flipped {{ animation: flipForward 0.7s ease forwards; }}
    .flip-card-inner.animate:not(.flipped) {{ animation: flipBackward 0.7s ease forwards; }}
    .flip-card-front, .flip-card-back {{
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        backface-visibility: hidden;
        -webkit-backface-visibility: hidden;
        border-radius: 20px;
        border: 1px solid rgba(30, 41, 59, 0.06);
        box-shadow: 0 1px 2px rgba(30, 41, 59, 0.04), 0 8px 24px rgba(30, 41, 59, 0.06);
        background-color: {SURFACE};
        padding: 1.5rem;
        overflow-y: auto;
    }}
    .flip-card-back {{ transform: rotateY(180deg); }}
    .flip-card-front .tip-icon {{
        font-size: 2.2rem;
        text-align: center;
        margin-bottom: 0.25rem;
    }}
    .flip-card-front h4 {{
        text-align: center;
        margin: 0 0 1rem 0;
    }}
    .flip-card-front ul {{
        margin: 0;
        padding-left: 1.1rem;
    }}
    .flip-card-front li {{
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        line-height: 1.4;
    }}

    /* invalid image error card - reuses the same red as the High risk badge */
    .error-card {{
        background-color: #fbe9e9;
        border-left: 4px solid #d03b3b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #a02a2a;
    }}
    .error-card strong {{ font-size: 1.05rem; }}

    /* footer disclaimer */
    .disclaimer {{
        text-align: center;
        color: {TEXT_SECONDARY};
        font-size: 0.85rem;
        border-top: 1px solid {GRIDLINE};
        padding-top: 1.25rem;
        margin-top: 2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# cache it so we don't reload the model on every rerun
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model("skin_lesion_classifier.keras")


# same preprocessing as training - resize + scale to 0-1 (model undoes the scaling internally)
def preprocess_image(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return tf.expand_dims(image, axis=0)  # add batch dimension


MIN_DIMENSION = 100  # anything smaller than 100x100 is too low quality to bother with

# sanity-check gate before the real model runs - not a classifier, just catches
# obviously-wrong uploads (screenshots, random photos, blank/tiny images).
# thresholds are tuned with margin so real HAM10000 photos (including pale
# or low-contrast skin) don't get rejected
def validate_lesion_image(pil_image):
    width, height = pil_image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False, "too small"

    rgb = pil_image.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    hsv = np.asarray(rgb.convert("HSV")).astype(np.float32)
    hue = hsv[..., 0] / 255.0 * 360.0
    sat = hsv[..., 1]
    val = hsv[..., 2]

    # check 2: color distribution - any plausible skin-toned pixels at all.
    # broad hue window since skin runs pale pink to deep brown, and low
    # saturation is normal for pale or overexposed skin, not just screenshots
    skin_mask = ((hue <= 60) | (hue >= 330)) & (sat >= 8) & (val >= 30)
    skin_fraction = float(skin_mask.mean())
    if skin_fraction < 0.02:
        return False, "not skin-toned"

    # check 3: contrast/texture - flat solid-color blocks, blown-out
    # brightness, or a screenshot's wall of flat/desaturated pixels (UI
    # chrome, body text)
    gray = arr.mean(axis=2)
    contrast = float(gray.std())
    brightness = float(gray.mean())
    low_sat_fraction = float((sat < 15).mean())
    if contrast < 8 or contrast > 75:
        return False, "flat or noisy texture"
    if brightness > 220:
        return False, "too bright"
    if low_sat_fraction > 0.5:
        return False, "looks like a screenshot"

    return True, None


def render_invalid_image_error():
    st.markdown(
        '<div class="error-card"><strong>Invalid image</strong> — This does not appear to be a skin '
        "lesion photo. Please upload a clear close-up clinical or dermoscopy image of a skin lesion "
        "for accurate results.</div>",
        unsafe_allow_html=True,
    )


# out-of-distribution check - catches things that dodge the pixel heuristics
# above but still aren't real lesion photos (a realistic AI-generated skin/face
# image, say). compares the upload's EfficientNetB0 backbone embedding against
# a reference set of real HAM10000 embeddings; see precompute_reference_embeddings.py
@st.cache_resource(show_spinner=False)
def load_reference_embeddings():
    data = np.load("reference_embeddings.npz")
    return data["embeddings"], float(data["threshold"]), int(data["k_neighbors"])


@st.cache_resource(show_spinner=False)
def build_embedding_model():
    model = load_model()
    # nested submodel output isn't directly connectable in a new Model(), so
    # rebuild the forward pass explicitly instead of slicing get_layer().output
    inputs = model.input
    x = model.get_layer("rescaling_2")(inputs)
    x = model.get_layer("efficientnetb0")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def is_out_of_distribution(image_tensor):
    reference_embeddings, threshold, k = load_reference_embeddings()
    embedding = build_embedding_model().predict(image_tensor, verbose=0)
    embedding = embedding / np.clip(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8, None)
    dists = np.linalg.norm(reference_embeddings[None, :, :] - embedding[:, None, :], axis=2)
    distance = float(np.sort(dists, axis=1)[:, :k].mean())
    return distance > threshold


# start from the class's base risk, bump it up if they said yes to changed or family history
# never lower it, only escalate
def adjust_risk(base_risk, changed, family_history):
    concern_points = (changed == "Yes") + (family_history == "Yes")
    idx = min(RISK_LEVELS.index(base_risk) + concern_points, len(RISK_LEVELS) - 1)
    return RISK_LEVELS[idx]


def render_risk_badge(risk):
    style = STATUS_COLORS[risk]
    st.markdown(
        f"""
        <span class="risk-badge" style="background-color:{style['bg']}; color:{style['color']};">
            {style['icon']} {risk} Risk
        </span>
        """,
        unsafe_allow_html=True,
    )


# semi-circular confidence dial - white needle/arc on the results card's sky
# blue background, matching the same white-on-blue treatment as the chart
def render_confidence_gauge(confidence):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={"suffix": "%", "font": {"color": TEXT_PRIMARY, "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": TEXT_PRIMARY, "tickfont": {"color": TEXT_PRIMARY, "size": 10}},
                "bar": {"color": "#FFFFFF"},
                "bgcolor": "rgba(255, 255, 255, 0.25)",
                "borderwidth": 0,
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=ACCENT,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=TEXT_PRIMARY),
        height=180,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig


# horizontal bar chart of all 7 probabilities, living on the results card's
# sky blue background now - predicted class gets a solid white bar, the rest
# stay a translucent white so it's obvious what won without disappearing
# against the blue
def render_probability_chart(probabilities, predicted_class):
    order = np.argsort(probabilities)[::-1]
    classes_sorted = [CLASS_NAMES[i] for i in order]
    labels_sorted = [f"{CLASS_INFO[c]['name']} ({c})" for c in classes_sorted]
    probs_sorted = [float(probabilities[i]) * 100 for i in order]
    colors = [
        "#FFFFFF" if c == predicted_class else "rgba(255, 255, 255, 0.45)"
        for c in classes_sorted
    ]

    fig = go.Figure(
        go.Bar(
            x=probs_sorted,
            y=labels_sorted,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.1f}%" for p in probs_sorted],
            textposition="outside",
            textfont=dict(color=TEXT_PRIMARY, size=13),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis=dict(
            range=[0, max(probs_sorted) * 1.25],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.35)",
            ticksuffix="%",
            title=None,
        ),
        yaxis=dict(autorange="reversed", title=None, automargin=True),
        plot_bgcolor=ACCENT,
        paper_bgcolor=ACCENT,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=TEXT_PRIMARY),
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        bargap=0.35,
        showlegend=False,
    )
    return fig


# left column - the always-visible condition glossary
def render_glossary():
    with st.container(border=True, key="glossary_card"):
        st.markdown('<div class="section-title">Condition glossary</div>', unsafe_allow_html=True)
        for code in CLASS_NAMES:
            info = CLASS_INFO[code]
            style = STATUS_COLORS[info["risk"]]
            label = info.get("short_name", info["name"])
            # a real colored pill (bg + text), not just colored text -
            # reuses the same risk-badge look as the results panel
            st.markdown(
                f'<span class="risk-badge risk-badge-sm" '
                f'style="background-color:{style["bg"]}; color:{style["color"]};">'
                f'{style["icon"]} {label}</span>',
                unsafe_allow_html=True,
            )
            st.caption(info["glossary"])


SKIN_HEALTH_TIPS = [
    "Dermatologists recommend broad-spectrum SPF 50+ sunscreen — apply daily, even on cloudy days",
    "Perform a monthly self skin check from head to toe",
    "Avoid tanning beds — they increase melanoma risk by 75%",
    "Schedule an annual skin exam with a board-certified dermatologist",
]


# right column - a flip card. front = general skin health tips (default view).
# back = the predicted condition's info, shown once a result exists. flips
# automatically based on session_state, not user interaction
def render_flip_card():
    result = st.session_state.get("result")
    predicted_class = result["predicted_class"] if result else None

    # card_flipped is the logical state; card_just_transitioned marks whether
    # that state changed on this rerun specifically, so the animation class
    # only gets attached on an actual flip, not every unrelated rerun
    if "card_flipped" not in st.session_state:
        st.session_state.card_flipped = False
    if "card_just_transitioned" not in st.session_state:
        st.session_state.card_just_transitioned = False

    flipped = st.session_state.card_flipped
    animate = st.session_state.card_just_transitioned
    st.session_state.card_just_transitioned = False  # consume it - one-shot

    classes = "flip-card-inner"
    if flipped:
        classes += " flipped"
    if animate:
        classes += " animate"

    tips_html = "".join(f"<li>{tip}</li>" for tip in SKIN_HEALTH_TIPS)
    front_html = (
        '<div class="flip-card-front">'
        '<div class="tip-icon">☀️</div>'
        "<h4>Skin Health Tips</h4>"
        f"<ul>{tips_html}</ul>"
        "</div>"
    )

    if predicted_class:
        info = CLASS_INFO[predicted_class]
        symptoms_html = "".join(f"<li>{s}</li>" for s in info["symptoms"])
        back_html = (
            '<div class="flip-card-back">'
            '<div class="section-title">About this condition</div>'
            f"<p><strong>{info['name']}</strong></p>"
            f"<p>{info['description']}</p>"
            "<p><strong>Common symptoms to watch for</strong></p>"
            f"<ul>{symptoms_html}</ul>"
            "<p><strong>When to see a doctor</strong></p>"
            f'<p style="color:{TEXT_SECONDARY}; font-size:0.85rem;">{info["see_doctor"]}</p>'
            "</div>"
        )
    else:
        # empty placeholder - never actually visible since the card only
        # flips to the back once a result (and thus a predicted_class) exists
        back_html = '<div class="flip-card-back"></div>'

    st.markdown(
        f'<div class="flip-card"><div class="{classes}">{front_html}{back_html}</div></div>',
        unsafe_allow_html=True,
    )


# hero section
# tagline/description/who-for lines use plain st.markdown (not raw HTML) so
# the "who it's for" line's :material/icon: still renders - raw HTML skips
# streamlit's own icon substitution
# play the load-in animation only once per session, not on every rerun
_hero_class = "fade-in-up" if "hero_animated" not in st.session_state else ""
st.session_state.hero_animated = True
st.markdown(
    f'<h1 class="{_hero_class}"><span class="logo-icon">✨</span> Derm<span class="accent">AI</span></h1>',
    unsafe_allow_html=True,
)
with st.container(key="hero_copy"):
    st.markdown("AI-powered skin lesion screening for early detection")
    st.markdown(
        "Built on 10,000+ clinical dermoscopy images, DermAI helps identify "
        "7 skin conditions for educational and research purposes."
    )
    st.markdown(
        ":material/group: For anyone concerned about a skin lesion - patients, "
        "caregivers, medical students, and researchers."
    )
st.markdown(
    '<div class="stat-callout">Early detection matters ⚠️ Melanoma has a '
    f'<strong style="color:{STATUS_COLORS["Low"]["color"]};">99%</strong> five-year survival rate when '
    f'caught early, dropping to <strong style="color:{STATUS_COLORS["High"]["color"]};">35%</strong> '
    "if it spreads. Don't wait.</div>",
    unsafe_allow_html=True,
)

# how it works: 3 steps between the hero and the main app
# plain emoji here, not :material/icon: shortcodes - those only render
# inside streamlit's own markdown pipeline, not raw HTML
HOW_IT_WORKS_STEPS = [
    ("📤", "Upload photo", "Take or upload a clear close-up photo of the lesion you're concerned about."),
    ("🔬", "AI analysis", "DermAI compares it against patterns learned from 10,000+ clinical images."),
    ("✅", "Get results", "See the predicted condition, a risk level, and what to watch for."),
]
step_cols = st.columns(3, gap="medium")
for step_col, (icon, title, description) in zip(step_cols, HOW_IT_WORKS_STEPS):
    with step_col:
        st.markdown(
            f'<div class="how-it-works-step">'
            f'<div class="how-it-works-icon">{icon}</div>'
            f"<h4>{title}</h4>"
            f"<p>{description}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

# three-column layout: condition glossary | main flow | condition info.
# streamlit's real st.sidebar is a single left-only panel, so a left+right
# pair is just three regular columns instead
glossary_col, main_col, info_col = st.columns([1, 2, 1], gap="medium")

with glossary_col:
    render_glossary()

with main_col:
    # questionnaire
    # st.container(border=True) rather than raw open/close divs - markdown
    # calls don't nest across separate st.markdown calls
    with st.container(border=True, key="questionnaire_card"):
        st.markdown('<div class="section-title">Symptom questionnaire</div>', unsafe_allow_html=True)

        duration = st.selectbox(
            ":material/calendar_month: How long have you had this lesion?",
            ["Less than 1 month", "1-6 months", "6-12 months", "More than 1 year", "Not sure"],
        )
        changed = st.segmented_control(
            ":material/trending_up: Has it changed in size, shape, or color recently?",
            ["No", "Yes", "Not sure"],
            default="No",
        )
        family_history = st.segmented_control(
            ":material/groups: Do you have a family history of skin cancer?",
            ["No", "Yes", "Not sure"],
            default="No",
        )

    # image upload
    with st.container(border=True, key="upload_card"):
        st.markdown('<div class="section-title">Upload a lesion photo</div>', unsafe_allow_html=True)

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Drag and drop an image here, or click to browse",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image", width="stretch")

        analyze_clicked = st.button("Analyze image", icon=":material/biotech:", width="stretch")
        st.caption("Accepted formats: close-up skin lesion photos, dermoscopy images.")

    # analysis - runs once when the button's clicked, then stashes the result
    # in session_state so it (and the "scan another image" button) survive
    # later reruns. analyze_clicked itself reverts to False on the very next
    # rerun, so anything only rendered inside `if analyze_clicked:` would
    # otherwise vanish before a button inside it could ever be clicked
    if analyze_clicked:
        if uploaded_file is None:
            st.warning("Please upload an image before analyzing.")
            st.session_state.result = None
        elif not validate_lesion_image(Image.open(io.BytesIO(uploaded_file.getvalue())))[0]:
            # runs before the real model - catches non-skin uploads so we
            # don't waste a prediction (or give a misleading one) on them
            render_invalid_image_error()
            st.session_state.result = None
        else:
            image_tensor = preprocess_image(uploaded_file)
            with st.spinner("Analyzing image..."):
                model = load_model()
                # second line of defense - pixel stats can miss a realistic
                # non-lesion image, so also check it against real embeddings
                out_of_distribution = is_out_of_distribution(image_tensor)
                if not out_of_distribution:
                    probabilities = model.predict(image_tensor, verbose=0)[0]

            if out_of_distribution:
                render_invalid_image_error()
                st.session_state.result = None
            else:
                predicted_idx = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence = float(probabilities[predicted_idx]) * 100
                final_risk = adjust_risk(CLASS_INFO[predicted_class]["risk"], changed, family_history)
                st.session_state.result = {
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "final_risk": final_risk,
                }
                # flip the right-column card to the back, animated
                st.session_state.card_flipped = True
                st.session_state.card_just_transitioned = True

    # results - rendered from session_state (not the momentary analyze_clicked)
    # so it, and the "scan another image" button inside it, survive reruns
    if st.session_state.get("result") is not None:
        result = st.session_state.result
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]
        final_risk = result["final_risk"]
        info = CLASS_INFO[predicted_class]

        with st.container(border=True, key="results_card"):
            st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"### {info['name']}")
                st.markdown(
                    f'<p style="color:{TEXT_PRIMARY};">Confidence: '
                    f'<span style="color:{TEXT_PRIMARY}; font-weight:700;">{confidence:.1f}%</span></p>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(render_confidence_gauge(confidence), width="stretch", theme=None)
            with col2:
                render_risk_badge(final_risk)
                if final_risk != info["risk"]:
                    st.markdown(
                        f'<p style="color:{TEXT_PRIMARY}; font-size:0.8rem; margin-top:0.5rem;">'
                        f"Elevated from {info['risk']} based on your answers</p>",
                        unsafe_allow_html=True,
                    )

            st.markdown("#### Probability by condition")
            # theme=None so streamlit doesn't re-tint the axis labels over
            # top of the custom colors set in render_probability_chart
            st.plotly_chart(render_probability_chart(probabilities, predicted_class), width="stretch", theme=None)

        # bumping the uploader's key forces a brand new (empty) file_uploader
        # widget on rerun - that's the standard way to "reset" one in streamlit
        if st.button("Scan another image", icon=":material/refresh:", width="stretch"):
            st.session_state.result = None
            st.session_state.uploader_key += 1
            # flip the card back to the front, animated
            st.session_state.card_flipped = False
            st.session_state.card_just_transitioned = True
            st.rerun()

# right column - rendered after main_col so it reflects this rerun's result,
# not the previous one
with info_col:
    render_flip_card()

# disclaimer
st.markdown(
    '<div class="disclaimer">This tool is for educational purposes only and is '
    "not a substitute for professional medical advice.</div>",
    unsafe_allow_html=True,
)
