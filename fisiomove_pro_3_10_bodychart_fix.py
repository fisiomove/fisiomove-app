# Consolidated and corrected Streamlit app
# - Fixed syntax errors
# - Safe toggle info buttons
# - Thomas test: supports cm -> degrees conversion
# - Single-page clinical PDF with improved header/footer/legend
# - Radar with colored nodes and status icons in table
# Note: keep this file as streamlit_app.py and run with `streamlit run streamlit_app.py`

import io
import os
import random
import re
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader

# Optional QR code support (best-effort)
try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

st.set_page_config(page_title="Fisiomove MobilityPro", layout="centered")

# -----------------------------
# Utilities
# -----------------------------
emoji_pattern = re.compile(
    "["
    u"\U0001F300-\U0001F5FF"
    u"\U0001F600-\U0001F64F"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F700-\U0001F77F"
    u"\U0001F780-\U0001F7FF"
    u"\U0001F800-\U0001F8FF"
    u"\U0001F900-\U0001F9FF"
    u"\U0001FA00-\U0001FA6F"
    u"\u2600-\u26FF"
    "]+",
    flags=re.UNICODE,
)


def sanitize_text_for_plot(s):
    if not isinstance(s, str):
        return s
    return emoji_pattern.sub("", s)


def short_key(s: str) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"t_{h}"


# -----------------------------
# Constants & assets
# -----------------------------
APP_TITLE = "Fisiomove MobilityPro"
SUBTITLE = "Valutazione Funzionale — versione 1.0"
PRIMARY = "#1E6CF4"
CONTACT = "info@fisiomove.example"

LOGO_PATHS = ["logo 2600x1000.jpg", "logo.png", "logo.jpg"]


def load_logo_bytes():
    for p in LOGO_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    img = Image.new("RGB", (1000, 260), (30, 108, 244))
    d = ImageDraw.Draw(img)
    d.text((30, 100), "Fisiomove", fill=(255, 255, 255))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


LOGO = load_logo_bytes()

# -----------------------------
# Tests definitions
# Each tuple: (name, unit, ref, bilat, region, desc, higher_is_better)
# higher_is_better=False means lower numeric value = better (inverted scoring)
# -----------------------------
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 12.0, True, "ankle", "Test dorsiflessione in carico.", True),
        ("Passive Hip Flexion", "°", 120.0, True, "hip", "Flessione anca passiva.", True),
        ("Hip Rotation (flexed 90°)", "°", 40.0, True, "hip", "Rotazione anca (flessione 90°).", True),
        ("Wall Angel Test", "cm", 12.0, False, "thoracic", "Distanza cm tra braccio e muro; valori alti indicano rigidità.", False),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "Rotazione esterna spalla (low-bar).", True),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Flessione spalla (supina).", True),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "ER a 90° abduzione.", True),
        ("Wall Angel Test", "cm", 12.0, False, "thoracic", "Distanza cm tra braccio e muro; valori alti indicano rigidità.", False),
        ("Pectoralis Minor Length", "cm", 5.0, True, "shoulder", "Distanza PM: valori più bassi indicano maggiore mobilità.", False),
        ("Thomas Test (modified)", "°", 10.0, False, "hip", "Thomas test (modificato).", True),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90.0, True, "knee", "Estensione attiva ginocchio (AKE).", True),
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "SLR catena posteriore.", True),
        ("Weight Bearing Lunge Test", "cm", 12.0, True, "ankle", "Test dorsiflessione in carico.", True),
        ("Sorensen Endurance", "sec", 180.0, False, "lumbar", "Test endurance estensori lombari.", True),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "SLR neurodinamica.", True),
        ("ULNT1A (Median nerve)", "°", 90.0, True, "shoulder", "ULNT1A (nervo mediano).", True),
    ],
}

# Short labels for radar
SHORT_RADAR_LABELS = {
    "Weight Bearing Lunge Test": "Mobilità caviglia",
    "Passive Hip Flexion": "Flessione anca",
    "Hip Rotation (flexed 90°)": "Rotazione anca",
    "Wall Angel Test": "Wall Angel",
    "Shoulder ER (adducted, low-bar)": "ER spalla",
    "Shoulder Flexion (supine)": "Flessione spalla",
    "External Rotation (90° abd)": "ER 90° abd",
    "Pectoralis Minor Length": "PM length",
    "Thomas Test (modified)": "Thomas (flessori anca)",
    "Active Knee Extension (AKE)": "AKE hamstring",
    "Straight Leg Raise (SLR)": "SLR",
    "Sorensen Endurance": "Endurance lombare",
    "ULNT1A (Median nerve)": "ULNT1A (mediano)",
}

# Labels to use ONLY inside PDFs (anatomical area)
PDF_TEST_LABELS = {
    "Weight Bearing Lunge Test": "Test caviglia",
    "Passive Hip Flexion": "Test mob. flessione anca",
    "Hip Rotation (flexed 90°)": "Test rotazione anca",
    "Wall Angel Test": "Test mobilità toracica",
    "Shoulder ER (adducted, low-bar)": "Test rotazione spalla",
    "Shoulder Flexion (supine)": "Test flessione spalla",
    "External Rotation (90° abd)": "Test rot spalla",
    "Pectoralis Minor Length": "Test pettorale minore",
    "Thomas Test (modified)": "Test flessori anca",
    "Active Knee Extension (AKE)": "Test estensione ginocchio",
    "Straight Leg Raise (SLR)": "Test sciatico",
    "Sorensen Endurance": "Test endurance lombare",
    "ULNT1A (Median nerve)": "Test neurodinamico spalla",
}


def pdf_test_label(name: str) -> str:
    return PDF_TEST_LABELS.get(name, name)


# -----------------------------
# Init session state
# -----------------------------
def init_state():
    if "vals" not in st.session_state:
        st.session_state["vals"] = {}
    if "athlete" not in st.session_state:
        st.session_state["athlete"] = "Mario Rossi"
    if "evaluator" not in st.session_state:
        st.session_state["evaluator"] = "Dott. Alessandro Ferreri"
    if "date" not in st.session_state:
        st.session_state["date"] = datetime.now().strftime("%Y-%m-%d")
    if "section" not in st.session_state:
        st.session_state["section"] = "Valutazione Generale"


init_state()


def seed_defaults():
    if st.session_state["vals"]:
        return
    for sec, items in TESTS.items():
        for (name, unit, ref, bilat, region, desc, hib) in items:
            if bilat:
                st.session_state["vals"][name] = {
                    "Dx": ref * 0.9 if unit != "sec" else ref * 0.8,
                    "Sx": ref * 0.88 if unit != "sec" else ref * 0.78,
                    "DoloreDx": False,
                    "DoloreSx": False,
                    "unit": unit,
                    "ref": ref,
                    "bilat": True,
                    "region": region,
                    "desc": desc,
                    "section": sec,
                    "higher_is_better": hib,
                }
            else:
                # Thomas test handled as non-bilateral here (single numeric input)
                st.session_state["vals"][name] = {
                    "Val": ref * 0.85,
                    "Dolore": False,
                    "unit": unit,
                    "ref": ref,
                    "bilat": False,
                    "region": region,
                    "desc": desc,
                    "section": sec,
                    "higher_is_better": hib,
                    "input_method": "degrees" if name == "Thomas Test (modified)" else "degrees",
                }


seed_defaults()

# -----------------------------
# EBM interpretation texts (no practical recommendations)
# -----------------------------
EBM_LIBRARY = {
    "Weight Bearing Lunge Test": {
        "title": "Dorsiflessione caviglia (WBLT)",
        "text": "Test: WBLT — dorsiflessione in carico. Interpretazione: valuta mobilità tibio‑talarica e simmetria.",
    },
    "Passive Hip Flexion": {
        "title": "Flessione anca passiva",
        "text": "Test: flessione passiva. Interpretazione: misura il ROM passivo dell'anca.",
    },
    "Hip Rotation (flexed 90°)": {
        "title": "Rotazione anca (flessione 90°)",
        "text": "Test: rotazione in flessione 90°. Interpretazione: ROM rotazionale funzionale.",
    },
    "Wall Angel Test": {
        "title": "Wall Angel",
        "text": "Test: distanza cm tra braccio e muro. Interpretazione: valori maggiori indicano maggiore rigidità (scala invertita).",
    },
    "Pectoralis Minor Length": {
        "title": "Lunghezza piccolo pettorale",
        "text": "Test: lunghezza PM. Interpretazione: valori più bassi indicano maggiore mobilità (scala invertita).",
    },
    "Thomas Test (modified)": {
        "title": "Thomas test (modificato)",
        "text": "Test: accorciamento flessori d'anca. Interpretazione: deficit in gradi rispetto a 0°.",
    },
    "Active Knee Extension (AKE)": {
        "title": "AKE",
        "text": "Test: estensione attiva ginocchio (90/90). Interpretazione: lunghezza hamstrings.",
    },
    "Straight Leg Raise (SLR)": {
        "title": "SLR",
        "text": "Test: SLR. Interpretazione: differenziare componente muscolare da neurale.",
    },
    "Sorensen Endurance": {
        "title": "Sorensen",
        "text": "Test: endurance lombare (secondi). Interpretazione: tempi ridotti indicano deficit di endurance.",
    },
    "ULNT1A (Median nerve)": {
        "title": "ULNT1A",
        "text": "Test: ULNT1A. Interpretazione: mobilità neurale e riproduzione dei sintomi.",
    },
    "Shoulder ER (adducted, low-bar)": {
        "title": "Rotazione esterna spalla",
        "text": "Test: ER in adduzione. Interpretazione: capacità di ER per posizionamento low‑bar.",
    },
    "Shoulder Flexion (supine)": {
        "title": "Flessione spalla",
        "text": "Test: flessione spalla supina. Interpretazione: differenza attivo/passivo.",
    },
    "External Rotation (90° abd)": {
        "title": "ER 90° abd",
        "text": "Test: ER a 90° abduzione. Interpretazione: mobilità e stabilità per overhead.",
    },
}

TEST_INSTRUCTIONS = {
    k: v["text"] for k, v in EBM_LIBRARY.items()
}

# -----------------------------
# Scoring helpers
# -----------------------------
def ability_linear(val, ref, higher_is_better=True):
    try:
        if ref <= 0:
            return 0.0
        val = float(val)
        if higher_is_better:
            score = (val / float(ref)) * 10.0
        else:
            v = min(val, ref)
            score = (1.0 - (v / float(ref))) * 10.0
        return max(0.0, min(10.0, score))
    except Exception:
        return 0.0


def symmetry_score(dx, sx, unit):
    try:
        diff = abs(float(dx) - float(sx))
        if "°" in unit:
            scale = 20.0
        elif unit == "cm":
            scale = 8.0
        else:
            scale = 10.0
        return 10.0 * max(0.0, 1.0 - min(diff, scale) / scale)
    except Exception:
        return 0.0


# -----------------------------
# Toggle callback for info buttons
# -----------------------------
def toggle_info(session_key: str):
    st.session_state[session_key] = not st.session_state.get(session_key, False)


# -----------------------------
# Rendering inputs
# -----------------------------
def get_all_unique_tests():
    unique = {}
    for s, its in TESTS.items():
        for item in its:
            name = item[0]
            if name not in unique:
                unique[name] = (s, *item)
    return list(unique.values())


def render_inputs_for_section(section):
    tests = get_all_unique_tests() if section == "Valutazione Generale" else [(section, *t) for t in TESTS.get(section, [])]
    region_map = {}
    for sec, name, unit, ref, bilat, region, desc, hib in tests:
        region_map.setdefault(region or "other", []).append((sec, name, unit, ref, bilat, region, desc, hib))

    for region, items in region_map.items():
        with st.expander(region.capitalize(), expanded=False):
            for sec, name, unit, ref, bilat, region, desc, hib in items:
                rec = st.session_state["vals"].get(name)
                if not rec:
                    continue
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                cols = st.columns([7, 1])
                with cols[0]:
                    st.markdown(f"**{name}**  \n*{desc}*  \n*Rif:* {ref} {unit}")
                with cols[1]:
                    session_key = f"info_{short_key(name)}"
                    button_key = f"btn_{short_key(name)}"
                    st.button("ℹ️", key=button_key, on_click=toggle_info, args=(session_key,))
                if st.session_state.get(f"info_{short_key(name)}", False):
                    instr = TEST_INSTRUCTIONS.get(name, "Istruzioni non disponibili.")
                    st.info(instr)

                key = short_key(name)
                # Thomas test supports cm or degrees
                if name == "Thomas Test (modified)":
                    method_key = f"{key}_method"
                    current_method = rec.get("input_method", "degrees")
                    method = st.selectbox("Metodo input", options=["degrees", "cm"], index=0 if current_method == "degrees" else 1, key=method_key)
                    rec["input_method"] = method
                    if method == "cm":
                        max_cm = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 20.0
                        val_cm = st.slider("Distanza coscia‑tavolo (cm)", 0.0, max_cm, float(rec.get("Val_cm", 0.0)), 0.1, key=f"{key}_Val_cm")
                        rec["Val_cm"] = val_cm
                        ref_cm = 10.0
                        deg = float(val_cm) * (rec.get("ref", ref) / ref_cm)
                        rec["Val"] = deg
                    else:
                        max_deg = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 30.0
                        val_deg = st.slider("Angolo estensione (°)", 0.0, max_deg, float(rec.get("Val", 0.0)), 0.5, key=f"{key}_Val_deg")
                        rec["Val"] = val_deg
                    # show computed score
                    sc = ability_linear(rec["Val"], rec.get("ref", ref), rec.get("higher_is_better", hib))
                    st.caption(f"Score (calcolato su gradi): **{sc:.1f}/10**")
                else:
                    max_val = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 10.0
                    if rec.get("bilat", False):
                        c1, c2 = st.columns([1, 1])
                        with c1:
                            dx = st.slider(f"Dx ({unit})", 0.0, max_val, float(rec.get("Dx", 0.0)), 0.1, key=f"{key}_Dx")
                            pdx = st.checkbox("Dolore Dx", value=bool(rec.get("DoloreDx", False)), key=f"{key}_pDx")
                        with c2:
                            sx = st.slider(f"Sx ({unit})", 0.0, max_val, float(rec.get("Sx", 0.0)), 0.1, key=f"{key}_Sx")
                            psx = st.checkbox("Dolore Sx", value=bool(rec.get("DoloreSx", False)), key=f"{key}_pSx")
                        rec.update({"Dx": dx, "Sx": sx, "DoloreDx": pdx, "DoloreSx": psx})
                        sc = ability_linear((dx + sx) / 2.0, rec.get("ref", ref), rec.get("higher_is_better", hib))
                        sym = symmetry_score(dx, sx, unit)
                        st.caption(f"Score: **{sc:.1f}/10** — Δ {abs(dx - sx):.1f} {unit} — Sym: **{sym:.1f}/10")
                    else:
                        val = st.slider(f"Valore ({unit})", 0.0, max_val, float(rec.get("Val", 0.0)), 0.1, key=f"{key}_Val")
                        p = st.checkbox("Dolore", value=bool(rec.get("Dolore", False)), key=f"{key}_p")
                        rec.update({"Val": val, "Dolore": p})
                        sc = ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib))
                        st.caption(f"Score: **{sc:.1f}/10**")
                st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# DF builder
# -----------------------------
def build_df(section):
    rows = []
    seen_tests = set()
    for sec, items in TESTS.items():
        if section != "Valutazione Generale" and sec != section:
            continue
        for (name, unit, ref, bilat, region, desc, hib) in items:
            if section == "Valutazione Generale":
                if name in seen_tests:
                    continue
                seen_tests.add(name)

            rec = st.session_state["vals"].get(name)
            if not rec:
                continue

            # Thomas conversion handled via rec["Val"] which was set on input
            if rec.get("bilat", False):
                dx = pd.to_numeric(rec.get("Dx", 0.0), errors="coerce")
                sx = pd.to_numeric(rec.get("Sx", 0.0), errors="coerce")
                dx = 0.0 if pd.isna(dx) else float(dx)
                sx = 0.0 if pd.isna(sx) else float(sx)
                avg = (dx + sx) / 2.0
                sc = round(ability_linear(avg, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                delta = round(abs(dx - sx), 2)
                sym = round(symmetry_score(dx, sx, unit), 2)
                dolore_dx = bool(rec.get("DoloreDx", False))
                dolore_sx = bool(rec.get("DoloreSx", False))
                dolore_any = dolore_dx or dolore_sx
                rows.append(
                    [
                        sec,
                        name,
                        unit,
                        rec.get("ref", ref),
                        f"{avg:.1f}",
                        sc,
                        round(dx, 2),
                        round(sx, 2),
                        delta,
                        sym,
                        dolore_any,
                        region,
                        dolore_dx,
                        dolore_sx,
                    ]
                )
            else:
                # non-bilateral
                if name == "Thomas Test (modified)":
                    display_val = rec.get("Val_cm") if rec.get("input_method") == "cm" else rec.get("Val", 0.0)
                    val_for_score = rec.get("Val", 0.0)
                    sc = round(ability_linear(val_for_score, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                    dolore = bool(rec.get("Dolore", False))
                    rows.append([sec, name, unit, rec.get("ref", ref), f"{display_val:.1f}", sc, "", "", "", "", dolore, region, False, False])
                else:
                    val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce")
                    val = 0.0 if pd.isna(val) else float(val)
                    sc = round(ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                    dolore = bool(rec.get("Dolore", False))
                    rows.append([sec, name, unit, rec.get("ref", ref), f"{val:.1f}", sc, "", "", "", "", dolore, region, False, False])

    df = pd.DataFrame(
        rows,
        columns=[
            "Sezione",
            "Test",
            "Unità",
            "Rif",
            "Valore",
            "Score",
            "Dx",
            "Sx",
            "Delta",
            "SymScore",
            "Dolore",
            "Regione",
            "DoloreDx",
            "DoloreSx",
        ],
    )
    for col in ["Score", "Dx", "Sx", "Delta", "SymScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# -----------------------------
# Plot functions
# -----------------------------
def radar_plot_matplotlib(df, title="Punteggi (0–10)"):
    import numpy as np

    labels_raw = df["Test"].tolist()
    labels = [SHORT_RADAR_LABELS.get(name, name) for name in labels_raw]
    values = df["Score"].astype(float).tolist()

    if len(labels) < 3:
        raise ValueError("Servono almeno 3 test per il radar.")

    values += values[:1]
    labels += labels[:1]
    num_vars = len(labels) - 1
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, linewidth=2, linestyle="solid", color=PRIMARY)
    ax.fill(angles, values, alpha=0.25, color=PRIMARY)

    node_colors = []
    for v in values[:-1]:
        if v >= 7:
            node_colors.append("#16A34A")
        elif v >= 4:
            node_colors.append("#F59E0B")
        else:
            node_colors.append("#DC2626")
    node_angles = angles[:-1]
    ax.scatter(node_angles, values[:-1], c=node_colors, s=70, zorder=5, edgecolors="k")

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=9)
    ax.set_title(sanitize_text_for_plot(title), y=1.08, fontsize=14)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


# Mapping per nomi più semplici da usare nel grafico delle asimmetrie nel PDF
SIMPLE_TEST_LABELS = {
    "Weight Bearing Lunge Test": "Caviglia",
    "Passive Hip Flexion": "Anca passiva",
    "Hip Rotation (flexed 90°)": "Rotazione anca",
    "Wall Angel Test": "Rigidità braccio-muro",
    "Shoulder ER (adducted, low-bar)": "Spalla ER Low-Bar",
    "Shoulder Flexion (supine)": "Flessione spalla",
    "External Rotation (90° abd)": "ER Spalla 90° abd",
    "Pectoralis Minor Length": "Lunghezza Pettorale Min.",
    "Thomas Test (modified)": "Thomas Test",
    "Active Knee Extension (AKE)": "Estensione attiva ginocchio",
    "Straight Leg Raise (SLR)": "SLR",
    "Sorensen Endurance": "Endurance lombare",
    "ULNT1A (Median nerve)": "ULNT1A",
}

def asymmetry_plot_matplotlib(df, title="SymScore – Simmetria Dx/Sx"):
    df_bilat = df[df["SymScore"].notnull()].copy()
    try:
        df_bilat["SymScore"] = pd.to_numeric(df_bilat["SymScore"], errors="coerce")
        df_bilat = df_bilat.dropna(subset=["SymScore"])
    except Exception:
        return None

    if df_bilat.empty:
        return None

    # Passaggio dai nomi dei test a nomi più semplici
    labels = df_bilat["Test"].apply(lambda name: SIMPLE_TEST_LABELS.get(name, name)).tolist()
    scores = df_bilat["SymScore"].tolist()

    colors_map = []
    for score in scores:
        if score >= 7:
            colors_map.append("#16A34A")
        elif score >= 4:
            colors_map.append("#F59E0B")
        else:
            colors_map.append("#DC2626")

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(labels, scores, color=colors_map)
    ax.set_xlabel("SymScore (0–10)")
    ax.set_title(sanitize_text_for_plot(title))
    ax.set_xlim(0, 10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va="center")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


@st.cache_data
def plotly_radar(df):
    df_r = df[df["Score"].notnull()].copy()
    if len(df_r) < 3:
        return None
    fig = px.line_polar(df_r, r="Score", theta="Test", line_close=True, template="plotly_white",
                        color_discrete_sequence=[PRIMARY])
    fig.update_traces(fill="toself", marker=dict(size=6))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), polar=dict(radialaxis=dict(range=[0, 10])))
    return fig


@st.cache_data
def plotly_asymmetry(df):
    df_bilat = df[df["SymScore"].notnull()].copy()
    if df_bilat.empty:
        return None
    df_bilat["SymScore"] = pd.to_numeric(df_bilat["SymScore"], errors="coerce")
    fig = px.bar(df_bilat, x="SymScore", y="Test", orientation="h", template="plotly_white",
                 color="SymScore", color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"], range_x=[0, 10])
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


# -----------------------------
# UI main
# -----------------------------
st.markdown(
    f"""
<style>
:root {{ --primary: {PRIMARY}; }}
body {{ background: #f6f8fb; }}
.header-card {{ background: linear-gradient(90deg, #ffffff, #f1f7ff); padding:12px; border-radius:12px; }}
.card {{ background: white; padding:12px; border-radius:10px; margin-bottom:10px; }}
.small-muted {{ color:#6b7280; font-size:0.9rem; }}
</style>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(LOGO, width=100)
with col2:
    st.markdown(f"<div class='header-card'><h2 style='color:{PRIMARY};margin:0'>{APP_TITLE}</h2><div class='small-muted'>{SUBTITLE}</div></div>", unsafe_allow_html=True)
with col3:
    metric_container = st.container()

with st.sidebar:
    st.markdown("### Dati atleta")
    st.session_state["athlete"] = st.text_input("Atleta", st.session_state["athlete"])
    st.session_state["evaluator"] = st.text_input("Valutatore", st.session_state["evaluator"])
    st.session_state["date"] = st.date_input("Data", datetime.strptime(st.session_state["date"], "%Y-%m-%d")).strftime("%Y-%m-%d")
    st.markdown("---")
    st.session_state["section"] = st.selectbox("Sezione", ["Valutazione Generale"], index=0)
    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("Reset valori", use_container_width=True):
            st.session_state["vals"].clear()
            seed_defaults()
            st.experimental_rerun()
    with colb2:
        if st.button("Randomizza", use_container_width=True):
            for name, rec in st.session_state["vals"].items():
                ref = rec.get("ref", 10.0)
                if rec.get("bilat", False):
                    rec["Dx"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["Sx"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["DoloreDx"] = random.random() < 0.15
                    rec["DoloreSx"] = random.random() < 0.15
                else:
                    if name == "Thomas Test (modified)":
                        rec["input_method"] = random.choice(["degrees", "cm"])
                        if rec["input_method"] == "cm":
                            rec["Val_cm"] = max(0.0, rec.get("ref", 10.0) * random.uniform(0.2, 1.2))
                            ref_cm = 10.0
                            rec["Val"] = rec["Val_cm"] * (rec.get("ref", 10.0) / ref_cm)
                        else:
                            rec["Val"] = max(0.0, rec.get("ref", 10.0) * random.uniform(0.5, 1.2))
                        rec["Dolore"] = random.random() < 0.15
                    else:
                        rec["Val"] = max(0.0, ref * random.uniform(0.5, 1.2))
                        rec["Dolore"] = random.random() < 0.15
            st.success("Valori random impostati.")

# Render inputs
render_inputs_for_section(st.session_state["section"])

# Build df and show results
df_show = build_df(st.session_state["section"])
st.markdown("### Risultati")
# Add status icon
def status_icon(score):
    try:
        s = float(score)
        if s >= 7:
            return "✔"
        elif s >= 4:
            return "⚠"
        else:
            return "✖"
    except Exception:
        return ""

df_display = df_show.copy()
df_display["Stato"] = df_display["Score"].apply(status_icon)
cols_order = ["Stato", "Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]
st.write(df_display[cols_order].style.format(precision=1))

with metric_container:
    avg_score = df_show["Score"].mean() if not df_show["Score"].isna().all() else 0.0
    painful = int(df_show["Dolore"].sum()) if "Dolore" in df_show.columns else 0
    sym_mean = df_show["SymScore"].mean() if "SymScore" in df_show.columns else np.nan
    st.metric("Score medio", f"{avg_score:.1f}/10")
    st.metric("Test con dolore", f"{painful}")
    st.metric("Symmetry medio", f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a")

col_a, col_b = st.columns(2)
with col_a:
    radar_fig = plotly_radar(df_show)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Radar non disponibile: servono almeno 3 test con punteggio.")
with col_b:
    asym_fig = plotly_asymmetry(df_show)
    if asym_fig:
        st.plotly_chart(asym_fig, use_container_width=True)
    else:
        st.info("Grafico asimmetria non disponibile.")

# Prepare chart buffers
radar_buf = None
asym_buf = None
try:
    df_radar = df_show[df_show["Score"].notnull()].copy()
    if len(df_radar) >= 3:
        radar_buf = radar_plot_matplotlib(df_radar, title=f"{st.session_state['section']} - Punteggi (0-10)")
except Exception:
    radar_buf = None

try:
    asym_buf = asymmetry_plot_matplotlib(df_show, title=f"Asimmetrie - {st.session_state['section']}")
except Exception:
    asym_buf = None

# EBM notes
def ebm_from_df(df):
    notes = []
    for _, r in df.iterrows():
        test = str(r["Test"])
        score = float(r["Score"]) if not pd.isna(r["Score"]) else 10.0
        entry = EBM_LIBRARY.get(test)
        if not entry:
            continue
        title_it = pdf_test_label(test)
        if score < 7:
            paragraph = f"{title_it}: {entry['text']}"
        else:
            paragraph = f"{title_it}: Risultato nella norma (score {score:.1f}/10)."
        notes.append(paragraph)
    return notes

ebm_notes = ebm_from_df(df_show)

# -----------------------------
# PDF footer (page number & contact) and optional QR
# -----------------------------
def add_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    footer_text = f"{CONTACT} • Valutatore: {st.session_state.get('evaluator', '')}"
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(doc.leftMargin, 1.0 * cm, footer_text)
    page_num_text = f"Pagina {canvas.getPageNumber()}"
    canvas.drawRightString(w - doc.rightMargin, 1.0 * cm, page_num_text)
    if QR_AVAILABLE:
        try:
            qr_data = f"Atleta:{st.session_state.get('athlete','')}|Data:{st.session_state.get('date','')}"
            qr = qrcode.make(qr_data)
            bio = io.BytesIO()
            qr.save(bio, format="PNG")
            bio.seek(0)
            img_reader = ImageReader(bio)
            canvas.drawImage(img_reader, doc.leftMargin, 1.4 * cm, width=2 * cm, height=2 * cm)
        except Exception:
            pass
    canvas.restoreState()

# -----------------------------
# PDF generation (clinico single page improved)
# -----------------------------
def pdf_report_clinico(logo_bytes, athlete, evaluator, date_str, section, df, ebm_notes, radar_buf=None, asym_buf=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm, bottomMargin=2.4 * cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title_style = styles["Title"]
    heading = ParagraphStyle("heading", parent=styles["Heading2"], alignment=TA_LEFT, textColor=colors.HexColor(PRIMARY))
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)

    story = []
    header_table = Table(
        [
            [
                RLImage(io.BytesIO(logo_bytes), width=4.0 * cm, height=1.0 * cm),
                Paragraph(f"<b>Report Valutazione</b><br/>{sanitize_text_for_plot(section)}", title_style),
                Paragraph(f"<b>Atleta:</b> {athlete}<br/><b>Valutatore:</b> {evaluator}<br/><b>Data:</b> {date_str}", small),
            ]
        ],
        colWidths=[4.2 * cm, 8.8 * cm, 4.0 * cm],
    )
    header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(header_table)
    story.append(Spacer(1, 8))

    # Metrics
    avg_score = df["Score"].mean() if "Score" in df.columns and not df["Score"].isna().all() else 0.0
    n_dolore = int(df["Dolore"].sum()) if "Dolore" in df.columns else 0
    sym_mean = df["SymScore"].mean() if "SymScore" in df.columns else np.nan
    metrics_table = Table(
        [[
            Paragraph("<b>Score medio</b>", small), Paragraph(f"{avg_score:.1f}/10", small),
            Paragraph("<b>Test con dolore</b>", small), Paragraph(str(n_dolore), small),
            Paragraph("<b>Symmetry medio</b>", small), Paragraph(f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a", small)
        ]],
        colWidths=[2.4*cm, 2.0*cm, 2.6*cm, 1.6*cm, 3.0*cm, 2.2*cm]
    )
    metrics_table.setStyle(TableStyle([("BOX", (0,0), (-1,-1), 0.25, colors.lightgrey), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
    story.append(metrics_table)
    story.append(Spacer(1, 10))

    # Top 3
    story.append(Paragraph("<b>Top 3 priorità</b>", heading))
    bad_df = df.copy()
    bad_df = bad_df[bad_df["Score"].notnull()].sort_values("Score", ascending=True)
    priorities = bad_df[bad_df["Score"] < 7].head(3)
    if not priorities.empty:
        for _, row in priorities.iterrows():
            lbl = pdf_test_label(row["Test"])
            story.append(Paragraph(f"• {lbl} — Score {row['Score']:.1f}/10 — area da approfondire", normal))
    else:
        story.append(Paragraph("Nessuna priorità critica.", normal))
    story.append(Spacer(1, 10))

    # Results table
    disp = df.copy()
    disp["Status"] = disp["Score"].apply(lambda s: "✔" if s >= 7 else ("⚠" if s >= 4 else "✖"))
    disp["TestPdf"] = disp["Test"].apply(pdf_test_label)
    table_cols = ["Status", "Test", "Valore", "Unità", "Rif", "Score"]
    table_data = [table_cols]
    for _, r in disp.iterrows():
        table_data.append([r["Status"], r["TestPdf"], r["Valore"], r["Unità"], r["Rif"], f"{r['Score']:.1f}"])
    colWidths = [1.0*cm, 7.0*cm, 2.0*cm, 2.0*cm, 1.6*cm, 2.0*cm]
    result_table = Table(table_data, colWidths=colWidths, repeatRows=1)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])
    for i in range(1, len(table_data)):
        bg = colors.whitesmoke if i % 2 == 0 else colors.white
        style.add("BACKGROUND", (0, i), (-1, i), bg)
        try:
            score = float(table_data[i][5])
            if score >= 7:
                color = colors.HexColor("#ecfdf5")
            elif score >= 4:
                color = colors.HexColor("#fffaf0")
            else:
                color = colors.HexColor("#fff1f2")
            style.add("BACKGROUND", (5, i), (5, i), color)
        except Exception:
            pass
    result_table.setStyle(style)
    story.append(result_table)
    story.append(Spacer(1, 12))

    # Charts (keep together & centered)
    chart_block = []
    if radar_buf:
        chart_block.append(Paragraph("<b>Grafico radar</b>", normal))
        chart_block.append(Spacer(1, 6))
        chart_block.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10 * cm, height=10 * cm, hAlign="CENTER"))
        chart_block.append(Spacer(1, 8))
    if asym_buf:
        chart_block.append(Paragraph("<b>Asimmetrie</b>", normal))
        chart_block.append(Spacer(1, 6))
        chart_block.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14 * cm, height=6 * cm, hAlign="CENTER"))
        chart_block.append(Spacer(1, 8))

    if chart_block:
        story.append(KeepTogether(chart_block))

    # Pain regions
    pain_regions = []
    for _, row in df.iterrows():
        regione = str(row.get("Regione", "") or "").strip()
        if not regione:
            continue
        try:
            if bool(row.get("DoloreDx", False)):
                pain_regions.append(f"{regione} destra")
            if bool(row.get("DoloreSx", False)):
                pain_regions.append(f"{regione} sinistra")
            if bool(row.get("Dolore", False)) and not (row.get("DoloreDx") or row.get("DoloreSx")):
                pain_regions.append(f"{regione}")
        except Exception:
            if bool(row.get("Dolore", False)):
                pain_regions.append(f"{regione}")
    pain_regions = list(dict.fromkeys(pain_regions))
    story.append(Paragraph("<b>Regioni dolorose rilevate</b>", heading))
    if pain_regions:
        for pr in pain_regions:
            story.append(Paragraph(f"• {pr.capitalize()}", normal))
    else:
        story.append(Paragraph("Nessuna regione dolorosa segnalata.", normal))
    story.append(Spacer(1, 12))

    # EBM comments
    story.append(Paragraph("<b>Commento clinico (EBM)</b>", heading))
    story.append(Spacer(1, 6))
    if ebm_notes:
        for p in ebm_notes:
            story.append(Paragraph(sanitize_text_for_plot(p).replace("\n", "<br/>"), normal))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("Nessun commento disponibile.", normal))

    story.append(Spacer(1, 18))
    story.append(Paragraph("Firma: ______________________", small))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buf.seek(0)
    return buf

# Client-friendly PDF uses same single-page template but simplified content
def pdf_report_client_friendly(logo_bytes, athlete, evaluator, date_str, section, df, radar_buf=None, asym_buf=None):
    return pdf_report_clinico(logo_bytes, athlete, evaluator, date_str, section, df, ebm_notes, radar_buf=radar_buf, asym_buf=asym_buf)

# Export buttons
colpdf1, colpdf2 = st.columns(2)
with colpdf1:
    if st.button("Esporta PDF Clinico", use_container_width=True):
        try:
            pdf = pdf_report_clinico(
                logo_bytes=LOGO,
                athlete=st.session_state["athlete"],
                evaluator=st.session_state["evaluator"],
                date_str=st.session_state["date"],
                section=st.session_state["section"],
                df=df_show,
                ebm_notes=ebm_notes,
                radar_buf=radar_buf,
                asym_buf=asym_buf,
            )
            st.download_button("Scarica PDF Clinico", data=pdf.getvalue(), file_name=f"Fisiomove_Report_Clinico_{st.session_state['date']}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Errore durante la generazione del PDF clinico: {e}")

with colpdf2:
    if st.button("Esporta PDF Cliente", use_container_width=True):
        try:
            pdf = pdf_report_client_friendly(
                logo_bytes=LOGO,
                athlete=st.session_state["athlete"],
                evaluator=st.session_state["evaluator"],
                date_str=st.session_state["date"],
                section=st.session_state["section"],
                df=df_show,
                radar_buf=radar_buf,
                asym_buf=asym_buf,
            )
            st.download_button("Scarica PDF Cliente", data=pdf.getvalue(), file_name=f"Fisiomove_Report_Cliente_{st.session_state['date']}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Errore durante la generazione del PDF cliente: {e}")
