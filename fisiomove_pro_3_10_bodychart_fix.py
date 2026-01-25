# streamlit_app.py
# Aggiornato: rimosso seconda pagina consigli interpretativi;
# aggiunta conversione automatica Thomas Test (cm -> °) con scelta input cm/°.

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

st.set_page_config(page_title="Fisiomove MobilityPro v. 1.0", layout="centered")

# -----------------------------
# Utility
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
# Constants & Assets
# -----------------------------
APP_TITLE = "Fisiomove MobilityPro"
SUBTITLE = "Valutazione Funzionale — versione 1.0"
PRIMARY = "#1E6CF4"
ACCENT = "#10A37F"

TEST_NAME_TRANSLATIONS = {
    "Weight Bearing Lunge Test": "Test dorsiflessione caviglia",
    "Passive Hip Flexion": "Flessione anca passiva",
    "Hip Rotation (flexed 90°)": "Rotazione anca (flessione 90°)",
    "Wall Angel Test": "Wall Angel",
    "Shoulder ER (adducted, low-bar)": "Rotazione esterna spalla (low-bar)",
    "Shoulder Flexion (supine)": "Flessione spalla (supina)",
    "External Rotation (90° abd)": "Rotazione esterna a 90° abduzione",
    "Pectoralis Minor Length": "Lunghezza piccolo pettorale",
    "Thomas Test (modified)": "Test di Thomas (modificato)",
    "Active Knee Extension (AKE)": "Estensione attiva ginocchio (AKE)",
    "Straight Leg Raise (SLR)": "Straight Leg Raise (SLR)",
    "Sorensen Endurance": "Test Sorensen (endurance lombare)",
    "ULNT1A (Median nerve)": "ULNT1A (nervo mediano)",
}

SHORT_RADAR_LABELS = {
    "Weight Bearing Lunge Test": "Mobilità caviglia",
    "Passive Hip Flexion": "Flessione anca",
    "Hip Rotation (flexed 90°)": "Rotazione anca",
    "Wall Angel Test": "Wall Angel",
    "Shoulder ER (adducted, low-bar)": "ER spalla (low-bar)",
    "Shoulder Flexion (supine)": "Flessione spalla",
    "External Rotation (90° abd)": "ER 90° abd",
    "Pectoralis Minor Length": "PM length",
    "Thomas Test (modified)": "Lunghezza flessori anca",
    "Active Knee Extension (AKE)": "Hamstring AKE",
    "Straight Leg Raise (SLR)": "SLR",
    "Sorensen Endurance": "Endurance lombare",
    "ULNT1A (Median nerve)": "ULNT1A (mediano)",
}

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
# Scoring helper (supports inverted scales)
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
    except:
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
    except:
        return 0.0


# -----------------------------
# Tests definition
# Each tuple: (name, unit, ref, bilat, region, desc, higher_is_better)
# -----------------------------
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 12.0, True, "ankle", "Test caviglia: dorsiflessione in carico.", True),
        ("Passive Hip Flexion", "°", 120.0, True, "hip", "Test anca: flessione passiva supina.", True),
        ("Hip Rotation (flexed 90°)", "°", 40.0, True, "hip", "Test anca: rotazione a 90° flessione.", True),
        ("Wall Angel Test", "cm", 12.0, False, "thoracic", "Distanza cm tra braccio e muro a braccio teso; valori alti indicano rigidità.", False),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "Test spalla: extrarotazione low-bar.", True),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Test spalla: flessione supina.", True),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "Test spalla: ER a 90° abduzione.", True),
        ("Wall Angel Test", "cm", 12.0, False, "thoracic", "Distanza cm tra braccio e muro a braccio teso; valori alti indicano rigidità.", False),
        ("Pectoralis Minor Length", "cm", 5.0, True, "shoulder", "Distanza PM: valori più bassi indicano maggiore mobilità.", False),
        ("Thomas Test (modified)", "°", 10.0, True, "hip", "Test flessori anca (modificato).", True),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90.0, True, "knee", "Test hamstrings: estensione attiva.", True),
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Test SLR catena posteriore.", True),
        ("Weight Bearing Lunge Test", "cm", 12.0, True, "ankle", "Test caviglia: dorsiflessione in carico.", True),
        ("Sorensen Endurance", "sec", 180.0, False, "lumbar", "Test endurance estensori lombari.", True),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Test neurodinamica posteriore LE.", True),
        ("ULNT1A (Median nerve)", "°", 90.0, True, "shoulder", "Test neurodinamica arto superiore.", True),
    ],
}


# -----------------------------
# Seed defaults (includes input_method default for Thomas)
# -----------------------------
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
                    "input_method": "degrees" if name == "Thomas Test (modified)" else "degrees",
                }
            else:
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
# DataFrame builder
# -----------------------------
def build_df(section):
    rows = []
    seen_tests = set()
    for sec, items in TESTS.items():
        if section != "Valutazione Generale" and sec != section:
            continue
        for (name, unit, ref, bilat, region, desc, hib) in items:
            if section == "Valutazione Generale" and name in seen_tests:
                continue
            seen_tests.add(name)
            rec = st.session_state["vals"].get(name)
            if not rec:
                continue

            # Special handling Thomas Test: if user entered cm, convert to degrees (simple linear map)
            if name == "Thomas Test (modified)":
                method = rec.get("input_method", "degrees")
                if method == "cm":
                    # conversion: deg = cm * (ref_deg / ref_cm)
                    # using ref_cm = 10.0 cm (heuristic); ref_deg = rec['ref']
                    ref_cm = 10.0
                    deg = float(rec.get("Val_cm", 0.0)) * (rec.get("ref", ref) / ref_cm)
                    rec_val_for_score = deg
                else:
                    rec_val_for_score = float(rec.get("Val", 0.0))
            else:
                rec_val_for_score = float(rec.get("Val", 0.0)) if not rec.get("bilat", False) else None

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
                # For non-bilat: use rec_val_for_score (Thomas handled above)
                if name == "Thomas Test (modified)":
                    val_display = rec.get("Val_cm", None) if rec.get("input_method") == "cm" else rec.get("Val", 0.0)
                    val_display = val_display if val_display is not None else rec.get("Val", 0.0)
                    sc = round(ability_linear(rec_val_for_score, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                    dolore = bool(rec.get("Dolore", False))
                    rows.append([sec, name, unit, rec.get("ref", ref), f"{val_display:.1f}", sc, "", "", "", "", dolore, region, False, False])
                else:
                    val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce')
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
# Plot functions (unchanged)
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

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=9)
    ax.set_title(sanitize_text_for_plot(title), y=1.1, fontsize=14)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf


def asymmetry_plot_matplotlib(df, title="SymScore – Simmetria Dx/Sx"):
    df_bilat = df[df["SymScore"].notnull()].copy()
    try:
        df_bilat["SymScore"] = pd.to_numeric(df_bilat["SymScore"], errors="coerce")
        df_bilat = df_bilat.dropna(subset=["SymScore"])
    except Exception:
        return None

    if df_bilat.empty:
        return None

    labels = df_bilat["Test"].tolist()
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
    plt.savefig(buf, format="png")
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
# EBM library (interpretation only; no recommendations)
# -----------------------------
EBM_LIBRARY = {
    "Weight Bearing Lunge Test": {"title": "Dorsiflessione caviglia (WBLT)",
                                  "text": "Test: Weight Bearing Lunge Test — dorsiflessione in carico.\nInterpretazione: valuta mobilità tibio‑talarica in carico e simmetria tra i lati; valori ridotti indicano limitazione funzionale."},
    "Passive Hip Flexion": {"title": "Flessione d'anca passiva",
                             "text": "Test: flessione d'anca passiva.\nInterpretazione: misura il ROM passivo dell'anca; limitazioni suggeriscono restrizioni capsulari o muscolari."},
    "Hip Rotation (flexed 90°)": {"title": "Rotazione anca (flessione 90°)",
                                   "text": "Test: rotazione interna/esterna con anca flessa a 90°.\nInterpretazione: valuta il ROM rotazionale in posizione funzionale; utile per identificare limitazioni capsulari o miofasciali."},
    "Wall Angel Test": {"title": "Wall Angel — controllo scapolare / mobilità toracica",
                       "text": "Test: misura la distanza (cm) tra braccio teso e muro.\nInterpretazione: valori maggiori indicano maggiore rigidità o posture proiettate; la scala numerica è inversa rispetto al punteggio funzionale."},
    "Shoulder ER (adducted, low-bar)": {"title": "Rotazione esterna spalla (low-bar)",
                                         "text": "Test: rotazione esterna in adduzione.\nInterpretazione: valuta la capacità di ER necessaria per il posizionamento low‑bar; deficit possono limitare posizione e causare compensi."},
    "Shoulder Flexion (supine)": {"title": "Flessione spalla (supina)",
                                  "text": "Test: flessione spalla in supino.\nInterpretazione: distingue tra limitazioni attive e passive; utile per valutare capacità overhead."},
    "External Rotation (90° abd)": {"title": "Rotazione esterna spalla a 90° abduzione",
                                    "text": "Test: ER a 90° abduzione.\nInterpretazione: valuta stabilità e mobilità in range overhead; deficit possono essere rilevanti per specifiche attività."},
    "Pectoralis Minor Length": {"title": "Lunghezza piccolo pettorale",
                                "text": "Test: misura la distanza del piccolo pettorale.\nInterpretazione: valori più bassi indicano maggiore mobilità; la scala è inversa rispetto a test dove valore alto indica buon risultato."},
    "Thomas Test (modified)": {"title": "Test di Thomas (modificato)",
                               "text": "Test: valuta l'accorciamento dei flessori d'anca.\nInterpretazione: misurare angolo di estensione d'anca o distanza coscia‑tavolo; il deficit in gradi rispetto a 0° è il riferimento clinico."},
    "Active Knee Extension (AKE)": {"title": "Estensione attiva ginocchio (AKE)",
                                    "text": "Test: misura lunghezza funzionale degli hamstring (posizione 90/90).\nInterpretazione: angolo residuo di flessione indica grado di tensione degli hamstring."},
    "Straight Leg Raise (SLR)": {"title": "Straight Leg Raise (SLR)",
                                 "text": "Test: sollevamento gamba tesa.\nInterpretazione: utile per distinguere limitazioni muscolari da sensibilità neurodinamica mediante manovre di differenziazione."},
    "Sorensen Endurance": {"title": "Test di Sorensen (endurance lombare)",
                           "text": "Test: misura il tempo di mantenimento per gli estensori lombari.\nInterpretazione: tempi ridotti indicano deficit di endurance muscolare del comparto estensore."},
    "ULNT1A (Median nerve)": {"title": "ULNT1A (nervo mediano)",
                              "text": "Test: valutazione della mobilità neurale del nervo mediano.\nInterpretazione: registrare l'angolo e le caratteristiche dei sintomi; comparare con il lato sano e usare differenziazione per conferma neurale."},
}


# -----------------------------
# TEST INSTRUCTIONS and toggle callback (safe)
# -----------------------------
TEST_INSTRUCTIONS = {
    "Weight Bearing Lunge Test": "Posizione: paziente in piedi di fronte al muro.\nEsecuzione: avanzare il ginocchio verso il muro mantenendo il tallone a terra.\nMisura: distanza in cm o angolo con inclinometro; registrare per lato.",
    "Passive Hip Flexion": "Posizione: supino; ginocchio flesso per isolare l'anca.\nMisura: angolo di flessione d'anca con goniometro; stabilizzare il bacino.",
    "Hip Rotation (flexed 90°)": "Posizione: anca flessa 90°, ginocchio flesso.\nMisura: rotazione interna/esterna in gradi con goniometro; confrontare i lati.",
    "Wall Angel Test": "Posizione: paziente con schiena contro il muro, braccia estese.\nMisura: distanza (cm) tra braccio esteso e muro; valori maggiori indicano rigidità.",
    "Shoulder ER (adducted, low-bar)": "Posizione: braccio addotto, gomito 90°.\nMisura: angolo di ER con goniometro; verificare capacità di posizionamento low‑bar.",
    "Shoulder Flexion (supine)": "Posizione: supino.\nMisura: angolo di flessione (passiva/attiva) con goniometro.",
    "External Rotation (90° abd)": "Posizione: abduzione 90°, gomito 90°.\nMisura: angolo di ER con goniometro.",
    "Pectoralis Minor Length": "Posizione: supino.\nMisura: distanza (cm) di riferimento; valori più bassi indicano maggiore mobilità.",
    "Thomas Test (modified)": "Posizione: paziente al bordo del lettino, ginocchia al petto, poi lascia cadere una gamba.\nMisura: angolo di estensione d'anca o distanza coscia‑tavolo (cm).",
    "Active Knee Extension (AKE)": "Posizione: 90/90.\nMisura: angolo residuo di flessione del ginocchio al termine dell'estensione attiva.",
    "Straight Leg Raise (SLR)": "Posizione: supino.\nMisura: angolo di sollevamento e localizzazione del dolore; usare differenziazione per distinguere origine.",
    "Sorensen Endurance": "Posizione: Sorensen.\nMisura: durata in secondi fino a fatica.",
    "ULNT1A (Median nerve)": "Posizione: sdraiato; sequenza che tensiona nervo mediano.\nMisura: angolo di estensione del gomito e caratteristiche dei sintomi."
}


def toggle_info(session_key: str):
    st.session_state[session_key] = not st.session_state.get(session_key, False)


# -----------------------------
# Render inputs (Thomas supports cm or degrees input)
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
                cols = st.columns([7, 1, 1])
                with cols[0]:
                    st.markdown(f"**{name}**  \n*{desc}*  \n*Rif:* {ref} {unit}")
                with cols[1]:
                    session_key = f"info_{short_key(name)}"
                    button_key = f"btn_{short_key(name)}"
                    st.button("ℹ️", key=button_key, on_click=toggle_info, args=(session_key,))
                if st.session_state.get(f"info_{short_key(name)}", False):
                    instr = TEST_INSTRUCTIONS.get(name, "Istruzioni non disponibili.")
                    st.info(instr)

                # Special handling for Thomas Test input mode
                key = short_key(name)
                if name == "Thomas Test (modified)":
                    method_key = f"{key}_method"
                    current_method = rec.get("input_method", "degrees")
                    method = st.selectbox("Metodo input", options=["degrees", "cm"], index=0 if current_method == "degrees" else 1, key=method_key)
                    rec["input_method"] = method
                    if method == "cm":
                        max_cm = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 20.0
                        val_cm = st.slider("Distanza coscia‑tavolo (cm)", 0.0, max_cm, float(rec.get("Val_cm", rec.get("Val", 0.0))), 0.1, key=f"{key}_Val_cm")
                        rec["Val_cm"] = val_cm
                        # convert cm -> degrees (heuristic: ref_cm = 10 cm -> ref_deg = rec['ref'])
                        ref_cm = 10.0
                        deg = float(val_cm) * (rec.get("ref", ref) / ref_cm)
                        rec["Val"] = deg  # store converted degrees for scoring
                    else:
                        max_deg = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 30.0
                        val_deg = st.slider("Angolo estensione (°)", 0.0, max_deg, float(rec.get("Val", 0.0)), 0.5, key=f"{key}_Val_deg")
                        rec["Val"] = val_deg
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
# Render UI and compute DF
# -----------------------------
st.markdown(
    f"""
<style>
:root {{ --primary: {PRIMARY}; --accent: {ACCENT}; }}
body {{ background: #f6f8fb; }}
.header-card {{ background: linear-gradient(90deg, #ffffff, #f1f7ff); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(16,24,40,0.04); }}
.card {{ background: white; padding:12px; border-radius:10px; box-shadow: 0 3px 10px rgba(16,24,40,0.04); margin-bottom:10px; }}
.small-muted {{ color:#6b7280; font-size:0.9rem; }}
</style>
""",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image(LOGO, width=120)
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
                    # For Thomas, randomize both methods sensibly
                    if name == "Thomas Test (modified)":
                        rec["input_method"] = random.choice(["degrees", "cm"])
                        if rec["input_method"] == "cm":
                            rec["Val_cm"] = max(0.0, rec.get("ref", 10.0) * random.uniform(0.2, 1.2))
                            # convert and store Val as degrees
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

# Build DF and show
df_show = build_df(st.session_state["section"])
st.markdown("### Risultati")
st.write(df_show.style.format(precision=1))

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

# Prepare PDF buffers
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

# EBM comments (interpretation only)
def ebm_from_df(df, friendly=False):
    notes = []
    for _, r in df.iterrows():
        test = str(r["Test"])
        score = float(r["Score"]) if not pd.isna(r["Score"]) else 10.0
        entry = EBM_LIBRARY.get(test)
        if not entry:
            continue
        title_it = TEST_NAME_TRANSLATIONS.get(test, test)
        if score < 7:
            paragraph = f"{title_it}: {entry['text']}"
        else:
            paragraph = f"{title_it}: Risultato nella norma (score {score:.1f}/10)."
        notes.append(paragraph)
    return notes

ebm_notes = ebm_from_df(df_show)

# PDF generation: single page only
def pdf_report_no_bodychart(logo_bytes, athlete, evaluator, date_str, section, df, ebm_notes, radar_buf=None, asym_buf=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], alignment=TA_LEFT, textColor=colors.HexColor(PRIMARY))

    story = []
    # Header
    story.append(RLImage(io.BytesIO(logo_bytes), width=14 * cm, height=3.2 * cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Report Valutazione — {sanitize_text_for_plot(section)}</b>", title))
    story.append(Spacer(1, 6))

    # Info
    info_data = [["Atleta", athlete, "Valutatore", evaluator, "Data", date_str]]
    info_table = Table(info_data, colWidths=[2.2 * cm, 6.0 * cm, 2.8 * cm, 5.0 * cm, 1.6 * cm, 2.0 * cm])
    info_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FF")),
                                    ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.whitesmoke),
                                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    # Metrics
    avg_score = df["Score"].mean() if "Score" in df.columns and not df["Score"].isna().all() else 0.0
    n_dolore = int(df["Dolore"].sum()) if "Dolore" in df.columns else 0
    sym_mean = df["SymScore"].mean() if "SymScore" in df.columns else np.nan
    metrics = Table([["Score medio", f"{avg_score:.1f}/10", "Test con dolore", str(n_dolore), "Symmetry medio", f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a"]],
                    colWidths=[3 * cm, 3 * cm, 3 * cm, 2.6 * cm, 3 * cm, 3 * cm])
    metrics.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                                 ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                 ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold")]))
    story.append(metrics)
    story.append(Spacer(1, 12))

    # Top 3 priorities
    story.append(Paragraph("<b>Top 3 priorità di intervento</b>", heading))
    bad_df = df.copy()
    bad_df = bad_df[bad_df["Score"].notnull()]
    bad_df = bad_df.sort_values("Score", ascending=True)
    priorities = bad_df[bad_df["Score"] < 7].head(3)
    if not priorities.empty:
        for _, row in priorities.iterrows():
            test_label = TEST_NAME_TRANSLATIONS.get(row["Test"], row["Test"])
            story.append(Paragraph(f"• {test_label} — Score: {row['Score']:.1f}/10 — Area da approfondire: interpretazione clinica.", normal))
    else:
        story.append(Paragraph("Nessuna priorità critica (tutti i test sono >= 7).", normal))
    story.append(Spacer(1, 12))

    # Results table
    disp = df[["Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]].copy()
    for col in ["Valore", "Score", "Dx", "Sx", "Delta", "SymScore"]:
        disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)
    table_data = [disp.columns.tolist()] + disp.values.tolist()
    colWidths = [2.2 * cm, 6.0 * cm, 1.2 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm]
    table = Table(table_data, repeatRows=1, colWidths=colWidths)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("FONTSIZE", (0, 0), (-1, -1), 8),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    for i, row in enumerate(table_data[1:], start=1):
        try:
            score = float(row[5]) if row[5] != "" and row[5] is not None else None
            if score is not None:
                if score < 4:
                    bg = colors.HexColor("#fff1f2")
                elif score < 7:
                    bg = colors.HexColor("#fffaf0")
                else:
                    bg = colors.HexColor("#ecfdf5")
                table.setStyle([("BACKGROUND", (5, i), (5, i), bg)])
        except Exception:
            pass

    story.append(Paragraph("<b>Tabella risultati</b>", heading))
    story.append(Spacer(1, 6))
    story.append(table)
    story.append(Spacer(1, 12))

    # Charts
    if radar_buf:
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10 * cm, height=10 * cm))
        story.append(Spacer(1, 8))
    if asym_buf:
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14 * cm, height=6 * cm))
        story.append(Spacer(1, 8))

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
    story.append(Paragraph("<b>Regioni dolorose riscontrate durante il test</b>", heading))
    story.append(Spacer(1, 6))
    if pain_regions:
        for reg in pain_regions:
            story.append(Paragraph(f"• {reg.capitalize()}", normal))
    else:
        story.append(Paragraph("Nessuna regione segnalata come dolorosa.", normal))
    story.append(Spacer(1, 12))

    # EBM Comments (interpretation only)
    story.append(Paragraph("<b>Commento clinico (EBM)</b>", heading))
    story.append(Spacer(1, 6))
    if ebm_notes:
        for para in ebm_notes:
            story.append(Paragraph(sanitize_text_for_plot(para).replace("\n", "<br/>"), normal))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Nessun commento disponibile.", normal))

    # Footer
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Valutatore: {evaluator}", small))
    story.append(Paragraph("Firma: ______________________", small))
    story.append(Spacer(1, 6))

    doc.build(story)
    buf.seek(0)
    return buf

# Export buttons
colpdf1, colpdf2 = st.columns(2)
with colpdf1:
    if st.button("Esporta PDF Clinico", use_container_width=True):
        try:
            pdf = pdf_report_no_bodychart(
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
            st.error(f"Errore durante generazione PDF clinico: {e}")

with colpdf2:
    if st.button("Esporta PDF Client Friendly", use_container_width=True):
        try:
            # fallback: simple client friendly PDF (kept unchanged)
            pdf_client = pdf_report_no_bodychart(
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
            st.download_button("Scarica PDF Client Friendly", data=pdf_client.getvalue(), file_name=f"Fisiomove_Report_Facile_{st.session_state['date']}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Errore durante generazione PDF semplificato: {e}")
