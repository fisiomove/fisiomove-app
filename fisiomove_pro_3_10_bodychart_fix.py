# (ATTENZIONE: salva questo contenuto in streamlit_app.py, sostituendo il file esistente)
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
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Fisiomove MobilityPro v. 1.0", layout="centered")


# -----------------------------
# Utility: sanitize testo (rimuove emoji e simboli non-ASCII pesanti)
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
    "Wall Angel Test": "Test Wall Angel",
    "Shoulder ER (adducted, low-bar)": "Rotazione esterna spalla (low bar)",
    "Shoulder Flexion (supine)": "Flessione spalla supina",
    "External Rotation (90° abd)": "Rotazione esterna a 90° abduzione",
    "Pectoralis Minor Length": "Lunghezza piccolo pettorale",
    "Thomas Test (modified)": "Test di Thomas modificato",
    "Active Knee Extension (AKE)": "Estensione attiva ginocchio",
    "Straight Leg Raise (SLR)": "Sollevamento gamba tesa",
    "Sorensen Endurance": "Test endurance estensori lombari",
    "ULNT1A (Median nerve)": "Test neurodinamico mediano (ULNT1A)"
}

LOGO_PATHS = ["logo 2600x1000.jpg", "logo.png", "logo.jpg"]
BODYCHART_PATHS = ["8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png", "body_chart.png"]


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


def load_bodychart_image():
    for p in BODYCHART_PATHS:
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                pass
    img = Image.new("RGBA", (1200, 800), (245, 245, 245, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Body chart non disponibile", fill=(10, 10, 10))
    return img


LOGO = load_logo_bytes()
BODYCHART_BASE = load_bodychart_image()


# -----------------------------
# Stato Streamlit
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


# -----------------------------
# Funzioni di scoring
# -----------------------------
def ability_linear(val, ref):
    try:
        if ref <= 0:
            return 0.0
        if ref == 3.0:
            return float(val) / 3.0 * 10.0
        score = (float(val) / float(ref)) * 10.0
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
# Sezioni e TESTS (UNICA DEFINIZIONE)
# -----------------------------
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10.0, True, "ankle", "Test caviglia: dorsiflessione in carico."),
        ("Passive Hip Flexion", "°", 120.0, True, "hip", "Test anca: flessione passiva supina."),
        ("Hip Rotation (flexed 90°)", "°", 40.0, True, "hip", "Test anca: rotazione a 90° flessione."),
        ("Wall Angel Test", "score", 3.0, False, "thoracic", "Contatto scapolare/test posturale (0–3)."),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "Test spalla: extrarotazione low-bar."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Test spalla: flessione supina."),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "Test spalla: ER a 90° abduzione."),
        ("Wall Angel Test", "score", 3.0, False, "thoracic", "Contatto scapolare/test posturale (0–3)."),
        ("Pectoralis Minor Length", "cm", 10.0, True, "shoulder", "Test pettorale: distanza da lettino."),
        ("Thomas Test (modified)", "°", 10.0, True, "hip", "Test flessori anca (modificato)."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90.0, True, "knee", "Test hamstrings: estensione attiva."),
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Test SLR catena posteriore."),
        ("Weight Bearing Lunge Test", "cm", 10.0, True, "ankle", "Test caviglia: dorsiflessione in carico."),
        ("Sorensen Endurance", "sec", 180.0, False, "lumbar", "Test endurance estensori lombari."),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Test neurodinamica posteriore LE."),
        ("ULNT1A (Median nerve)", "°", 90.0, True, "shoulder", "Test neurodinamica arto superiore."),
    ],
}


# -----------------------------
# Seed valori di default
# -----------------------------
def seed_defaults():
    if st.session_state["vals"]:
        if "ULNT1A (Median nerve)" in st.session_state["vals"]:
            rec = st.session_state["vals"]["ULNT1A (Median nerve)"]
            if rec.get("bilat", False):
                rec.setdefault("Dx", 0.0)
                rec.setdefault("Sx", 0.0)
                rec.setdefault("DoloreDx", False)
                rec.setdefault("DoloreSx", False)
        return

    for sec, items in TESTS.items():
        for (name, unit, ref, bilat, region, desc) in items:
            if name not in st.session_state["vals"]:
                if name == "ULNT1A (Median nerve)":
                    if bilat:
                        st.session_state["vals"][name] = {
                            "Dx": 0.0,
                            "Sx": 0.0,
                            "DoloreDx": False,
                            "DoloreSx": False,
                            "unit": unit,
                            "ref": ref,
                            "bilat": True,
                            "region": region,
                            "desc": desc,
                            "section": sec,
                        }
                    else:
                        st.session_state["vals"][name] = {
                            "Val": 0.0,
                            "Dolore": False,
                            "unit": unit,
                            "ref": ref,
                            "bilat": False,
                            "region": region,
                            "desc": desc,
                            "section": sec,
                        }
                else:
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
                        }


seed_defaults()


# -----------------------------
# Costruzione DataFrame (unico per Valutazione Generale, senza duplicati)
# -----------------------------
def build_df(section):
    rows = []
    seen_tests = set()
    for sec, items in TESTS.items():
        if section != "Valutazione Generale" and sec != section:
            continue
        for (name, unit, ref, bilat, region, desc) in items:
            if section == "Valutazione Generale":
                if name in seen_tests:
                    continue
                seen_tests.add(name)

            rec = st.session_state["vals"].get(name)
            if not rec:
                continue
            if bilat:
                dx = pd.to_numeric(rec.get("Dx", 0.0), errors="coerce")
                sx = pd.to_numeric(rec.get("Sx", 0.0), errors="coerce")
                dx = 0.0 if pd.isna(dx) else float(dx)
                sx = 0.0 if pd.isna(sx) else float(sx)
                avg = (dx + sx) / 2.0
                sc = round(ability_linear(avg, ref), 2)
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
                        ref,
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
                val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce")
                val = 0.0 if pd.isna(val) else float(val)
                sc = round(ability_linear(val, ref), 2)
                dolore = bool(rec.get("Dolore", False))
                rows.append([sec, name, unit, ref, f"{val:.1f}", sc, "", "", "", "", dolore, region, False, False])
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
# Plots: Matplotlib (for PDF) and Plotly (UI)
# -----------------------------
def radar_plot_matplotlib(df, title="Punteggi (0–10)"):
    import numpy as np

    labels = df["Test"].tolist()
    values = df["Score"].astype(float).tolist()

    if len(labels) < 3:
        raise ValueError("Servono almeno 3 test per il radar.")

    values += values[:1]
    num_vars = len(labels)
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
    ax.set_xticklabels(labels, fontsize=9)
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
    except Exception as e:
        print(f"Errore nella conversione di SymScore: {e}")
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
# EBM library (text templates + source URLs)
# Each test entry contains:
#  - 'title': short title
#  - 'text': template explanation (will be used if test fails)
#  - 'refs': list of source strings (URLs or citation texts)
# -----------------------------
EBM_LIBRARY = {
    "Weight Bearing Lunge Test": {
        "title": "Ankle dorsiflexion (WBLT)",
        "text": (
            "Ridotta dorsiflessione in carico: il Weight Bearing Lunge Test (WBLT) è una misura affidabile e funzionale "
            "per valutare la dorsiflessione sotto carico. Valori ridotti sono associati a compensi durante squat e cammino, "
            "aumento del rischio di infortuni e limitata performance funzionale. Intervento: mobilizzazione talo-crurale, "
            "lavoro eccentrico/controllo della caviglia e integrazione nella progressione di squat. "
            "Evidenza: studi di validazione e review confermano affidabilità, valori normativi e rilevanza clinica [REFS]."
        ),
        "refs": [
            "https://peerj.com/articles/10253.pdf",
            "https://link.springer.com/article/10.1186/s12891-018-2113-8",
            "https://www.physio-pedia.com/Knee_to_Wall_Test"
        ],
    },
    "Passive Hip Flexion": {
        "title": "Passive hip flexion (ROM)",
        "text": (
            "Ridotta flessione passiva d'anca: valori normativi sono circa 110–120°. Limitazioni possono compromettere la profondità "
            "dello squat e favorire compensi lombo‑pelvici. Intervento: stretching specifico, mobilità articolare e controllo pelvico. "
            "Evidenza: linee guida cliniche e studi di goniometria supportano la rilevanza funzionale del ROM d'anca [REFS]."
        ),
        "refs": [
            "https://www.physiotutors.com/wiki/hip-passive-range-of-motion/",
            "https://link.springer.com/rwe/10.1007/978-1-4614-7321-3_2-2",
            "https://www.orthopt.org/uploads/content_files/files/Hip%20OA%20Revision%202017.pdf"
        ],
    },
    "Hip Rotation (flexed 90°)": {
        "title": "Hip rotation (90° flexion)",
        "text": (
            "Limitata rotazione anca in flessione: la rotazione interna/esterna a 90° è rilevante per movimenti di squat, "
            "pivot e per la biomeccanica dell'arto inferiore. Asimmetrie o deficit possono indicare rischio di infortuni o "
            "patologie femoro‑acetabolari; interventi includono mobilità specifica e controllo neuromuscolare [REFS]."
        ),
        "refs": [
            "https://www.jstage.jst.go.jp/article/jpts/27/2/27_jpts-2014-454/_pdf",
            "https://www.mdpi.com/2673-8392/5/4/170"
        ],
    },
    "Wall Angel Test": {
        "title": "Wall Angel (scapular control / thoracic mobility)",
        "text": (
            "Punteggio ridotto al Wall Angel: suggerisce deficit di controllo scapolare e mobilità toracica, potenziali compensi agli overhead "
            "movimenti e dolore riferito. Intervento: esercizi di estensione toracica, rinforzo dei stabilizzatori scapolari e retraining del pattern motorio. "
            "Evidenza: ricerche recenti supportano l'utilità clinica del Wall Angel come test funzionale correlato a outcome riabilitativi [REFS]."
        ),
        "refs": [
            "https://ijspt.scholasticahq.com/article/123512-the-clinical-utility-of-the-seated-wall-angel-as-a-test-with-scoring",
            "https://www.physitrack.com/exercise-library/how-to-perform-the-wall-angels-exercise"
        ],
    },
    "Shoulder ER (adducted, low-bar)": {
        "title": "Shoulder ER (adducted, low-bar)",
        "text": (
            "Ridotta rotazione esterna spalla in adduzione: può impedire una corretta posizione \"low‑bar\" nella sollevamento del bilanciere, "
            "causando compensi a polso/avambraccio o una sistemazione più alta del bilanciere. Intervento: mobilità capsulare, esercizi di ER progressive, "
            "control della scapola. Evidenza: guide cliniche e testi di goniometria indicano che deficit di ER hanno impatto funzionale rilevante [REFS]."
        ),
        "refs": [
            "https://www.physio-pedia.com/Goniometry:_Shoulder_Internal_%26_External_Rotation",
            "https://www.orthopaedicsone.com/orthopaedicsone-articles-shoulder-external-rotation-assessment/"
        ],
    },
    "Shoulder Flexion (supine)": {
        "title": "Shoulder flexion (supine)",
        "text": (
            "Ridotta flessione spalla: limita attività overhead e può indicare deficit attivi o passivi (cuffia, capsula). Interventi: mobilità passiva, "
            "rinforzo e recupero del pattern scapolo‑omerale. Evidenza: studi di norma e linee guida descrivono l'importanza della misurazione del ROM [REFS]."
        ),
        "refs": [
            "https://link.springer.com/article/10.1186/s12891-020-03665-9",
            "https://www.massgeneral.org/assets/mgh/pdf/orthopaedics/sports-medicine/physical-therapy/functional-movement-assessment-upper-extremity.pdf"
        ],
    },
    "External Rotation (90° abd)": {
        "title": "Shoulder ER at 90° abduction",
        "text": (
            "Limitazione a ER a 90° di abduzione: importante per overhead e per stabilità dinamica; deficit ha implicazioni per prevenzione infortuni e performance. "
            "Esercizi specifici di ER e rinforzo scapolare sono raccomandati [REFS]."
        ),
        "refs": [
            "https://www.mdpi.com/2227-9032/11/14/1977",
            "https://iaom-us.com/restoring-external-rotation-in-the-shoulder/"
        ],
    },
    "Pectoralis Minor Length": {
        "title": "Pectoralis minor length",
        "text": (
            "Accorciamento del piccolo pettorale: associato a proiezione scapolare anteriore, diminuita rotazione superiore e posterior tilt ridotto, contribuendo a sindromi da impingement. "
            "Intervento: stretching mirato del PM, riequilibrio muscolare e rinforzo dei muscoli antagonisti. Evidenza: studi mostrano associazione; utilizzare in combinazione con altri test [REFS]."
        ),
        "refs": [
            "https://pdfs.semanticscholar.org/3d47/2b3c59ffda2f7f7e97e7d6388114e4237d11.pdf",
            "https://uhra.herts.ac.uk/id/eprint/3916/"
        ],
    },
    "Thomas Test (modified)": {
        "title": "Modified Thomas Test (hip flexor tightness)",
        "text": (
            "Test di Thomas positivo: indica accorciamento dei flessori d'anca (iliopsoas, retto femorale). Può contribuire a limitazione dell'estensione dell'anca e compensi lombari. "
            "Intervento: rilascio, mobilità e lavoro di controllo pelvico. Studi recenti segnalano buona affidabilità con protocolli standardizzati [REFS]."
        ),
        "refs": [
            "https://ijspt.scholasticahq.com/article/120899-reliability-of-goniometric-techniques-for-measuring-hip-flexor-length-using-the-modified-thomas-test",
            "https://www.physio-pedia.com/Thomas_Test"
        ],
    },
    "Active Knee Extension (AKE)": {
        "title": "Active Knee Extension (AKE)",
        "text": (
            "AKE positivo: ridotta lunghezza degli hamstrings; associato a rischio di infortunio e alterata meccanica lombo‑pelvica. Interventi: stretching funzionale, "
            "eccentrico e integrazione neuromuscolare. AKE è un test affidabile per hamstring length [REFS]."
        ),
        "refs": [
            "https://www.physiotutors.com/wiki/90-90-straight-leg-raise-test/",
            "https://www.jstage.jst.go.jp/article/jpts/25/8/25_jpts-2013-059/_pdf"
        ],
    },
    "Straight Leg Raise (SLR)": {
        "title": "Straight Leg Raise (SLR)",
        "text": (
            "SLR positivo: attenzione a distinzione tra limitazione muscolare e sensibilità neurodinamica (sciatica/irritazione radicolare). Usare differenziazione strutturale "
            "(dorsiflessione caviglia, flessione collo) per discriminare. Intervento: se neurodinamico, trattamenti specifici per nervo; se muscolare, lavoro di flessibilità [REFS]."
        ),
        "refs": [
            "https://www.physio-pedia.com/Straight_Leg_Raise_Test",
            "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298257"
        ],
    },
    "Sorensen Endurance": {
        "title": "Sorensen (lumbar endurance)",
        "text": (
            "Basso tempo al test di Sorensen: deficit di endurance estensori lombari associato a rischio e presenza di lombalgia. Interventi: progessioni di endurance lombare e integrazione "
            "di controllo motorio. Interpretare con cautela e considerare fattori psico‑sociali e motivazionali [REFS]."
        ),
        "refs": [
            "https://www.sralab.org/rehabilitation-measures/biering-sorensen-test",
            "https://www.sciencedirect.com/science/article/pii/S1297319X04001708"
        ],
    },
    "ULNT1A (Median nerve)": {
        "title": "ULNT1A (median nerve)",
        "text": (
            "ULNT1A: test neurodinamico del nervo mediano. Valori prossimi a 0° (o riproduzione sintomatica precoce) indicano meccanica neurale alterata o irritazione; miglioramenti nella mobilità neurale "
            "possono essere ottenuti con tecniche neurali e mobilità perimetrica. Interpretare con structural differentiation. [REFS]"
        ),
        "refs": [
            "https://www.physiotutors.com/research/upper-limb-neurodynamic-test-1-sequencing/",
            "https://www.mdpi.com/2075-4418/14/24/2881"
        ],
    },
}


# -----------------------------
# Costruzione commenti EBM + bibliografia
# - Restituisce (notes_list, bibliography_list)
# - notes_list: list of strings (narrative paragraphs with inline [n] markers)
# - bibliography_list: list of citation strings (numbered later)
# -----------------------------
def ebm_from_df(df, friendly=False):
    notes = []
    biblio_urls = []
    # map URL -> index in bibliography
    bib_index = {}

    def add_ref(url):
        if not url:
            return None
        if url in bib_index:
            return bib_index[url]
        idx = len(biblio_urls) + 1
        bib_index[url] = idx
        biblio_urls.append(url)
        return idx

    problematic = []

    for _, r in df.iterrows():
        test = str(r["Test"])
        score = float(r["Score"]) if not pd.isna(r["Score"]) else 10.0
        pain = bool(r["Dolore"])
        sym = float(r.get("SymScore", 10.0) if not pd.isna(r.get("SymScore", np.nan)) else 10.0)

        entry = EBM_LIBRARY.get(test)
        if not entry:
            continue

        para_lines = []
        issue = False

        # if score low -> detailed paragraph
        if score < 7:
            # build paragraph using entry text and attach inline refs
            refs = entry.get("refs", [])
            ref_nums = []
            for u in refs:
                n = add_ref(u)
                if n:
                    ref_nums.append(f"[{n}]")
            # insert refs into text where placeholder [REFS] appears
            text = entry["text"].replace("[REFS]", " ".join(ref_nums))
            para_lines.append(text)
            issue = True

        # asymmetry
        if sym < 7:
            para_lines.append(
                f"Asimmetria significativa tra Dx/Sx (SymScore {sym:.1f}/10): suggerisce differenze laterali che possono richiedere priorità di intervento sul lato peggiore."
            )
            issue = True

        # pain
        if pain:
            para_lines.append("Dolore riportato durante il test: valutare intensità, distribuzione e relazione con il movimento rilevato.")
            issue = True

        if not issue:
            para_lines.append(f"Il test '{test}' risulta nella norma (score {score:.1f}/10).")

        notes.append(" ".join(para_lines))
        if issue:
            problematic.append(test)

    # Build bibliography list (human readable)
    bibliography = []
    for i, url in enumerate(biblio_urls, start=1):
        # try to shorten: use URL as citation; user can expand later
        bibliography.append(f"[{i}] {url}")

    return notes, bibliography


# -----------------------------
# PDF generation (restyled)
# - Now accepts ebm_bibliography (list of strings)
# -----------------------------
def pdf_report_no_bodychart(logo_bytes, athlete, evaluator, date_str, section, df, ebm_notes, ebm_bibliography=None, radar_buf=None, asym_buf=None):
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    heading = ParagraphStyle("heading", parent=styles["Heading2"], alignment=TA_LEFT, textColor=colors.HexColor(PRIMARY))
    centerb = ParagraphStyle("centerb", parent=styles["Heading3"], alignment=TA_CENTER)

    story = []

    # Header: logo + title
    story.append(RLImage(io.BytesIO(logo_bytes), width=14 * cm, height=3.2 * cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Report Valutazione — {sanitize_text_for_plot(section)}</b>", title))
    story.append(Spacer(1, 6))

    # Info band
    info_data = [
        ["Atleta", athlete, "Valutatore", evaluator, "Data", date_str],
    ]
    info_table = Table(info_data, colWidths=[2.2 * cm, 6.0 * cm, 2.8 * cm, 5.0 * cm, 1.6 * cm, 2.0 * cm])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FF")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 8))

    # Summary metrics
    avg_score = df["Score"].mean() if "Score" in df.columns and not df["Score"].isna().all() else 0.0
    n_dolore = int(df["Dolore"].sum()) if "Dolore" in df.columns else 0
    sym_mean = df["SymScore"].mean() if "SymScore" in df.columns else np.nan
    metrics = Table(
        [
            ["Score medio", f"{avg_score:.1f}/10", "Test con dolore", str(n_dolore), "Symmetry medio", f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a"]
        ],
        colWidths=[3 * cm, 3 * cm, 3 * cm, 2.6 * cm, 3 * cm, 3 * cm],
    )
    metrics.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(metrics)
    story.append(Spacer(1, 12))

    # Results table (styled)
    disp = df[["Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]].copy()
    # Rimuovi eventuali test non desiderati
    disp = disp[~disp["Test"].str.lower().str.contains("schober", na=False)]
    for col in ["Valore", "Score", "Dx", "Sx", "Delta", "SymScore"]:
        disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)

    # Color score cell: convert to strings for Table
    table_data = [disp.columns.tolist()]
    for _, row in disp.iterrows():
        row_list = row.tolist()
        table_data.append(row_list)

    # Column widths
    colWidths = [2.2 * cm, 6.0 * cm, 1.2 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm]
    table = Table(table_data, repeatRows=1, colWidths=colWidths)
    # style header + grid
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    # add conditional background for Score column (col index 5)
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

    # Charts area
    charts = []
    if radar_buf:
        charts.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10 * cm, height=10 * cm))
    if asym_buf:
        charts.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14 * cm, height=6 * cm))
    # Place charts: radar then asymmetry
    if charts:
        for ch in charts:
            story.append(ch)
            story.append(Spacer(1, 8))

    # Painful regions
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

    # EBM Comments (narrative)
    story.append(Paragraph("<b>Commento clinico (EBM)</b>", heading))
    story.append(Spacer(1, 6))
    if ebm_notes:
        for para in ebm_notes:
            # sanitize
            story.append(Paragraph(sanitize_text_for_plot(para), normal))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("Nessun commento disponibile.", normal))

    story.append(Spacer(1, 14))

    # Bibliography final
    if ebm_bibliography:
        story.append(Paragraph("<b>Bibliografia</b>", heading))
        story.append(Spacer(1, 6))
        for ref in ebm_bibliography:
            story.append(Paragraph(sanitize_text_for_plot(ref), small))
            story.append(Spacer(1, 4))

    # Footer with signature
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Valutatore: {evaluator}", small))
    story.append(Paragraph("Firma: ______________________", small))
    story.append(Spacer(1, 6))

    doc.build(story)
    buf.seek(0)
    return buf


def pdf_report_client_friendly(logo_bytes, athlete, evaluator, date_str, section, df, radar_buf=None, asym_buf=None):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]

    story = []
    story.append(RLImage(io.BytesIO(logo_bytes), width=14 * cm, height=3.2 * cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Valutazione Funzionale – {sanitize_text_for_plot(section)}</b>", title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"Atleta: {athlete}", normal))
    story.append(Paragraph(f"Valutatore: {evaluator}", normal))
    story.append(Paragraph(f"Data: {date_str}", normal))
    story.append(Spacer(1, 12))

    if radar_buf:
        story.append(Paragraph("<b>Radar delle capacità funzionali</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10 * cm, height=10 * cm))
        story.append(Spacer(1, 12))

    if asym_buf:
        story.append(Paragraph("<b>Simmetria tra lato destro e sinistro</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14 * cm, height=6 * cm))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Ogni test è valutato su un punteggio da 0 a 10.", normal))
    story.append(Paragraph("0–3: da migliorare • 4–6: accettabile • 7–10: ottimale", normal))
    story.append(Spacer(1, 10))

    simple_rows = []
    for _, r in df.iterrows():
        score = round(float(r["Score"]) if not pd.isna(r["Score"]) else 0.0, 1)
        test_name = TEST_NAME_TRANSLATIONS.get(r["Test"], r["Test"])
        simple_rows.append([test_name, f"{score}/10"])

    t = Table([["Test", "Punteggio"]] + simple_rows, repeatRows=1, colWidths=[10 * cm, 4 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(t)

    doc.build(story)
    buf.seek(0)
    return buf


# -----------------------------
# UI Styling & Header
# -----------------------------
st.markdown(
    f"""
<style>
:root {{ --primary: {PRIMARY}; --accent: {ACCENT}; }}
body {{ background: #f6f8fb; }}
.header-card {{ background: linear-gradient(90deg, #ffffff, #f1f7ff); padding:12px; border-radius:12px; box-shadow: 0 6px 18px rgba(16,24,40,0.04); }}
.card {{ background: white; padding:12px; border-radius:10px; box-shadow: 0 3px 10px rgba(16,24,40,0.04); margin-bottom:10px; }}
.small-muted {{ color:#6b7280; font-size:0.9rem; }}
.badge-ok {{ background: #ecfdf5; color: #166534; padding:4px 8px; border-radius:999px; font-weight:600; }}
.badge-warn {{ background:#fff7ed; color:#92400e; padding:4px 8px; border-radius:999px; font-weight:600; }}
.badge-bad {{ background:#fff1f2; color:#7f1d1d; padding:4px 8px; border-radius:999px; font-weight:600; }}
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


# -----------------------------
# Sidebar UI
# -----------------------------
ALL_SECTIONS = ["Valutazione Generale"]

with st.sidebar:
    st.markdown("### Dati atleta")
    st.session_state["athlete"] = st.text_input("Atleta", st.session_state["athlete"])
    st.session_state["evaluator"] = st.text_input("Valutatore", st.session_state["evaluator"])
    st.session_state["date"] = st.date_input("Data", datetime.strptime(st.session_state["date"], "%Y-%m-%d")).strftime(
        "%Y-%m-%d"
    )

    st.markdown("---")
    st.session_state["section"] = st.selectbox("Sezione", ALL_SECTIONS, index=0)

    colb1, colb2 = st.columns(2)
    with colb1:
        if st.button("Reset valori", use_container_width=True):
            st.session_state["vals"].clear()
            seed_defaults()
            st.experimental_rerun()
    with colb2:
        if st.button("Randomizza", use_container_width=True):
            for name, rec in st.session_state["vals"].items():
                if name == "ULNT1A (Median nerve)":
                    continue
                ref = rec.get("ref", 10.0)
                if rec.get("bilat", False):
                    rec["Dx"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["Sx"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["DoloreDx"] = random.random() < 0.15
                    rec["DoloreSx"] = random.random() < 0.15
                else:
                    rec["Val"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["Dolore"] = random.random() < 0.15
            st.success("Valori random impostati.")


# -----------------------------
# Render inputs grouped by region
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
    for sec, name, unit, ref, bilat, region, desc in tests:
        region_map.setdefault(region or "other", []).append((sec, name, unit, ref, bilat, region, desc))

    for region, items in region_map.items():
        with st.expander(region.capitalize(), expanded=False):
            for sec, name, unit, ref, bilat, region, desc in items:
                rec = st.session_state["vals"].get(name)
                if not rec:
                    continue
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{name}**  \n*{desc}*  \n*Rif:* {ref} {unit}")
                key = short_key(name)
                max_val = ref if name == "ULNT1A (Median nerve)" else (ref * 1.5 if ref > 0 else 10.0)

                if bilat:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        dx = st.slider(f"Dx ({unit})", 0.0, max_val, float(rec.get("Dx", 0.0)), 0.1, key=f"{key}_Dx")
                        pdx = st.checkbox("Dolore Dx", value=bool(rec.get("DoloreDx", False)), key=f"{key}_pDx")
                    with c2:
                        sx = st.slider(f"Sx ({unit})", 0.0, max_val, float(rec.get("Sx", 0.0)), 0.1, key=f"{key}_Sx")
                        psx = st.checkbox("Dolore Sx", value=bool(rec.get("DoloreSx", False)), key=f"{key}_pSx")
                    rec.update({"Dx": dx, "Sx": sx, "DoloreDx": pdx, "DoloreSx": psx})
                    sc = ability_linear((dx + sx) / 2.0, ref)
                    sym = symmetry_score(dx, sx, unit)
                    status = "badge-ok" if sc >= 7 else ("badge-warn" if sc >= 4 else "badge-bad")
                    st.markdown(f"<div style='margin-top:8px'><span class='{status}'>Score {sc:.1f}/10</span>  — Δ {abs(dx - sx):.1f} {unit} — Sym: <b>{sym:.1f}/10</b></div>", unsafe_allow_html=True)
                else:
                    val = st.slider(f"Valore ({unit})", 0.0, max_val, float(rec.get("Val", 0.0)), 0.1, key=f"{key}_Val")
                    p = st.checkbox("Dolore", value=bool(rec.get("Dolore", False)), key=f"{key}_p")
                    rec.update({"Val": val, "Dolore": p})
                    sc = ability_linear(val, ref)
                    status = "badge-ok" if sc >= 7 else ("badge-warn" if sc >= 4 else "badge-bad")
                    st.markdown(f"<div style='margin-top:8px'><span class='{status}'>Score {sc:.1f}/10</span></div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


render_inputs_for_section(st.session_state["section"])


# -----------------------------
# Build DF and visuals
# -----------------------------
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


# -----------------------------
# Prepare PDF buffers
# -----------------------------
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


# -----------------------------
# EBM notes + bibliography
# -----------------------------
ebm_notes, ebm_bibliography = ebm_from_df(df_show, friendly=False)


# -----------------------------
# Export PDFs
# -----------------------------
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
                ebm_bibliography=ebm_bibliography,
                radar_buf=radar_buf,
                asym_buf=asym_buf,
            )
            st.download_button(
                "Scarica PDF Clinico",
                data=pdf.getvalue(),
                file_name=f"Fisiomove_Report_Clinico_{st.session_state['date']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Errore durante generazione PDF clinico: {e}")

with colpdf2:
    if st.button("Esporta PDF Client Friendly", use_container_width=True):
        try:
            pdf_client = pdf_report_client_friendly(
                logo_bytes=LOGO,
                athlete=st.session_state["athlete"],
                evaluator=st.session_state["evaluator"],
                date_str=st.session_state["date"],
                section=st.session_state["section"],
                df=df_show,
                radar_buf=radar_buf,
                asym_buf=asym_buf,
            )
            st.download_button(
                "Scarica PDF Client Friendly",
                data=pdf_client.getvalue(),
                file_name=f"Fisiomove_Report_Facile_{st.session_state['date']}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Errore durante generazione PDF semplificato: {e}")
