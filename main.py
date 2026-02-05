# Fisiomove MobilityPro - Complete Physiotherapy Assessment System
# Version 2.0 - Enhanced with clinical features
# Run with: streamlit run streamlit_app.py

import io
import os
import json
import random
import re
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import plotly.graph_objects as go
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
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader

# Optional QR code support
try:
    import qrcode
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

st.set_page_config(page_title="Fisiomove MobilityPro", layout="wide")

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
SUBTITLE = "Sistema Completo di Valutazione Fisioterapica — v2.0"
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
# Clinical Data Structures
# -----------------------------

# Red Flags
RED_FLAGS = {
    "trauma_recente": "Trauma significativo nelle ultime 48-72h",
    "dolore_notturno": "Dolore notturno che interrompe il sonno",
    "perdita_forza": "Perdita improvvisa di forza muscolare",
    "febbre": "Febbre o sintomi sistemici",
    "intorpidimento": "Intorpidimento/formicolio persistente",
    "perdita_peso": "Perdita di peso non intenzionale",
    "dolore_toracico": "Dolore toracico o difficoltà respiratorie",
    "incontinenza": "Perdita controllo sfinterico"
}

# Anamnesis structure
SPORTS_LIST = [
    "Powerlifting",
    "Weightlifting", 
    "CrossFit",
    "Bodybuilding",
    "Functional Training",
    "Sport di squadra",
    "Endurance",
    "Altro"
]

PAIN_BEHAVIORS = [
    "Meccanico (peggiora con movimento/carico)",
    "Infiammatorio (rigidità mattutina >30min)",
    "Neuropatico (bruciore/formicolio/scossa)",
    "Misto"
]

AGGRAVATING_FACTORS = [
    "Carico pesante (>80% 1RM)",
    "Range finale movimento",
    "Posizioni statiche prolungate",
    "Movimenti ripetitivi",
    "Attività specifiche (specificare in note)",
    "Mattina al risveglio",
    "Sera dopo allenamento"
]

RELIEVING_FACTORS = [
    "Riposo",
    "Movimento leggero",
    "Stretching",
    "Calore",
    "Ghiaccio",
    "Farmaci antinfiammatori",
    "Terapia manuale"
]

# Tests definitions
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

# Movement quality assessments
MOVEMENT_QUALITY_TESTS = {
    "Overhead Squat Assessment": {
        "parametri": [
            "Braccia cadono in avanti",
            "Tronco si inclina eccessivamente",
            "Ginocchia collassano in valgo",
            "Talloni si sollevano",
            "Asimmetrie destra/sinistra"
        ],
        "scoring": ["No compensi", "Compensi lievi", "Compensi marcati"]
    },
    "Single Leg Squat": {
        "parametri": [
            "Valgo dinamico ginocchio",
            "Drop pelvico (Trendelenburg)",
            "Rotazione tronco",
            "Equilibrio instabile",
            "Controllo discesa"
        ],
        "scoring": ["Ottimo", "Accettabile", "Deficit"]
    },
    "Plank Test": {
        "parametri": [
            "Allineamento corpo",
            "Stabilità scapolare",
            "Controllo lombo-pelvico"
        ],
        "scoring": ["Ottimo (>60s)", "Buono (30-60s)", "Deficit (<30s)"]
    }
}

# Exercise protocols
EXERCISE_PROTOCOLS = {
    "ankle_mobility": {
        "nome": "Protocollo Mobilità Caviglia",
        "esercizi": [
            "Wall ankle mobilization con ginocchio piegato: 3x30sec per lato",
            "Heel elevated goblet squat: 3x10 ripetizioni",
            "Banded dorsiflexion: 3x15 ripetizioni",
            "Calf stretch eccentrico: 3x12 ripetizioni"
        ],
        "frequenza": "Giornaliera (anche giorni di riposo)",
        "durata": "2-4 settimane",
        "progressione": "Aumentare ROM e resistenza progressivamente"
    },
    "hip_rotation": {
        "nome": "Protocollo Mobilità Rotazione Anca",
        "esercizi": [
            "90/90 hip stretch: 3x45sec per lato",
            "Cossack squat: 3x8 ripetizioni per lato",
            "Hip CARs (Controlled Articular Rotations): 2x5 per direzione",
            "Pigeon stretch: 2x60sec per lato"
        ],
        "frequenza": "5 volte/settimana",
        "durata": "3-6 settimane",
        "progressione": "Aumentare ROM, aggiungere carico leggero"
    },
    "hip_flexion": {
        "nome": "Protocollo Flessione Anca",
        "esercizi": [
            "Supine hip flexion con band: 3x12",
            "Lying leg raises: 3x10",
            "Dead bug variations: 3x8 per lato",
            "Deep squat hold: 3x30sec"
        ],
        "frequenza": "4-5 volte/settimana",
        "durata": "3-4 settimane",
        "progressione": "Aumentare tempo sotto tensione"
    },
    "thoracic_mobility": {
        "nome": "Protocollo Mobilità Toracica",
        "esercizi": [
            "Thoracic extension su foam roller: 3x10",
            "Thread the needle: 3x8 per lato",
            "Wall slides: 3x12",
            "Cat-cow: 2x10 ripetizioni"
        ],
        "frequenza": "Giornaliera",
        "durata": "2-3 settimane",
        "progressione": "Aumentare ROM e controllo"
    },
    "shoulder_mobility": {
        "nome": "Protocollo Mobilità Spalla",
        "esercizi": [
            "Sleeper stretch: 3x30sec per lato",
            "Cross-body stretch: 3x30sec",
            "Band pull-aparts: 3x15",
            "Wall angels: 3x10",
            "Shoulder CARs: 2x5 per direzione"
        ],
        "frequenza": "6 volte/settimana",
        "durata": "4-6 settimane",
        "progressione": "Aumentare ROM, aggiungere rotazioni con carico"
    },
    "hamstring_length": {
        "nome": "Protocollo Lunghezza Hamstrings",
        "esercizi": [
            "Neural glides in supine: 3x10",
            "Single leg RDL leggero: 3x10 per lato",
            "Eccentric hamstring curls: 3x6",
            "PNF contract-relax: 3x30sec"
        ],
        "frequenza": "4 volte/settimana",
        "durata": "4-8 settimane",
        "progressione": "Aumentare carico eccentrico gradualmente"
    },
    "core_endurance": {
        "nome": "Protocollo Endurance Core/Lombare",
        "esercizi": [
            "Plank progressions: 3-5 sets, tempo crescente",
            "Bird dog: 3x8 per lato con hold 3sec",
            "Dead bug: 3x10 alternati",
            "Sorensen hold: 3 sets progressivi"
        ],
        "frequenza": "3-4 volte/settimana",
        "durata": "4-6 settimane",
        "progressione": "Aumentare tempo di hold del 10% settimanale"
    },
    "hip_flexors": {
        "nome": "Protocollo Lunghezza Flessori Anca",
        "esercizi": [
            "Half-kneeling hip flexor stretch: 3x45sec",
            "Couch stretch: 2x60sec per lato",
            "Bulgarian split squat: 3x8 per lato",
            "Dead bug con focus anti-estensione: 3x10"
        ],
        "frequenza": "Giornaliera",
        "durata": "4-6 settimane",
        "progressione": "Aumentare ROM stretch, carico split squat"
    },
    "neural_mobility": {
        "nome": "Protocollo Mobilità Neurale",
        "esercizi": [
            "Neural flossing SLR: 3x10 oscillazioni",
            "Slump stretch progressivo: 3x30sec",
            "ULNT1 self-mobilization: 3x10 per lato",
            "Nerve glides cervicali: 2x10"
        ],
        "frequenza": "Giornaliera (bassa intensità)",
        "durata": "3-6 settimane",
        "progressione": "Aumentare ROM gradualmente, evitare provocazione sintomi"
    }
}

# Sport-specific critical thresholds
SPORT_SPECIFIC_INTERPRETATION = {
    "Powerlifting": {
        "Squat": {
            "critical_tests": ["Weight Bearing Lunge Test", "Hip Rotation (flexed 90°)", "Passive Hip Flexion"],
            "threshold": 7.0,
            "note": "Mobilità caviglia critica per depth ATG; ROM anca essenziale per stance largo"
        },
        "Panca": {
            "critical_tests": ["Shoulder ER (adducted, low-bar)", "Pectoralis Minor Length", "Shoulder Flexion (supine)"],
            "threshold": 6.5,
            "note": "Retrazione scapolare e stabilità spalla essenziali per setup sicuro"
        },
        "Deadlift": {
            "critical_tests": ["Active Knee Extension (AKE)", "Straight Leg Raise (SLR)", "Sorensen Endurance"],
            "threshold": 7.0,
            "note": "Lunghezza hamstrings per setup ottimale; endurance lombare per volume"
        }
    },
    "CrossFit": {
        "overhead_movements": {
            "critical_tests": ["Shoulder Flexion (supine)", "External Rotation (90° abd)", "Wall Angel Test"],
            "threshold": 7.5,
            "note": "ROM spalla completo essenziale per snatch, overhead squat"
        }
    },
    "Weightlifting": {
        "snatch_clean": {
            "critical_tests": ["Weight Bearing Lunge Test", "Hip Rotation (flexed 90°)", "Shoulder Flexion (supine)"],
            "threshold": 8.0,
            "note": "Mobilità caviglia e anca critiche per receiving position profonda"
        }
    }
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

# PDF labels
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

# EBM Library
EBM_LIBRARY = {
    "Weight Bearing Lunge Test": {
        "title": "Dorsiflessione caviglia (WBLT)",
        "text": "Test: WBLT — dorsiflessione in carico. Interpretazione: valuta mobilità tibio‑talarica e simmetria. Valore <10cm associato a rischio aumentato di infortuni arto inferiore.",
    },
    "Passive Hip Flexion": {
        "title": "Flessione anca passiva",
        "text": "Test: flessione passiva. Interpretazione: misura il ROM passivo dell'anca. Deficit (<110°) può limitare profondità squat.",
    },
    "Hip Rotation (flexed 90°)": {
        "title": "Rotazione anca (flessione 90°)",
        "text": "Test: rotazione in flessione 90°. Interpretazione: ROM rotazionale funzionale. Asimmetrie >15° possono indicare problematiche articolari.",
    },
    "Wall Angel Test": {
        "title": "Wall Angel",
        "text": "Test: distanza cm tra braccio e muro. Interpretazione: valori maggiori indicano maggiore rigidità toracica (scala invertita). Importante per overhead e bench press.",
    },
    "Pectoralis Minor Length": {
        "title": "Lunghezza piccolo pettorale",
        "text": "Test: lunghezza PM. Interpretazione: valori più bassi indicano maggiore mobilità (scala invertita). Accorciamento può causare discinesia scapolare.",
    },
    "Thomas Test (modified)": {
        "title": "Thomas test (modificato)",
        "text": "Test: accorciamento flessori d'anca. Interpretazione: deficit in gradi rispetto a 0°. Accorciamento può influenzare estensione anca in deadlift.",
    },
    "Active Knee Extension (AKE)": {
        "title": "AKE",
        "text": "Test: estensione attiva ginocchio (90/90). Interpretazione: lunghezza hamstrings. Deficit può limitare setup deadlift e aumentare carico lombare.",
    },
    "Straight Leg Raise (SLR)": {
        "title": "SLR",
        "text": "Test: SLR. Interpretazione: differenziare componente muscolare da neurale. <70° con dorsiflex+ indica tensione neurale.",
    },
    "Sorensen Endurance": {
        "title": "Sorensen",
        "text": "Test: endurance lombare (secondi). Interpretazione: tempi ridotti (<60s) indicano deficit di endurance, fattore di rischio per low back pain.",
    },
    "ULNT1A (Median nerve)": {
        "title": "ULNT1A",
        "text": "Test: ULNT1A. Interpretazione: mobilità neurale e riproduzione dei sintomi. Positivo se riproduce sintomi con desensibilizzazione cervicale.",
    },
    "Shoulder ER (adducted, low-bar)": {
        "title": "Rotazione esterna spalla",
        "text": "Test: ER in adduzione. Interpretazione: capacità di ER per posizionamento low‑bar. Deficit può causare dolore spalla in squat.",
    },
    "Shoulder Flexion (supine)": {
        "title": "Flessione spalla",
        "text": "Test: flessione spalla supina. Interpretazione: differenza attivo/passivo indica controllo o limitazione capsulare. Essenziale per movimenti overhead.",
    },
    "External Rotation (90° abd)": {
        "title": "ER 90° abd",
        "text": "Test: ER a 90° abduzione. Interpretazione: mobilità e stabilità per overhead. Deficit comune in atleti con volume alto di pressing.",
    },
}

TEST_INSTRUCTIONS = {k: v["text"] for k, v in EBM_LIBRARY.items()}

# -----------------------------
# File management utilities
# -----------------------------
ASSESSMENTS_DIR = Path("assessments_data")
ASSESSMENTS_DIR.mkdir(exist_ok=True)

def save_assessment_to_file(assessment_data):
    """Save assessment to JSON file"""
    athlete_name = assessment_data["athlete"].replace(" ", "_")
    timestamp = assessment_data["date"]
    filename = f"{athlete_name}_{timestamp}.json"
    filepath = ASSESSMENTS_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(assessment_data, f, ensure_ascii=False, indent=2)
    
    return filepath

def load_athlete_history(athlete_name):
    """Load all assessments for an athlete"""
    athlete_slug = athlete_name.replace(" ", "_")
    history = []
    
    for file in ASSESSMENTS_DIR.glob(f"{athlete_slug}_*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                history.append(data)
        except Exception:
            continue
    
    # Sort by date
    history.sort(key=lambda x: x.get("date", ""), reverse=True)
    return history

def get_all_athletes():
    """Get list of all athletes with assessments"""
    athletes = set()
    for file in ASSESSMENTS_DIR.glob("*.json"):
        parts = file.stem.split("_")
        if len(parts) >= 2:
            athlete_name = "_".join(parts[:-1]).replace("_", " ")
            athletes.add(athlete_name)
    return sorted(list(athletes))

# -----------------------------
# Session state initialization
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
    
    # Anamnesis
    if "sport" not in st.session_state:
        st.session_state["sport"] = "Powerlifting"
    if "training_frequency" not in st.session_state:
        st.session_state["training_frequency"] = 4
    if "injury_history" not in st.session_state:
        st.session_state["injury_history"] = ""
    if "current_symptoms" not in st.session_state:
        st.session_state["current_symptoms"] = ""
    if "goals" not in st.session_state:
        st.session_state["goals"] = ""
    
    # Red flags
    if "red_flags" not in st.session_state:
        st.session_state["red_flags"] = []
    
    # Functional scales
    if "nprs" not in st.session_state:
        st.session_state["nprs"] = 0
    if "psfs_activities" not in st.session_state:
        st.session_state["psfs_activities"] = [
            {"activity": "Squat profondo", "score": 10},
            {"activity": "Corsa", "score": 10},
            {"activity": "Overhead press", "score": 10}
        ]
    
    # Pain assessment
    if "pain_behavior" not in st.session_state:
        st.session_state["pain_behavior"] = []
    if "aggravating_factors" not in st.session_state:
        st.session_state["aggravating_factors"] = []
    if "relieving_factors" not in st.session_state:
        st.session_state["relieving_factors"] = []
    
    # Movement quality
    if "movement_quality" not in st.session_state:
        st.session_state["movement_quality"] = {}
    
    # Clinical notes
    if "clinical_notes" not in st.session_state:
        st.session_state["clinical_notes"] = ""
    if "postural_observations" not in st.session_state:
        st.session_state["postural_observations"] = ""

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
# Scoring and validation
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

def validate_input(test_name, value, side=None):
    """Validate input values and provide clinical warnings"""
    warnings = []
    
    if test_name == "Weight Bearing Lunge Test":
        if value > 20:
            warnings.append("⚠️ Valore inusuale per WBLT (>20cm). Verificare misurazione.")
        elif value < 8:
            warnings.append("⚠️ Mobilità caviglia limitata. Rischio compensi in squat.")
    
    elif test_name == "Straight Leg Raise (SLR)":
        if value < 40:
            warnings.append("🚨 SLR <40° - possibile patologia significativa. Considerare imaging.")
        elif value < 70:
            warnings.append("⚠️ SLR ridotto. Valutare componente neurale vs muscolare.")
    
    elif test_name == "Passive Hip Flexion":
        if value < 100:
            warnings.append("⚠️ Flessione anca marcatamente ridotta. Limitazione significativa per squat.")
    
    elif test_name == "Sorensen Endurance":
        if value < 30:
            warnings.append("🚨 Endurance lombare critica (<30s). Alto rischio low back pain.")
        elif value < 60:
            warnings.append("⚠️ Endurance lombare sotto norma. Priorità allenamento core.")
    
    return warnings

# -----------------------------
# Clinical algorithms
# -----------------------------
def check_risk_factors(df, session_state):
    """Identify risk factors based on data patterns"""
    alerts = []
    
    # Check for bilateral pain in same region
    bilateral_pain = df[(df["DoloreDx"] == True) & (df["DoloreSx"] == True)]
    if not bilateral_pain.empty:
        regions = bilateral_pain["Regione"].unique()
        alerts.append({
            "level": "warning",
            "message": f"⚠️ Dolore bilaterale in: {', '.join(regions)}. Considerare cause sistemiche/centrali."
        })
    
    # Significant asymmetry + injury history
    if df["SymScore"].min() < 5 and session_state.get("injury_history", "").strip():
        alerts.append({
            "level": "warning",
            "message": "⚠️ Asimmetria significativa + storia infortuni. Rischio aumentato di recidiva."
        })
    
    # Multiple red flags
    if len(session_state.get("red_flags", [])) >= 2:
        alerts.append({
            "level": "danger",
            "message": "🚨 MULTIPLE RED FLAGS PRESENTI. Riferimento medico urgente raccomandato."
        })
    elif len(session_state.get("red_flags", [])) == 1:
        alerts.append({
            "level": "warning",
            "message": f"⚠️ Red flag presente: {session_state['red_flags'][0]}. Valutare riferimento medico."
        })
    
    # High pain + low function
    if session_state.get("nprs", 0) >= 7:
        avg_psfs = np.mean([a["score"] for a in session_state.get("psfs_activities", [])])
        if avg_psfs < 5:
            alerts.append({
                "level": "warning",
                "message": "⚠️ Alto dolore (NPRS≥7) + bassa funzione (PSFS<5). Gestione dolore prioritaria."
            })
    
    # Multiple tests with pain in same region
    pain_by_region = df[df["Dolore"] == True].groupby("Regione").size()
    for region, count in pain_by_region.items():
        if count >= 2:
            alerts.append({
                "level": "info",
                "message": f"ℹ️ Dolore in {count} test per regione {region}. Approfondire valutazione locale."
            })
    
    return alerts

def generate_recommendations(df, sport, session_state):
    """Generate evidence-based recommendations"""
    recommendations = []
    
    # Priority 1: Critical scores (<4)
    critical = df[df["Score"] < 4].copy()
    if not critical.empty:
        for _, row in critical.iterrows():
            test_name = row["Test"]
            region = row["Regione"]
            
            # Find appropriate protocol
            protocol = None
            if "ankle" in region:
                protocol = EXERCISE_PROTOCOLS["ankle_mobility"]
            elif "hip" in region and "rotation" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["hip_rotation"]
            elif "hip" in region and "flexion" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["hip_flexion"]
            elif "hip" in region and "thomas" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["hip_flexors"]
            elif "thoracic" in region:
                protocol = EXERCISE_PROTOCOLS["thoracic_mobility"]
            elif "shoulder" in region:
                protocol = EXERCISE_PROTOCOLS["shoulder_mobility"]
            elif "knee" in region or "hamstring" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["hamstring_length"]
            elif "lumbar" in region:
                protocol = EXERCISE_PROTOCOLS["core_endurance"]
            elif "neural" in test_name.lower() or "ulnt" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["neural_mobility"]
            
            rec = {
                "priority": "🚨 ALTA",
                "test": test_name,
                "score": row["Score"],
                "action": f"Intervento immediato su {region}",
                "detail": "Evitare carichi massimali (>85% 1RM) fino a miglioramento",
                "timeline": "2-4 settimane",
                "protocol": protocol
            }
            recommendations.append(rec)
    
    # Priority 2: Moderate scores (4-7)
    moderate = df[(df["Score"] >= 4) & (df["Score"] < 7)].copy()
    if not moderate.empty:
        for _, row in moderate.iterrows():
            test_name = row["Test"]
            region = row["Regione"]
            
            protocol = None
            if "ankle" in region:
                protocol = EXERCISE_PROTOCOLS["ankle_mobility"]
            elif "hip" in region:
                if "rotation" in test_name.lower():
                    protocol = EXERCISE_PROTOCOLS["hip_rotation"]
                elif "flexion" in test_name.lower():
                    protocol = EXERCISE_PROTOCOLS["hip_flexion"]
                elif "thomas" in test_name.lower():
                    protocol = EXERCISE_PROTOCOLS["hip_flexors"]
            elif "thoracic" in region:
                protocol = EXERCISE_PROTOCOLS["thoracic_mobility"]
            elif "shoulder" in region:
                protocol = EXERCISE_PROTOCOLS["shoulder_mobility"]
            elif "knee" in region or "hamstring" in test_name.lower():
                protocol = EXERCISE_PROTOCOLS["hamstring_length"]
            elif "lumbar" in region:
                protocol = EXERCISE_PROTOCOLS["core_endurance"]
            
            rec = {
                "priority": "⚠️ MEDIA",
                "test": test_name,
                "score": row["Score"],
                "action": f"Lavoro mirato su {region}",
                "detail": "Continuare allenamento con focus su mobilità/rinforzo specifico",
                "timeline": "4-8 settimane",
                "protocol": protocol
            }
            recommendations.append(rec)
    
    # Priority 3: Asymmetries
    asymmetries = df[df["SymScore"] < 6].copy()
    if not asymmetries.empty:
        for _, row in asymmetries.iterrows():
            rec = {
                "priority": "⚠️ ASIMMETRIA",
                "test": row["Test"],
                "score": row["SymScore"],
                "action": f"Correzione asimmetria {row['Regione']}",
                "detail": f"Lavoro unilaterale, pattern correction. Delta: {row['Delta']}{row['Unità']}",
                "timeline": "3-6 settimane",
                "protocol": None
            }
            recommendations.append(rec)
    
    # Sport-specific critical tests
    if sport in SPORT_SPECIFIC_INTERPRETATION:
        for lift, data in SPORT_SPECIFIC_INTERPRETATION[sport].items():
            critical_tests = data.get("critical_tests", [])
            threshold = data.get("threshold", 7.0)
            
            for test in critical_tests:
                test_data = df[df["Test"] == test]
                if not test_data.empty:
                    score = test_data.iloc[0]["Score"]
                    if score < threshold:
                        rec = {
                            "priority": f"🎯 SPORT-SPECIFIC ({sport})",
                            "test": test,
                            "score": score,
                            "action": f"Test critico per {lift}",
                            "detail": data.get("note", ""),
                            "timeline": "Priorità alta",
                            "protocol": None
                        }
                        recommendations.append(rec)
    
    return recommendations

# -----------------------------
# Toggle callback
# -----------------------------
def toggle_info(session_key: str):
    st.session_state[session_key] = not st.session_state.get(session_key, False)

# -----------------------------
# Rendering functions
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
        with st.expander(f"📍 {region.capitalize()}", expanded=False):
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
                
                if name == "Thomas Test (modified)":
                    method_key = f"{key}_method"
                    current_method = rec.get("input_method", "degrees")
                    method = st.selectbox("Metodo input", options=["degrees", "cm"], 
                                         index=0 if current_method == "degrees" else 1, key=method_key)
                    rec["input_method"] = method
                    
                    if method == "cm":
                        max_cm = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 20.0
                        val_cm = st.slider("Distanza coscia‑tavolo (cm)", 0.0, max_cm, 
                                          float(rec.get("Val_cm", 0.0)), 0.1, key=f"{key}_Val_cm")
                        rec["Val_cm"] = val_cm
                        ref_cm = 10.0
                        deg = float(val_cm) * (rec.get("ref", ref) / ref_cm)
                        rec["Val"] = deg
                    else:
                        max_deg = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 30.0
                        val_deg = st.slider("Angolo estensione (°)", 0.0, max_deg, 
                                           float(rec.get("Val", 0.0)), 0.5, key=f"{key}_Val_deg")
                        rec["Val"] = val_deg
                    
                    sc = ability_linear(rec["Val"], rec.get("ref", ref), rec.get("higher_is_better", hib))
                    st.caption(f"Score (calcolato su gradi): **{sc:.1f}/10**")
                    
                    # Validation
                    warnings = validate_input(name, rec["Val"])
                    for w in warnings:
                        st.warning(w)
                
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
                        
                        # Validation
                        warnings_dx = validate_input(name, dx, "Dx")
                        warnings_sx = validate_input(name, sx, "Sx")
                        for w in warnings_dx + warnings_sx:
                            st.warning(w)
                    
                    else:
                        val = st.slider(f"Valore ({unit})", 0.0, max_val, float(rec.get("Val", 0.0)), 0.1, key=f"{key}_Val")
                        p = st.checkbox("Dolore", value=bool(rec.get("Dolore", False)), key=f"{key}_p")
                        rec.update({"Val": val, "Dolore": p})
                        sc = ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib))
                        st.caption(f"Score: **{sc:.1f}/10**")
                        
                        # Validation
                        warnings = validate_input(name, val)
                        for w in warnings:
                            st.warning(w)
                
                st.markdown("</div>", unsafe_allow_html=True)

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
                rows.append([
                    sec, name, unit, rec.get("ref", ref), f"{avg:.1f}", sc,
                    round(dx, 2), round(sx, 2), delta, sym, dolore_any, region,
                    dolore_dx, dolore_sx
                ])
            else:
                if name == "Thomas Test (modified)":
                    display_val = rec.get("Val_cm") if rec.get("input_method") == "cm" else rec.get("Val", 0.0)
                    val_for_score = rec.get("Val", 0.0)
                    sc = round(ability_linear(val_for_score, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                    dolore = bool(rec.get("Dolore", False))
                    rows.append([sec, name, unit, rec.get("ref", ref), f"{display_val:.1f}", sc, 
                                "", "", "", "", dolore, region, False, False])
                else:
                    val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce")
                    val = 0.0 if pd.isna(val) else float(val)
                    sc = round(ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib)), 2)
                    dolore = bool(rec.get("Dolore", False))
                    rows.append([sec, name, unit, rec.get("ref", ref), f"{val:.1f}", sc, 
                                "", "", "", "", dolore, region, False, False])

    df = pd.DataFrame(rows, columns=[
        "Sezione", "Test", "Unità", "Rif", "Valore", "Score",
        "Dx", "Sx", "Delta", "SymScore", "Dolore", "Regione",
        "DoloreDx", "DoloreSx"
    ])
    
    for col in ["Score", "Dx", "Sx", "Delta", "SymScore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

# -----------------------------
# Visualization functions
# -----------------------------
def radar_plot_matplotlib(df, title="Punteggi (0–10)"):
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

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
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
    ax.scatter(node_angles, values[:-1], c=node_colors, s=100, zorder=5, edgecolors="k", linewidths=2)

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    ax.set_title(sanitize_text_for_plot(title), y=1.08, fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.7)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

SIMPLE_TEST_LABELS = {
    "Weight Bearing Lunge Test": "Caviglia",
    "Passive Hip Flexion": "Flessione anca",
    "Hip Rotation (flexed 90°)": "Rotazione anca",
    "Wall Angel Test": "Mobilità toracica",
    "Shoulder ER (adducted, low-bar)": "ER Spalla",
    "Shoulder Flexion (supine)": "Flessione spalla",
    "External Rotation (90° abd)": "ER 90°",
    "Pectoralis Minor Length": "Pettorale Min",
    "Thomas Test (modified)": "Thomas",
    "Active Knee Extension (AKE)": "AKE",
    "Straight Leg Raise (SLR)": "SLR",
    "Sorensen Endurance": "Sorensen",
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

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, scores, color=colors_map, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("SymScore (0–10)", fontsize=12, fontweight="bold")
    ax.set_title(sanitize_text_for_plot(title), fontsize=14, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height() / 2, 
               f"{width:.1f}", va="center", fontweight="bold")

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
    
    df_r["ShortLabel"] = df_r["Test"].apply(lambda x: SHORT_RADAR_LABELS.get(x, x))
    
    fig = px.line_polar(df_r, r="Score", theta="ShortLabel", line_close=True, 
                        template="plotly_white", color_discrete_sequence=[PRIMARY])
    fig.update_traces(fill="toself", marker=dict(size=8))
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40), 
        polar=dict(radialaxis=dict(range=[0, 10], showticklabels=True, tickfont=dict(size=10))),
        font=dict(size=11)
    )
    return fig

@st.cache_data
def plotly_asymmetry(df):
    df_bilat = df[df["SymScore"].notnull()].copy()
    if df_bilat.empty:
        return None
    df_bilat["SymScore"] = pd.to_numeric(df_bilat["SymScore"], errors="coerce")
    df_bilat["ShortLabel"] = df_bilat["Test"].apply(lambda x: SIMPLE_TEST_LABELS.get(x, x))
    
    fig = px.bar(df_bilat, x="SymScore", y="ShortLabel", orientation="h", 
                 template="plotly_white", color="SymScore", 
                 color_continuous_scale=["#DC2626", "#F59E0B", "#16A34A"], 
                 range_x=[0, 10])
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    fig.update_xaxes(title="Symmetry Score")
    fig.update_yaxes(title="")
    return fig

def plot_progress_over_time(history):
    """Plot athlete progress over multiple assessments"""
    if len(history) < 2:
        return None
    
    dates = []
    avg_scores = []
    pain_counts = []
    
    for assessment in reversed(history):  # chronological order
        dates.append(assessment["date"])
        
        # Calculate average score
        vals = assessment.get("data", {})
        scores = []
        pains = 0
        
        for test_name, test_data in vals.items():
            ref = test_data.get("ref", 10.0)
            hib = test_data.get("higher_is_better", True)
            
            if test_data.get("bilat", False):
                dx = test_data.get("Dx", 0.0)
                sx = test_data.get("Sx", 0.0)
                avg = (float(dx) + float(sx)) / 2.0
                score = ability_linear(avg, ref, hib)
                scores.append(score)
                if test_data.get("DoloreDx") or test_data.get("DoloreSx"):
                    pains += 1
            else:
                val = test_data.get("Val", 0.0)
                score = ability_linear(val, ref, hib)
                scores.append(score)
                if test_data.get("Dolore"):
                    pains += 1
        
        avg_scores.append(np.mean(scores) if scores else 0)
        pain_counts.append(pains)
    
    # Convert dates to datetime
    dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Average scores
    ax1.plot(dates_dt, avg_scores, marker='o', linewidth=2, markersize=8, color=PRIMARY)
    ax1.set_ylabel("Score Medio (0-10)", fontsize=12, fontweight="bold")
    ax1.set_title("Progressione Score nel Tempo", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 10)
    ax1.axhline(y=7, color='green', linestyle='--', alpha=0.5, label='Target (7)')
    ax1.legend()
    
    # Plot 2: Pain counts
    ax2.bar(dates_dt, pain_counts, color='#DC2626', alpha=0.7, edgecolor='black')
    ax2.set_ylabel("N° Test con Dolore", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Data Valutazione", fontsize=12, fontweight="bold")
    ax2.set_title("Evoluzione Dolore", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf

# -----------------------------
# PDF Generation
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

def pdf_report_clinico(logo_bytes, athlete, evaluator, date_str, section, df, 
                       recommendations, session_state, 
                       radar_buf=None, asym_buf=None, progress_buf=None):
    """Generate comprehensive clinical PDF report"""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, 
                           topMargin=1.6 * cm, bottomMargin=2.8 * cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title_style = styles["Title"]
    heading = ParagraphStyle("heading", parent=styles["Heading2"], alignment=TA_LEFT, 
                            textColor=colors.HexColor(PRIMARY), fontSize=14, spaceAfter=12)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=10)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=6)

    story = []
    
    # Header
    header_table = Table([
        [
            RLImage(io.BytesIO(logo_bytes), width=4.0 * cm, height=1.0 * cm),
            Paragraph(f"<b>Report Valutazione Completo</b><br/>{sanitize_text_for_plot(section)}", title_style),
            Paragraph(f"<b>Atleta:</b> {athlete}<br/><b>Valutatore:</b> {evaluator}<br/><b>Data:</b> {date_str}", small),
        ]
    ], colWidths=[4.2 * cm, 8.8 * cm, 4.0 * cm])
    header_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(header_table)
    story.append(Spacer(1, 12))

    # Red Flags Section (if any)
    if session_state.get("red_flags"):
        story.append(Paragraph("<b>⚠️ RED FLAGS IDENTIFICATE</b>", heading))
        for flag in session_state["red_flags"]:
            story.append(Paragraph(f"• {RED_FLAGS.get(flag, flag)}", body))
        story.append(Paragraph("<b>AZIONE: Riferimento medico raccomandato prima di procedere.</b>", 
                              ParagraphStyle("alert", parent=body, textColor=colors.red)))
        story.append(Spacer(1, 12))

    # Anamnesis
    story.append(Paragraph("<b>Anamnesi</b>", heading))
    anamnesis_data = [
        ["Sport/Attività:", session_state.get("sport", "N/A")],
        ["Frequenza allenamento:", f"{session_state.get('training_frequency', 0)} giorni/settimana"],
        ["Storia infortuni:", session_state.get("injury_history", "Nessuna") or "Nessuna"],
        ["Sintomi attuali:", session_state.get("current_symptoms", "Nessuno") or "Nessuno"],
        ["Obiettivi:", session_state.get("goals", "N/A") or "N/A"],
    ]
    anamnesis_table = Table(anamnesis_data, colWidths=[4 * cm, 12 * cm])
    anamnesis_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
    ]))
    story.append(anamnesis_table)
    story.append(Spacer(1, 12))

    # Functional Scales
    story.append(Paragraph("<b>Scale Funzionali</b>", heading))
    nprs = session_state.get("nprs", 0)
    story.append(Paragraph(f"<b>NPRS</b> (dolore medio ultima settimana): {nprs}/10", body))
    
    psfs_activities = session_state.get("psfs_activities", [])
    if psfs_activities:
        story.append(Paragraph("<b>PSFS</b> (Patient-Specific Functional Scale):", body))
        for act in psfs_activities:
            story.append(Paragraph(f"• {act['activity']}: {act['score']}/10", body))
    story.append(Spacer(1, 12))

    # Metrics Summary
    avg_score = df["Score"].mean() if "Score" in df.columns and not df["Score"].isna().all() else 0.0
    n_dolore = int(df["Dolore"].sum()) if "Dolore" in df.columns else 0
    sym_mean = df["SymScore"].mean() if "SymScore" in df.columns else np.nan
    
    story.append(Paragraph("<b>Sintesi Metriche</b>", heading))
    metrics_table = Table([[
        Paragraph("<b>Score medio</b>", small), Paragraph(f"{avg_score:.1f}/10", small),
        Paragraph("<b>Test con dolore</b>", small), Paragraph(str(n_dolore), small),
        Paragraph("<b>Symmetry medio</b>", small), Paragraph(f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a", small)
    ]], colWidths=[2.4*cm, 2.0*cm, 2.8*cm, 1.8*cm, 3.0*cm, 2.2*cm])
    metrics_table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(PRIMARY)),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F0F4FF"))
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # Results table
    story.append(Paragraph("<b>Risultati Test Oggettivi</b>", heading))
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
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
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
    story.append(Spacer(1, 16))

    # Charts
    if radar_buf or asym_buf:
        story.append(Paragraph("<b>Visualizzazioni</b>", heading))
        
        chart_elements = []
        if radar_buf:
            chart_elements.append(RLImage(io.BytesIO(radar_buf.getvalue()), 
                                         width=9 * cm, height=9 * cm, hAlign="CENTER"))
        if asym_buf:
            chart_elements.append(Spacer(1, 8))
            chart_elements.append(RLImage(io.BytesIO(asym_buf.getvalue()), 
                                         width=14 * cm, height=6 * cm, hAlign="CENTER"))
        
        story.append(KeepTogether(chart_elements))
        story.append(Spacer(1, 16))

    # Progress over time (if available)
    if progress_buf:
        story.append(PageBreak())
        story.append(Paragraph("<b>Progressione Temporale</b>", heading))
        story.append(RLImage(io.BytesIO(progress_buf.getvalue()), 
                            width=16 * cm, height=10 * cm, hAlign="CENTER"))
        story.append(Spacer(1, 16))

    # Recommendations
    story.append(PageBreak())
    story.append(Paragraph("<b>Raccomandazioni Cliniche</b>", heading))
    
    if recommendations:
        for i, rec in enumerate(recommendations[:6], 1):  # Limit to top 6 for space
            rec_text = f"""
            <b>{rec['priority']}</b> — {rec['test']} (Score: {rec['score']:.1f}/10)<br/>
            <b>Azione:</b> {rec['action']}<br/>
            <b>Dettaglio:</b> {rec['detail']}<br/>
            <b>Timeline:</b> {rec['timeline']}
            """
            story.append(Paragraph(rec_text, body))
            
            # Add exercise protocol if available
            if rec.get('protocol'):
                protocol = rec['protocol']
                story.append(Paragraph(f"<b>Protocollo suggerito:</b> {protocol['nome']}", body))
                story.append(Paragraph(f"<i>Frequenza: {protocol['frequenza']} — Durata: {protocol['durata']}</i>", small))
                
                ex_list = "<br/>".join([f"  • {ex}" for ex in protocol['esercizi'][:3]])  # First 3 exercises
                story.append(Paragraph(ex_list, small))
            
            story.append(Spacer(1, 10))
    else:
        story.append(Paragraph("Nessuna raccomandazione critica. Continuare monitoraggio.", body))
    
    story.append(Spacer(1, 16))

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
    
    story.append(Paragraph("<b>Regioni Dolorose Rilevate</b>", heading))
    if pain_regions:
        for pr in pain_regions:
            story.append(Paragraph(f"• {pr.capitalize()}", body))
    else:
        story.append(Paragraph("Nessuna regione dolorosa segnalata.", body))
    story.append(Spacer(1, 16))

    # Clinical notes
    if session_state.get("clinical_notes", "").strip():
        story.append(Paragraph("<b>Note Cliniche Aggiuntive</b>", heading))
        story.append(Paragraph(session_state["clinical_notes"], body))
        story.append(Spacer(1, 12))

    if session_state.get("postural_observations", "").strip():
        story.append(Paragraph("<b>Osservazioni Posturali</b>", heading))
        story.append(Paragraph(session_state["postural_observations"], body))
        story.append(Spacer(1, 12))

    # Signature
    story.append(Spacer(1, 20))
    story.append(Paragraph("Firma Fisioterapista: ______________________", small))
    story.append(Paragraph(f"Data: {date_str}", small))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buf.seek(0)
    return buf

# -----------------------------
# Main UI
# -----------------------------
st.markdown(f"""
<style>
:root {{ --primary: {PRIMARY}; }}
body {{ background: #f6f8fb; }}
.header-card {{ 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 20px; 
    border-radius: 15px; 
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}}
.card {{ 
    background: white; 
    padding: 15px; 
    border-radius: 12px; 
    margin-bottom: 12px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    border-left: 4px solid {PRIMARY};
}}
.small-muted {{ color: #6b7280; font-size: 0.9rem; }}
.metric-card {{
    background: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 12px 24px;
    background-color: #f3f4f6;
    border-radius: 8px;
}}
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY};
    color: white;
}}
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image(LOGO, width=120)
with col2:
    st.markdown(f"""
    <div class='header-card'>
        <h1 style='margin:0; color: white;'>{APP_TITLE}</h1>
        <p style='margin:5px 0 0 0; color: rgba(255,255,255,0.9);'>{SUBTITLE}</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.write("")  # Spacer

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Dati Paziente")
    
    # Load existing athlete option
    existing_athletes = get_all_athletes()
    if existing_athletes:
        load_existing = st.checkbox("Carica atleta esistente")
        if load_existing:
            selected_athlete = st.selectbox("Seleziona atleta", existing_athletes)
            st.session_state["athlete"] = selected_athlete
    
    st.session_state["athlete"] = st.text_input("Nome Atleta", st.session_state["athlete"])
    st.session_state["evaluator"] = st.text_input("Fisioterapista", st.session_state["evaluator"])
    st.session_state["date"] = st.date_input("Data Valutazione", 
                                             datetime.strptime(st.session_state["date"], "%Y-%m-%d")).strftime("%Y-%m-%d")
    
    st.markdown("---")
    st.markdown("## ⚙️ Azioni")
    
    col_reset, col_random = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state["vals"].clear()
            seed_defaults()
            st.rerun()
    
    with col_random:
        if st.button("🎲 Random", use_container_width=True):
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
            st.success("✓ Valori randomizzati")
            st.rerun()
    
    st.markdown("---")
    
    # Save assessment button
    if st.button("💾 Salva Valutazione", use_container_width=True, type="primary"):
        assessment_data = {
            "athlete": st.session_state["athlete"],
            "evaluator": st.session_state["evaluator"],
            "date": st.session_state["date"],
            "sport": st.session_state.get("sport", ""),
            "training_frequency": st.session_state.get("training_frequency", 0),
            "injury_history": st.session_state.get("injury_history", ""),
            "current_symptoms": st.session_state.get("current_symptoms", ""),
            "goals": st.session_state.get("goals", ""),
            "red_flags": st.session_state.get("red_flags", []),
            "nprs": st.session_state.get("nprs", 0),
            "psfs_activities": st.session_state.get("psfs_activities", []),
            "pain_behavior": st.session_state.get("pain_behavior", []),
            "aggravating_factors": st.session_state.get("aggravating_factors", []),
            "relieving_factors": st.session_state.get("relieving_factors", []),
            "movement_quality": st.session_state.get("movement_quality", {}),
            "clinical_notes": st.session_state.get("clinical_notes", ""),
            "postural_observations": st.session_state.get("postural_observations", ""),
            "data": st.session_state["vals"].copy()
        }
        
        try:
            filepath = save_assessment_to_file(assessment_data)
            st.success(f"✓ Valutazione salvata: {filepath.name}")
        except Exception as e:
            st.error(f"Errore nel salvataggio: {e}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Anamnesi", 
    "🔍 Test Oggettivi", 
    "📊 Risultati", 
    "💪 Qualità Movimento",
    "📈 Progressione",
    "📄 Report PDF"
])

# TAB 1: ANAMNESIS
with tab1:
    st.markdown("### 🩺 Anamnesi e Screening")
    
    col_anamnesi1, col_anamnesi2 = st.columns(2)
    
    with col_anamnesi1:
        st.markdown("#### Informazioni Generali")
        st.session_state["sport"] = st.selectbox("Sport/Attività principale", SPORTS_LIST, 
                                                  index=SPORTS_LIST.index(st.session_state.get("sport", "Powerlifting")))
        st.session_state["training_frequency"] = st.number_input("Frequenza allenamento (giorni/settimana)", 
                                                                 min_value=0, max_value=7, 
                                                                 value=st.session_state.get("training_frequency", 4))
        
        st.markdown("#### Storia Clinica")
        st.session_state["injury_history"] = st.text_area("Storia infortuni precedenti", 
                                                          value=st.session_state.get("injury_history", ""),
                                                          height=100,
                                                          help="Elencare infortuni significativi, interventi chirurgici, etc.")
        
        st.session_state["current_symptoms"] = st.text_area("Sintomi attuali", 
                                                            value=st.session_state.get("current_symptoms", ""),
                                                            height=100,
                                                            help="Descrizione sintomi che hanno portato alla valutazione")
        
        st.session_state["goals"] = st.text_area("Obiettivi della valutazione", 
                                                 value=st.session_state.get("goals", ""),
                                                 height=80,
                                                 help="Es: ritorno allo sport, preparazione gara, screening preventivo")
    
    with col_anamnesi2:
        st.markdown("#### 🚨 Red Flags Screening")
        st.caption("Selezionare se presenti (richiede attenzione medica):")
        
        selected_flags = []
        for flag_key, flag_desc in RED_FLAGS.items():
            if st.checkbox(flag_desc, key=f"redflag_{flag_key}", 
                          value=flag_key in st.session_state.get("red_flags", [])):
                selected_flags.append(flag_key)
        
        st.session_state["red_flags"] = selected_flags
        
        if selected_flags:
            st.error(f"⚠️ {len(selected_flags)} Red Flag(s) identificate! Riferimento medico raccomandato.")
        else:
            st.success("✓ Nessuna red flag identificata")
    
    st.markdown("---")
    
    # Functional Scales
    st.markdown("### 📏 Scale Funzionali")
    
    col_scale1, col_scale2 = st.columns([1, 2])
    
    with col_scale1:
        st.markdown("#### NPRS")
        st.caption("Numeric Pain Rating Scale")
        st.session_state["nprs"] = st.slider("Dolore medio ultima settimana", 
                                             0, 10, st.session_state.get("nprs", 0),
                                             help="0 = nessun dolore, 10 = peggior dolore immaginabile")
        
        if st.session_state["nprs"] >= 7:
            st.warning("⚠️ Dolore elevato - considerare gestione farmacologica")
        elif st.session_state["nprs"] >= 4:
            st.info("ℹ️ Dolore moderato")
        else:
            st.success("✓ Dolore minimo/assente")
    
    with col_scale2:
        st.markdown("#### PSFS")
        st.caption("Patient-Specific Functional Scale - Valutare 3 attività limitate dal problema")
        
        psfs_activities = st.session_state.get("psfs_activities", [
            {"activity": "Squat profondo", "score": 10},
            {"activity": "Corsa", "score": 10},
            {"activity": "Overhead press", "score": 10}
        ])
        
        for i in range(3):
            col_act, col_score = st.columns([2, 1])
            with col_act:
                psfs_activities[i]["activity"] = st.text_input(f"Attività {i+1}", 
                                                               value=psfs_activities[i]["activity"],
                                                               key=f"psfs_act_{i}")
            with col_score:
                psfs_activities[i]["score"] = st.slider(f"Capacità", 0, 10, 
                                                        psfs_activities[i]["score"],
                                                        key=f"psfs_score_{i}",
                                                        help="0=impossibile, 10=come prima del problema")
        
        st.session_state["psfs_activities"] = psfs_activities
        
        avg_psfs = np.mean([a["score"] for a in psfs_activities])
        st.metric("PSFS Medio", f"{avg_psfs:.1f}/10")
        
        if avg_psfs < 5:
            st.warning("⚠️ Limitazione funzionale significativa")
    
    st.markdown("---")
    
    # Pain characterization
    st.markdown("### 🎯 Caratterizzazione del Dolore")
    
    col_pain1, col_pain2, col_pain3 = st.columns(3)
    
    with col_pain1:
        st.markdown("#### Comportamento")
        pain_behavior = st.multiselect("Tipo di dolore", PAIN_BEHAVIORS,
                                       default=st.session_state.get("pain_behavior", []))
        st.session_state["pain_behavior"] = pain_behavior
    
    with col_pain2:
        st.markdown("#### Fattori Aggravanti")
        aggravating = st.multiselect("Cosa peggiora", AGGRAVATING_FACTORS,
                                     default=st.session_state.get("aggravating_factors", []))
        st.session_state["aggravating_factors"] = aggravating
    
    with col_pain3:
        st.markdown("#### Fattori Allevianti")
        relieving = st.multiselect("Cosa migliora", RELIEVING_FACTORS,
                                   default=st.session_state.get("relieving_factors", []))
        st.session_state["relieving_factors"] = relieving

# TAB 2: OBJECTIVE TESTS
with tab2:
    st.markdown("### 🔬 Test Oggettivi")
    section = "Valutazione Generale"
    render_inputs_for_section(section)

# TAB 3: RESULTS
with tab3:
    st.markdown("### 📊 Analisi Risultati")
    
    df_show = build_df("Valutazione Generale")
    
    if df_show.empty:
        st.warning("⚠️ Nessun dato disponibile. Compilare i test nella sezione 'Test Oggettivi'.")
    else:
        # Summary metrics
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        avg_score = df_show["Score"].mean() if not df_show["Score"].isna().all() else 0.0
        painful = int(df_show["Dolore"].sum()) if "Dolore" in df_show.columns else 0
        sym_mean = df_show["SymScore"].mean() if "SymScore" in df_show.columns else np.nan
        critical_count = len(df_show[df_show["Score"] < 4])
        
        with col_metric1:
            st.metric("Score Medio", f"{avg_score:.1f}/10",
                     delta="Buono" if avg_score >= 7 else "Da migliorare",
                     delta_color="normal" if avg_score >= 7 else "inverse")
        
        with col_metric2:
            st.metric("Test con Dolore", f"{painful}",
                     delta="OK" if painful == 0 else "Attenzione",
                     delta_color="normal" if painful == 0 else "inverse")
        
        with col_metric3:
            st.metric("Symmetry Medio", 
                     f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a",
                     delta="Simmetrico" if sym_mean >= 7 else "Asimmetrico" if not pd.isna(sym_mean) else "",
                     delta_color="normal" if sym_mean >= 7 else "inverse")
        
        with col_metric4:
            st.metric("Test Critici", f"{critical_count}",
                     delta="Intervento necessario" if critical_count > 0 else "Tutto OK",
                     delta_color="inverse" if critical_count > 0 else "normal")
        
        st.markdown("---")
        
        # Risk factors and alerts
        alerts = check_risk_factors(df_show, st.session_state)
        if alerts:
            st.markdown("#### ⚠️ Alert Clinici")
            for alert in alerts:
                if alert["level"] == "danger":
                    st.error(alert["message"])
                elif alert["level"] == "warning":
                    st.warning(alert["message"])
                else:
                    st.info(alert["message"])
            st.markdown("---")
        
        # Results table
        st.markdown("#### 📋 Tabella Risultati Completa")
        
        def status_icon(score):
            try:
                s = float(score)
                if s >= 7:
                    return "✔️"
                elif s >= 4:
                    return "⚠️"
                else:
                    return "❌"
            except Exception:
                return ""
        
        df_display = df_show.copy()
        df_display["Stato"] = df_display["Score"].apply(status_icon)
        
        # Color coding function
        def color_score(val):
            try:
                v = float(val)
                if v >= 7:
                    color = '#d4edda'
                elif v >= 4:
                    color = '#fff3cd'
                else:
                    color = '#f8d7da'
                return f'background-color: {color}'
            except:
                return ''
        
        cols_order = ["Stato", "Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]
        styled_df = df_display[cols_order].style.applymap(color_score, subset=['Score', 'SymScore']).format(precision=1)
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Visualizations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("#### 🎯 Radar Chart - Punteggi")
            radar_fig = plotly_radar(df_show)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Servono almeno 3 test per il radar chart.")
        
        with col_viz2:
            st.markdown("#### ⚖️ Asimmetrie Dx/Sx")
            asym_fig = plotly_asymmetry(df_show)
            if asym_fig:
                st.plotly_chart(asym_fig, use_container_width=True)
            else:
                st.info("Nessuna asimmetria da visualizzare.")
        
        st.markdown("---")
        
        # Recommendations
        st.markdown("### 💡 Raccomandazioni Evidence-Based")
        
        recommendations = generate_recommendations(df_show, st.session_state.get("sport", "Powerlifting"), st.session_state)
        
        if recommendations:
            for rec in recommendations:
                with st.expander(f"{rec['priority']} — {rec['test']} (Score: {rec['score']:.1f}/10)", expanded=False):
                    st.markdown(f"**Azione:** {rec['action']}")
                    st.markdown(f"**Dettaglio:** {rec['detail']}")
                    st.markdown(f"**Timeline:** {rec['timeline']}")
                    
                    if rec.get('protocol'):
                        protocol = rec['protocol']
                        st.markdown(f"**Protocollo suggerito:** {protocol['nome']}")
                        st.markdown(f"*Frequenza: {protocol['frequenza']} — Durata: {protocol['durata']}*")
                        st.markdown("**Esercizi:**")
                        for ex in protocol['esercizi']:
                            st.markdown(f"- {ex}")
                        if 'progressione' in protocol:
                            st.markdown(f"*Progressione: {protocol['progressione']}*")
        else:
            st.success("✓ Nessuna raccomandazione critica. Continuare monitoraggio regolare.")

# TAB 4: MOVEMENT QUALITY
with tab4:
    st.markdown("### 🏃 Valutazione Qualitativa del Movimento")
    st.caption("Valutazione pattern di movimento e controllo motorio")
    
    movement_quality = st.session_state.get("movement_quality", {})
    
    for test_name, test_data in MOVEMENT_QUALITY_TESTS.items():
        with st.expander(f"📹 {test_name}", expanded=False):
            st.markdown(f"**Parametri da osservare:**")
            
            test_scores = movement_quality.get(test_name, {})
            
            for param in test_data["parametri"]:
                param_key = f"{test_name}_{param}"
                score = st.select_slider(
                    param,
                    options=test_data["scoring"],
                    value=test_scores.get(param, test_data["scoring"][0]),
                    key=param_key
                )
                test_scores[param] = score
            
            movement_quality[test_name] = test_scores
            
            # Overall assessment
            st.markdown("**Note aggiuntive:**")
            notes_key = f"{test_name}_notes"
            notes = st.text_area("Osservazioni", 
                                value=test_scores.get("notes", ""),
                                key=notes_key,
                                height=80)
            test_scores["notes"] = notes
    
    st.session_state["movement_quality"] = movement_quality
    
    st.markdown("---")
    
    # Clinical notes
    st.markdown("### 📝 Note Cliniche Generali")
    
    col_notes1, col_notes2 = st.columns(2)
    
    with col_notes1:
        st.markdown("#### Osservazioni Posturali")
        st.session_state["postural_observations"] = st.text_area(
            "Postura statica e dinamica",
            value=st.session_state.get("postural_observations", ""),
            height=150,
            help="Es: iperlordosi lombare, spalle anteposte, rotazione bacino, etc."
        )
    
    with col_notes2:
        st.markdown("#### Note Cliniche Libere")
        st.session_state["clinical_notes"] = st.text_area(
            "Altre osservazioni rilevanti",
            value=st.session_state.get("clinical_notes", ""),
            height=150,
            help="Pattern di movimento, compensi, strategie motorie, etc."
        )

# TAB 5: PROGRESSION (CORRECTED)
with tab5:
    st.markdown("### 📈 Progressione nel Tempo")
    
    athlete_name = st.session_state.get("athlete", "")
    
    if athlete_name:
        history = load_athlete_history(athlete_name)
        
        if len(history) >= 2:
            st.success(f"✓ Trovate {len(history)} valutazioni per {athlete_name}")
            
            # Display history table
            history_table_data = []
            for h in history:
                # Calculate avg score from history
                vals = h.get("data", {})
                scores = []
                for test_data in vals.values():
                    ref = test_data.get("ref", 10.0)
                    hib = test_data.get("higher_is_better", True)
                    if test_data.get("bilat", False):
                        dx = test_data.get("Dx", 0.0)
                        sx = test_data.get("Sx", 0.0)
                        avg = (float(dx) + float(sx)) / 2.0
                        score = ability_linear(avg, ref, hib)
                        scores.append(score)
                    else:
                        val = test_data.get("Val", 0.0)
                        score = ability_linear(val, ref, hib)
                        scores.append(score)
                
                avg_score = np.mean(scores) if scores else 0
                nprs = h.get("nprs", 0)
                
                history_table_data.append({
                    "Data": h["date"],
                    "Score Medio": f"{avg_score:.1f}",
                    "NPRS": nprs,
                    "Valutatore": h.get("evaluator", "N/A")
                })
            
            history_df = pd.DataFrame(history_table_data)
            st.dataframe(history_df, use_container_width=True)
            
            st.markdown("---")
            
            # Progress chart
            st.markdown("#### 📊 Grafico Progressione")
            progress_buf = plot_progress_over_time(history)
            
            if progress_buf:
                st.image(progress_buf.getvalue(), use_column_width=True)
            else:
                st.info("Impossibile generare grafico progressione.")
            
            st.markdown("---")
            
            # Comparison selector
            st.markdown("#### 🔍 Confronto tra Valutazioni")
            
            if len(history) >= 2:
                dates = [h["date"] for h in history]
                
                col_comp1, col_comp2 = st.columns(2)
                
                with col_comp1:
                    date1 = st.selectbox("Prima valutazione", dates, index=1)
                
                with col_comp2:
                    date2 = st.selectbox("Seconda valutazione", dates, index=0)
                
                if st.button("Confronta Valutazioni"):
                    # Find assessments
                    assess1 = next((h for h in history if h["date"] == date1), None)
                    assess2 = next((h for h in history if h["date"] == date2), None)
                    
                    if assess1 and assess2:
                        st.markdown(f"**Confronto: {date1} vs {date2}**")
                        
                        # Build comparison table
                        comparison_data = []
                        
                        for test_name in assess1.get("data", {}).keys():
                            if test_name in assess2.get("data", {}):
                                data1 = assess1["data"][test_name]
                                data2 = assess2["data"][test_name]
                                
                                ref = data1.get("ref", 10.0)
                                hib = data1.get("higher_is_better", True)
                                
                                if data1.get("bilat", False):
                                    dx1 = data1.get("Dx", 0.0)
                                    sx1 = data1.get("Sx", 0.0)
                                    avg1 = (float(dx1) + float(sx1)) / 2.0
                                    score1 = ability_linear(avg1, ref, hib)
                                    
                                    dx2 = data2.get("Dx", 0.0)
                                    sx2 = data2.get("Sx", 0.0)
                                    avg2 = (float(dx2) + float(sx2)) / 2.0
                                    score2 = ability_linear(avg2, ref, hib)
                                else:
                                    val1 = data1.get("Val", 0.0)
                                    score1 = ability_linear(val1, ref, hib)
                                    
                                    val2 = data2.get("Val", 0.0)
                                    score2 = ability_linear(val2, ref, hib)
                                
                                delta = score2 - score1
                                trend = "📈" if delta > 0.5 else ("📉" if delta < -0.5 else "➡️")
                                
                                comparison_data.append({
                                    "Test": test_name,
                                    date1: f"{score1:.1f}",
                                    date2: f"{score2:.1f}",
                                    "Δ": f"{delta:+.1f}",
                                    "Trend": trend
                                })
                        
                        comp_df = pd.DataFrame(comparison_data)
                        st.dataframe(comp_df, use_container_width=True)
        
        elif len(history) == 1:
            st.info(f"ℹ️ Trovata 1 valutazione per {athlete_name}. Servono almeno 2 valutazioni per visualizzare la progressione.")
            
            # CORREZIONE: Mostra i dati in modo formattato invece di JSON raw
            h = history[0]
            
            st.markdown("#### 📋 Dettagli Valutazione")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data", h.get("date", "N/A"))
            with col2:
                st.metric("Sport", h.get("sport", "N/A"))
            with col3:
                st.metric("NPRS", f"{h.get('nprs', 0)}/10")
            
            st.markdown("---")
            
            # Calcola score medio
            vals = h.get("data", {})
            scores = []
            for test_data in vals.values():
                ref = test_data.get("ref", 10.0)
                hib = test_data.get("higher_is_better", True)
                if test_data.get("bilat", False):
                    dx = test_data.get("Dx", 0.0)
                    sx = test_data.get("Sx", 0.0)
                    avg = (float(dx) + float(sx)) / 2.0
                    score = ability_linear(avg, ref, hib)
                    scores.append(score)
                else:
                    val = test_data.get("Val", 0.0)
                    score = ability_linear(val, ref, hib)
                    scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.metric("Score Medio", f"{avg_score:.1f}/10")
            with col_score2:
                st.metric("Numero Test", len(vals))
            
            st.markdown("---")
            
            # Mostra test con score più bassi
            st.markdown("#### 🎯 Test con Score Più Bassi")
            
            test_scores = []
            for test_name, test_data in vals.items():
                ref = test_data.get("ref", 10.0)
                hib = test_data.get("higher_is_better", True)
                
                if test_data.get("bilat", False):
                    dx = test_data.get("Dx", 0.0)
                    sx = test_data.get("Sx", 0.0)
                    avg = (float(dx) + float(sx)) / 2.0
                    score = ability_linear(avg, ref, hib)
                else:
                    val = test_data.get("Val", 0.0)
                    score = ability_linear(val, ref, hib)
                
                test_scores.append({
                    "Test": test_name,
                    "Score": score,
                    "Regione": test_data.get("region", "N/A")
                })
            
            # Ordina per score (dal più basso)
            test_scores_df = pd.DataFrame(test_scores).sort_values("Score")
            
            # Mostra top 5 con score più bassi
            st.dataframe(test_scores_df.head(5), use_container_width=True)
            
            st.markdown("---")
            
            # Anamnesi
            with st.expander("📋 Anamnesi e Note Cliniche", expanded=False):
                st.markdown(f"**Storia Infortuni:** {h.get('injury_history', 'Nessuna') or 'Nessuna'}")
                st.markdown(f"**Sintomi Attuali:** {h.get('current_symptoms', 'Nessuno') or 'Nessuno'}")
                st.markdown(f"**Obiettivi:** {h.get('goals', 'N/A') or 'N/A'}")
                st.markdown(f"**Note Cliniche:** {h.get('clinical_notes', 'Nessuna') or 'Nessuna'}")
                st.markdown(f"**Osservazioni Posturali:** {h.get('postural_observations', 'Nessuna') or 'Nessuna'}")
            
            # PSFS
            psfs_activities = h.get("psfs_activities", [])
            if psfs_activities:
                with st.expander("📊 PSFS Activities", expanded=False):
                    for act in psfs_activities:
                        st.markdown(f"- **{act.get('activity', 'N/A')}**: {act.get('score', 0)}/10")
            
            # Red Flags
            red_flags = h.get("red_flags", [])
            if red_flags:
                with st.expander("🚨 Red Flags", expanded=False):
                    for flag in red_flags:
                        st.warning(f"⚠️ {RED_FLAGS.get(flag, flag)}")
            
            st.success("💡 **Suggerimento:** Crea una nuova valutazione per vedere la progressione nel tempo!")
        
        else:
            st.warning(f"⚠️ Nessuna valutazione precedente trovata per {athlete_name}")
            st.info("💡 Compila i test e clicca su '💾 Salva Valutazione' nella sidebar per iniziare a tracciare i tuoi progressi.")
    
    else:
        st.warning("⚠️ Inserire il nome dell'atleta per visualizzare la progressione")

# TAB 6: PDF REPORT
with tab6:
    st.markdown("### 📄 Generazione Report PDF")
    
    df_show = build_df("Valutazione Generale")
    
    if df_show.empty:
        st.warning("⚠️ Nessun dato disponibile per generare il report. Compilare prima i test.")
    else:
        st.info("Il report PDF includerà: anamnesi, red flags, scale funzionali, risultati test, grafici, raccomandazioni e note cliniche.")
        
        # Prepare data for PDF
        try:
            df_radar = df_show[df_show["Score"].notnull()].copy()
            radar_buf = radar_plot_matplotlib(df_radar, title="Punteggi Test (0-10)") if len(df_radar) >= 3 else None
        except Exception:
            radar_buf = None
        
        try:
            asym_buf = asymmetry_plot_matplotlib(df_show, title="Simmetria Dx/Sx")
        except Exception:
            asym_buf = None
        
        # Progress chart if history available
        progress_buf = None
        athlete_name = st.session_state.get("athlete", "")
        if athlete_name:
            history = load_athlete_history(athlete_name)
            if len(history) >= 2:
                try:
                    progress_buf = plot_progress_over_time(history)
                except Exception:
                    pass
        
        recommendations = generate_recommendations(df_show, st.session_state.get("sport", "Powerlifting"), st.session_state)
        
        col_pdf1, col_pdf2 = st.columns([2, 1])
        
        with col_pdf1:
            if st.button("📥 Genera Report PDF Completo", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Generazione PDF in corso..."):
                        pdf = pdf_report_clinico(
                            logo_bytes=LOGO,
                            athlete=st.session_state["athlete"],
                            evaluator=st.session_state["evaluator"],
                            date_str=st.session_state["date"],
                            section="Valutazione Generale",
                            df=df_show,
                            recommendations=recommendations,
                            session_state=st.session_state,
                            radar_buf=radar_buf,
                            asym_buf=asym_buf,
                            progress_buf=progress_buf
                        )
                    
                    st.success("✓ PDF generato con successo!")
                    
                    st.download_button(
                        label="💾 Scarica Report PDF",
                        data=pdf.getvalue(),
                        file_name=f"Fisiomove_Report_{st.session_state['athlete'].replace(' ', '_')}_{st.session_state['date']}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                
                except Exception as e:
                    st.error(f"❌ Errore nella generazione del PDF: {e}")
        
        with col_pdf2:
            st.markdown("**Contenuto Report:**")
            st.markdown("""
            - ✅ Header professionale
            - ✅ Dati anamnestici
            - ✅ Red flags screening
            - ✅ Scale funzionali (NPRS, PSFS)
            - ✅ Risultati test oggettivi
            - ✅ Grafici (radar + asimmetrie)
            - ✅ Progressione temporale
            - ✅ Raccomandazioni EBM
            - ✅ Protocolli esercizi
            - ✅ Note cliniche
            - ✅ QR code
            """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6b7280; font-size: 0.85rem;'>
    <p><b>{APP_TITLE}</b> • {SUBTITLE}</p>
    <p>Contatto: {CONTACT}</p>
    <p style='font-size: 0.75rem; margin-top: 10px;'>
        Valutazioni salvate in: <code>{ASSESSMENTS_DIR.absolute()}</code>
    </p>
</div>
""", unsafe_allow_html=True)
