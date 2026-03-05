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

st.set_page_config(page_title="Fisiomove MobilityPassport", layout="wide", page_icon="🩺")

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
APP_TITLE = "Fisiomove MobilityPassport"
SUBTITLE = "Sistema Completo di Valutazione Fisioterapica — v1.0"
PRIMARY = "#1E6CF4"
CONTACT = "info@fisiomove.net"

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

# Bibliografia scientifica completa
BIBLIOGRAPHY = [
    "Backman LJ, Danielson P. Low range of ankle dorsiflexion predisposes for patellar tendinopathy in junior elite basketball players: a 1-year prospective study. Am J Sports Med. 2011;39(12):2626-33.",
    "Borstad JD, Ludewig PM. The effect of long versus short pectoralis minor resting length on scapular kinematics in healthy individuals. J Orthop Sports Phys Ther. 2005;35(4):227-38.",
    "Dill KE, Begalle RL, Frank BS, Zinder SM, Padua DA. Altered knee and ankle kinematics during squatting in those with limited weight-bearing-lunge ankle-dorsiflexion range of motion. J Athl Train. 2014;49(6):723-32.",
    "Freckleton G, Pizzari T. Risk factors for hamstring muscle strain injury in sport: a systematic review and meta-analysis. Br J Sports Med. 2013;47(6):351-8.",
    "Laudner KG, Stanek JM, Meister K. Differences in scapular upward rotation between baseball pitchers and position players. Am J Sports Med. 2011;39(12):2682-7.",
    "Li Y, McClure PW, Pratt N. The effect of hamstring muscle stretching on standing posture and on lumbar and hip motions during forward bending. Phys Ther. 2015;76(8):836-45.",
    "Liaghat B, Skou ST, Sondergaard J, Boyle E, Søgaard K, Juul-Kristensen B. Short-term effectiveness of high-load compared with low-load strengthening exercise on self-reported function in patients with hypermobile shoulders: a randomised controlled trial. Br J Sports Med. 2021;55(23):1364-1371.",
    "Liaghat B, Juul-Kristensen B, Frydendal T, Marie Larsen C, Søgaard K, Iliadis S. Competitive swimmers with hypermobility have strength and proprioception deficits but no increased risk of shoulder pain: A prospective case-control study. J Sci Med Sport. 2021;24(11):1145-1150.",
    "Luoto S, Heliövaara M, Hurri H, Alaranta H. Static back endurance and the risk of low-back pain. Clin Biomech. 1995;10(6):323-4.",
    "McGill SM, Childs A, Liebenson C. Endurance times for low back stabilization exercises: clinical targets for testing and training from a normal database. Arch Phys Med Rehabil. 1999;80(8):941-4.",
    "Mosler AB, Weir A, Eirale C, et al. Epidemiology of time loss groin injuries in a men's professional football league: a 2-year prospective study of 17 clubs and 606 players. Br J Sports Med. 2018;52(5):292-297.",
    "Nathan JA, Davies K, Swaine IL. Hypermobility and sports injuries in junior netball players. Br J Sports Med. 2018;52(8):1-2.",
    "Nee RJ, Jull GA, Vicenzino B, Coppieters MW. The validity of upper-limb neurodynamic tests for detecting peripheral neuropathic pain. J Orthop Sports Phys Ther. 2012;42(5):413-24.",
    "Pasanen K, Ekola T, Vasankari T, et al. High ankle dorsiflexion and decreased knee frontal plane projection angle as risk factors for anterior cruciate ligament injury: a prospective study. Orthop J Sports Med. 2025;13(1):23259671241310242.",
    "Reiman MP, Goode AP, Cook CE, Holmich P, Thorborg K. Diagnostic accuracy of clinical tests for the diagnosis of hip femoroacetabular impingement/labral tear: a systematic review with meta-analysis. Br J Sports Med. 2012;49(12):811.",
    "Sanderson M, Maxwell S. The effect of lumbar extension on hip flexor length in chronic low back pain. Physiotherapy. 2015;101:eS1356-7.",
    "Shanley E, Rauh MJ, Michener LA, Ellenbecker TS, Garrison JC, Thigpen CA. Shoulder range of motion measures as risk factors for shoulder and elbow injuries in high school softball and baseball players. Am J Sports Med. 2011;39(9):1997-2006.",
    "Tyler TF, Nicholas SJ, Roy T, Gleim GW. Quantification of posterior capsule tightness and motion loss in patients with shoulder impingement. Am J Sports Med. 2010;28(5):668-73."
]

# Tests definitions
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 12.0, True, "ankle", "Test dorsiflessione in carico.", True),
        ("Passive Hip Flexion", "°", 120.0, True, "hip", "Flessione anca passiva.", True),
        ("Hip Internal Rotation", "°", 35.0, True, "hip", "Rotazione interna anca (flessione 90°).", True),
        ("Hip External Rotation", "°", 45.0, True, "hip", "Rotazione esterna anca (flessione 90°).", True),
        ("Wall Angel Test", "cm", 0.0, True, "thoracic", "Distanza braccio da schienale: 0cm=100%, negativo=ipermobile, positivo=deficit.", True),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "Rotazione esterna spalla (low-bar).", True),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Flessione spalla (supina).", True),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "ER a 90° abduzione.", True),
        ("Wall Angel Test", "cm", 0.0, True, "thoracic", "Distanza braccio da schienale: 0cm=100%, negativo=ipermobile, positivo=deficit.", True),
        ("Pectoralis Minor Length", "cm", 5.0, True, "shoulder", "Distanza PM: 5cm=100%, <5cm=ipermobile, >5cm=rigido.", False),
        ("Thomas Test (modified)", "°", 10.0, True, "hip", "Thomas Test: 10°=100%, >10°=ipermobile, <10°=rigido.", True),
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

# Injury Risk Database (EBM-based)
INJURY_RISK_DATABASE = {
    "Weight Bearing Lunge Test": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Tendinopatia Achillea", "Fascite plantare", "Sindrome impingement anteriore caviglia"],
        "mechanism": "Dorsiflessione limitata aumenta stress su tendine Achille e fascia plantare, altera biomeccanica squat",
        "sport_specific": {"Squat": "Compenso con inclinazione tronco eccessiva, talloni sollevati, stress lombare"},
        "evidence": "Decrease in ankle DF associated with increased injury risk (Backman & Danielson 2011)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Hip Internal Rotation": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Conflitto femoro-acetabolare (FAI)", "Lesioni labrali", "Lombalgia da compenso"],
        "mechanism": "Deficit IR anca causa compenso in rotazione lombare durante squat, aumenta stress su labrum",
        "sport_specific": {"Squat": "Buttwink eccessivo, perdita profondità, valgo ginocchio"},
        "evidence": "Limited hip IR predicts hip/groin pain in athletes (Mosler et al. 2018, BJSM)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Hip External Rotation": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Sindrome piriforme", "Tendinopatia glutei", "Dolore trocanterico"],
        "mechanism": "Deficit ER limita stabilità bacino, sovraccarica rotatori esterni e abduttori anca",
        "sport_specific": {"Squat": "Collasso valgo ginocchia, Trendelenburg stance", "Deadlift": "Perdita setup, rotazione bacino"},
        "evidence": "Hip ER deficit correlates with lateral hip pain (Reiman et al. 2012)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Passive Hip Flexion": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Impingement anca", "Lombalgia", "Lesioni labrali"],
        "mechanism": "ROM flessione <110° limita profondità squat, causa compenso lombare (buttwink precoce)",
        "sport_specific": {"Squat": "Impossibilità raggiungere profondità, stress lombare"},
        "evidence": "Hip flexion ROM critical for deep squat mechanics (Dill et al. 2014)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Active Knee Extension (AKE)": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Lesioni hamstring", "Lombalgia", "Tendinopatia prossimale hamstring"],
        "mechanism": "Deficit AKE indica rigidità hamstring, aumenta rischio strain durante deadlift/sprint",
        "sport_specific": {"Deadlift": "Setup compromesso, flessione lombare eccessiva, rischio ernia discale"},
        "evidence": "Limited knee extension (<20° from full) increases hamstring injury risk 2.4x (Freckleton & Pizzari 2013)",
        "priority_critical": 3.5,
        "priority_high": 6.5
    },
    "Straight Leg Raise (SLR)": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Lesioni hamstring", "Neuropatia sciatica", "Lombalgia"],
        "mechanism": "SLR <70° indica tensione neurale o rigidità hamstring, limita pattern hip hinge",
        "sport_specific": {"Deadlift": "Incapacità mantenere schiena neutra in setup"},
        "evidence": "Poor hamstring flexibility predicts lower back pain (Li et al. 2015)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Sorensen Endurance": {
        "risk_threshold": 6.0,
        "risk_injuries": ["Lombalgia cronica", "Spondilolistesi", "Strain erector spinae"],
        "mechanism": "Deficit endurance estensori lombari (<120s) aumenta carico su strutture passive",
        "sport_specific": {"Deadlift": "Loss of lordosis sotto carico, rischio injury lombare"},
        "evidence": "Sorensen test <58s predicts LBP (OR 3.4) in athletes (Luoto et al. 1995)",
        "priority_critical": 3.0,
        "priority_high": 5.0
    },
    "Wall Angel Test": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Sindrome impingement subacromiale", "Discinesia scapolare", "Lesioni cuffia rotatori"],
        "mechanism": "Rigidità toracica causa scapular winging e pattern overhead alterato",
        "sport_specific": {"Squat": "Barra scivola avanti in low-bar", "Panca": "Ridotto leg drive, arco sub-ottimale"},
        "evidence": "Thoracic extension deficit correlates with shoulder pain in overhead athletes (Laudner et al. 2011)",
        "priority_critical": 4.5,
        "priority_high": 7.0
    },
    "Shoulder Flexion (supine)": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Impingement subacromiale", "Borsite", "Tendinopatia bicipite"],
        "mechanism": "Flessione <170° indica deficit capsulare o rigidità scapolo-toracica",
        "sport_specific": {"Panca": "Traiettoria barra sub-ottimale, stress anteriore spalla"},
        "evidence": "Limited shoulder flexion increases impingement risk (Tyler et al. 2010)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "External Rotation (90° abd)": {
        "risk_threshold": 7.0,
        "risk_injuries": ["SLAP lesion", "Instabilità anteriore", "Lesione cuffia rotatori"],
        "mechanism": "Deficit ER con abduzione indica tensione capsula anteriore, aumenta stress su labrum",
        "sport_specific": {"Panca": "Perdita retrazione scapolare, stress capsula anteriore"},
        "evidence": "Bilateral ER deficit >5° increases shoulder injury risk (Shanley et al. 2011, AJSM)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Pectoralis Minor Length": {
        "risk_threshold": 6.0,
        "risk_injuries": ["Sindrome stretto toracico", "Discinesia scapolare", "Neuropatia ulnare"],
        "mechanism": "PM accorciato causa scapola protratta, riduce spazio subacromiale",
        "sport_specific": {"Panca": "Retrazione scapolare limitata, perdita stabilità"},
        "evidence": "Shortened pectoralis minor associated with shoulder dysfunction (Borstad & Ludewig 2005)",
        "priority_critical": 3.0,
        "priority_high": 6.0,
        "note": "Scala: 0cm=150% (ipermobile), 5cm=100% (ottimale), 13cm=0% (rigido critico)"
    },
    "Thomas Test (modified)": {
        "risk_threshold": 6.0,
        "risk_injuries": ["Tendinopatia flessori anca", "Sindrome da impingement anca", "Lombalgia"],
        "mechanism": "Deficit mobilità flessori anca altera tilt pelvico e meccanica squat/deadlift",
        "sport_specific": {"Squat": "Setup compromesso, limitazione profondità", "Deadlift": "Compenso lombare"},
        "evidence": "Hip flexor tightness correlates with anterior pelvic tilt and LBP (Sanderson & Maxwell 2015)",
        "priority_critical": 3.5,
        "priority_high": 6.0,
        "note": "Scala: -10°=0% (rigido critico), 10°=100% (ottimale), 20°=150% (ipermobile)"
    },
    "ULNT1A (Median nerve)": {
        "risk_threshold": 7.0,
        "risk_injuries": ["Sindrome tunnel carpale", "Neuropatia mediana", "Cervicobrachialgia"],
        "mechanism": "Tensione neurale aumentata può causare parestesie e deficit forza durante pressing",
        "sport_specific": {"Panca": "Parestesie mano durante set, perdita grip"},
        "evidence": "Positive ULNT correlates with nerve pathology (Nee et al. 2012)",
        "priority_critical": 4.0,
        "priority_high": 7.0
    },
    "Plank Test": {
        "risk_threshold": 5.0,
        "risk_injuries": ["Lombalgia cronica", "Instabilità lombo-pelvica", "Ernia discale"],
        "mechanism": "Core endurance <30s indica deficit stabilizzazione, aumenta carico su rachide",
        "sport_specific": {"Squat": "Loss of brace, Valsalva inefficace", "Deadlift": "Flessione lombare sotto carico"},
        "evidence": "Core endurance deficit predicts LBP (McGill et al. 1999)",
        "priority_critical": 2.5,
        "priority_high": 4.5
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
            "critical_tests": ["Weight Bearing Lunge Test", "Hip Internal Rotation", "Hip External Rotation", "Passive Hip Flexion"],
            "threshold": 7.0,
            "note": "ROM anca critico per depth ATG; mobilità anca essenziale per stance largo"
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
            "critical_tests": ["Weight Bearing Lunge Test", "Hip Internal Rotation", "Hip External Rotation", "Shoulder Flexion (supine)"],
            "threshold": 8.0,
            "note": "Mobilità caviglia e anca critiche per receiving position profonda"
        }
    }
}

# Short labels for radar
SHORT_RADAR_LABELS = {
    "Weight Bearing Lunge Test": "Mobilità caviglia",
    "Weight Bearing Lunge Test Dx": "Caviglia Dx",
    "Weight Bearing Lunge Test Sx": "Caviglia Sx",
    "Passive Hip Flexion": "Flessione anca",
    "Passive Hip Flexion Dx": "Flessione anca Dx",
    "Passive Hip Flexion Sx": "Flessione anca Sx",
    "Hip Internal Rotation": "IR anca",
    "Hip Internal Rotation Dx": "IR anca Dx",
    "Hip Internal Rotation Sx": "IR anca Sx",
    "Hip External Rotation": "ER anca",
    "Hip External Rotation Dx": "ER anca Dx",
    "Hip External Rotation Sx": "ER anca Sx",
    "Wall Angel Test": "Wall Angel",
    "Wall Angel Test Dx": "Wall Angel Dx",
    "Wall Angel Test Sx": "Wall Angel Sx",
    "Shoulder ER (adducted, low-bar)": "ER spalla",
    "Shoulder ER (adducted, low-bar) Dx": "ER spalla Dx",
    "Shoulder ER (adducted, low-bar) Sx": "ER spalla Sx",
    "Shoulder Flexion (supine)": "Flessione spalla",
    "Shoulder Flexion (supine) Dx": "Flessione spalla Dx",
    "Shoulder Flexion (supine) Sx": "Flessione spalla Sx",
    "External Rotation (90° abd)": "ER 90° abd",
    "External Rotation (90° abd) Dx": "ER 90° Dx",
    "External Rotation (90° abd) Sx": "ER 90° Sx",
    "Pectoralis Minor Length": "PM length",
    "Pectoralis Minor Length Dx": "PM Dx",
    "Pectoralis Minor Length Sx": "PM Sx",
    "Thomas Test (modified)": "Thomas (flessori anca)",
    "Thomas Test (modified) Dx": "Thomas Dx",
    "Thomas Test (modified) Sx": "Thomas Sx",
    "Active Knee Extension (AKE)": "AKE hamstring",
    "Active Knee Extension (AKE) Dx": "AKE Dx",
    "Active Knee Extension (AKE) Sx": "AKE Sx",
    "Straight Leg Raise (SLR)": "SLR",
    "Straight Leg Raise (SLR) Dx": "SLR Dx",
    "Straight Leg Raise (SLR) Sx": "SLR Sx",
    "Sorensen Endurance": "Endurance lombare",
    "ULNT1A (Median nerve)": "ULNT1A (mediano)",
    "ULNT1A (Median nerve) Dx": "ULNT1A Dx",
    "ULNT1A (Median nerve) Sx": "ULNT1A Sx",
}

# PDF labels
PDF_TEST_LABELS = {
    "Weight Bearing Lunge Test": "Test caviglia",
    "Passive Hip Flexion": "Test mob. flessione anca",
    "Hip Internal Rotation": "Test rotazione interna anca",
    "Hip External Rotation": "Test rotazione esterna anca",
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
    "Hip Internal Rotation": {
        "title": "Rotazione interna anca (flessione 90°)",
        "text": "Test: rotazione interna in flessione 90°. Range normale: 30-45°. Deficit (<25°) può limitare stance largo nello squat. Asimmetrie >10° possono indicare problematiche articolari.",
    },
    "Hip External Rotation": {
        "title": "Rotazione esterna anca (flessione 90°)",
        "text": "Test: rotazione esterna in flessione 90°. Range normale: 40-50°. Deficit (<35°) può limitare apertura dell'anca. Asimmetrie >10° possono indicare problematiche articolari.",
    },
    "Wall Angel Test": {
        "title": "Wall Angel",
        "text": "Test: distanza cm tra braccio e schienale. 0cm = mobilità ottimale (100%). Valori negativi (-1 a -10cm) = ipermobilità. Valori positivi (>0cm) = deficit/rigidità toracica. Importante per overhead e bench press.",
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
                }

seed_defaults()

# -----------------------------
# Scoring and validation
# -----------------------------
def ability_linear(val, ref, higher_is_better=True, test_name=None):
    """
    Calculate functional ability score as percentage of reference value.
    - For higher_is_better=True: score = (val / ref) * 100%
    - For higher_is_better=False: score decreases as value increases from 0 to ref
    - Special cases identified by test_name for precise handling
    
    Values >100% indicate hypermobility (will be flagged separately by detect_hypermobility).
    Reference value = 100%.
    """
    try:
        val = float(val)
        
        # Special handling for Wall Angel Test (ref=0 or by name)
        # Scale: -10cm (ipermobile) to +25cm (deficit rigido)
        # 0cm = 100%, +25cm = 0%, -10cm = 150%
        if ref == 0 or (test_name and "Wall Angel" in test_name):
            if val <= 0:
                # Negative values = hypermobility: -1cm = 105%, -10cm = 150%
                score = 100.0 + (abs(val) / 10.0) * 50.0
            else:
                # Positive values = deficit: +25cm = 0%, linear scale
                score = max(0.0, 100.0 - (val / 25.0) * 100.0)
            return score
        
        # Special handling for Pectoralis Minor Length
        # Scale: 0cm (ipermobile 150%) to 5cm (100%) to 13cm (rigido 0%)
        if test_name and "Pectoralis Minor" in test_name:
            if val <= 5.0:
                # Values ≤5cm: 0cm = 150%, 5cm = 100%
                score = 100.0 + ((5.0 - val) / 5.0) * 50.0
            else:
                # Values >5cm: 13cm = 0%, linear scale
                score = max(0.0, 100.0 - ((val - 5.0) / 8.0) * 100.0)
            return score
        
        # Special handling for Thomas Test
        # Scale: -10° (rigido 0%) to 10° (100%) to 20° (ipermobile 150%)
        if test_name and "Thomas Test" in test_name:
            if val >= 10.0:
                # Values ≥10°: 20° = 150%, 10° = 100%
                score = 100.0 + ((val - 10.0) / 10.0) * 50.0
            else:
                # Values <10°: -10° = 0%, linear scale
                score = max(0.0, 100.0 - ((10.0 - val) / 20.0) * 100.0)
            return score
        
        if ref <= 0:
            return 0.0
            
        if higher_is_better:
            # Simply calculate percentage - no penalty for hypermobility
            score = (val / float(ref)) * 100.0
        else:
            # For lower-is-better tests: score decreases as value increases
            # 0 = 100%, ref = 0%, values > ref get negative scores (capped at 0)
            score = max(0.0, (1.0 - (val / float(ref))) * 100.0)
        return max(0.0, score)  # Allow values >100% for hypermobility
    except Exception:
        return 0.0

def detect_hypermobility(val, ref, higher_is_better, unit):
    """
    Detect if value indicates hypermobility (excessive range of motion).
    Returns: (is_hypermobile, severity, threshold)
    Severity: 'MILD' (115-125%), 'MODERATE' (125-140%), 'SEVERE' (>140%)
    Special case: Wall Angel Test with ref=0 uses negative values for hypermobility
    """
    try:
        val = float(val)
        
        # Special case: Wall Angel Test (ref=0)
        if ref == 0 and higher_is_better:
            if val >= 0:  # Positive or zero values = normal/deficit
                return False, None, None
            # Negative values = hypermobility
            # -1 to -3 cm = MILD, -3 to -7 cm = MODERATE, < -7 cm = SEVERE
            abs_val = abs(val)
            if abs_val >= 7.0:
                return True, "SEVERE", -7.0
            elif abs_val >= 3.0:
                return True, "MODERATE", -3.0
            elif abs_val >= 1.0:
                return True, "MILD", -1.0
            return False, None, None
        
        # Special case: Thomas Test (ref=10.0, higher is better)
        # Hypermobility when val > 10: MILD >11°, MODERATE >14°, SEVERE >17°
        if ref == 10.0 and higher_is_better and "°" in unit:
            if val <= 10.0:  # Normal or deficit range
                return False, None, None
            # Hypermobility range
            if val >= 17.0:
                return True, "SEVERE", 17.0
            elif val >= 14.0:
                return True, "MODERATE", 14.0
            elif val > 11.0:
                return True, "MILD", 11.0
            return False, None, None
        
        # Special case: Pectoralis Minor (ref=5.0, lower is better)
        # Hypermobility when val < 5: MILD <4.5cm, MODERATE <3.5cm, SEVERE <2cm
        if ref == 5.0 and not higher_is_better and "cm" in unit:
            if val >= 5.0:  # Normal or deficit range
                return False, None, None
            # Hypermobility range
            if val < 2.0:
                return True, "SEVERE", 2.0
            elif val < 3.5:
                return True, "MODERATE", 3.5
            elif val < 4.5:
                return True, "MILD", 4.5
            return False, None, None
        
        # Standard logic for other tests
        if not higher_is_better or ref <= 0:
            return False, None, None
        
        percentage = (val / ref) * 100
        
        # Define thresholds based on unit type
        if "°" in unit or "deg" in unit.lower():
            # Angular measurements - tighter threshold
            mild_threshold = 115  # 15% above
            moderate_threshold = 125  # 25% above 
            severe_threshold = 140  # 40% above
        elif "cm" in unit or "mm" in unit:
            # Linear measurements
            mild_threshold = 120
            moderate_threshold = 135
            severe_threshold = 150
        elif "sec" in unit:
            # Endurance - wider range acceptable
            mild_threshold = 150
            moderate_threshold = 200
            severe_threshold = 300
        else:
            mild_threshold = 115
            moderate_threshold = 130
            severe_threshold = 150
        
        threshold_val = ref * (mild_threshold / 100)
        
        if percentage >= severe_threshold:
            return True, "SEVERE", threshold_val
        elif percentage >= moderate_threshold:
            return True, "MODERATE", threshold_val
        elif percentage >= mild_threshold:
            return True, "MILD", threshold_val
        else:
            return False, None, threshold_val
            
    except Exception:
        return False, None, None

def symmetry_score(dx, sx, unit, ref=None, higher_is_better=True, test_name=None):
    """
    Calculate symmetry score between left and right sides.
    Returns 100% for perfect symmetry, 0% for maximum asymmetry.
    
    If ref and higher_is_better are provided, calculates asymmetry based on 
    performance difference (score difference) rather than raw value difference.
    This ensures clinical relevance for all test types.
    """
    try:
        dx = float(dx)
        sx = float(sx)
        
        # If ref provided, calculate asymmetry based on score difference
        if ref is not None and ref != 0:
            score_dx = ability_linear(dx, ref, higher_is_better, test_name)
            score_sx = ability_linear(sx, ref, higher_is_better, test_name)
            score_diff = abs(score_dx - score_sx)
            
            # Asymmetry scale: 0-20% score difference is acceptable
            # >20% score difference indicates significant asymmetry
            return max(0.0, 100.0 - (score_diff / 20.0) * 100.0)
        
        # Fallback to absolute difference method
        diff = abs(dx - sx)
        if "°" in unit:
            scale = 20.0
        elif unit == "cm":
            scale = 8.0
        else:
            scale = 10.0
        return 100.0 * max(0.0, 1.0 - min(diff, scale) / scale)
    except Exception:
        return 0.0

def validate_input(test_name, value, side=None):
    """Validate input values and provide clinical warnings"""
    warnings = []
    
    if test_name == "Wall Angel Test":
        if value < -10:
            warnings.append("⚠️ Valore oltre range ipermobilità (<-10cm). Verificare misurazione.")
        elif value <= -7:
            warnings.append("🔴 Ipermobilità SEVERA (≤-7cm). Alto rischio instabilità. Necessaria stabilizzazione.")
        elif value <= -3:
            warnings.append("🟠 Ipermobilità MODERATA (-3 a -7cm). Rischio instabilità. Controllo motorio essenziale.")
        elif value < 0:
            warnings.append("🟡 Ipermobilità LIEVE (<0cm). Monitorare stabilità dinamica.")
        elif value > 20:
            warnings.append("🚨 Rigidità toracica severa (>20cm). Limitazione molto significativa overhead.")
        elif value > 15:
            warnings.append("🔴 Rigidità toracica marcata (>15cm). Limitazione significativa overhead.")
        elif value > 5:
            warnings.append("⚠️ Rigidità toracica moderata (>5cm). Priorità mobilizzazione toracica.")
    
    elif test_name == "Weight Bearing Lunge Test":
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
    
    elif test_name == "Thomas Test (modified)":
        if value > 20:
            warnings.append("⚠️ Valore oltre range ipermobilità (>20°). Verificare misurazione.")
        elif value >= 17:
            warnings.append("🔴 Ipermobilità SEVERA flessori anca (≥17°). Alto rischio instabilità. Necessario controllo motorio.")
        elif value >= 14:
            warnings.append("🟠 Ipermobilità MODERATA flessori anca (14-16°). Rischio instabilità. Priorità stabilizzazione.")
        elif value > 11:
            warnings.append("🟡 Ipermobilità LIEVE flessori anca (11-13°). Monitorare controllo dinamico.")
        elif value < -5:
            warnings.append("🚨 Rigidità severa flessori anca (<-5°). Limitazione significativa squat/overhead.")
        elif value < 5:
            warnings.append("⚠️ Rigidità moderata flessori anca (<5°). Priorità mobilizzazione anteriore anca.")
    
    elif test_name == "Pectoralis Minor Length":
        if value > 13:
            warnings.append("⚠️ Valore oltre range ipomobilità (>13cm). Verificare misurazione.")
        elif value >= 11:
            warnings.append("🚨 Rigidità severa pettorale minore (≥11cm). Alto rischio disfunzione scapolare.")
        elif value >= 8:
            warnings.append("🔴 Rigidità marcata pettorale minore (8-10cm). Limitazione significativa overhead.")
        elif value > 6:
            warnings.append("⚠️ Rigidità moderata pettorale minore (6-7cm). Priorità rilascio anteriore spalla.")
        elif value < 2:
            warnings.append("🔴 Ipermobilità SEVERA pettorale minore (<2cm). Instabilità anteriore spalla. Necessaria stabilizzazione.")
        elif value < 3.5:
            warnings.append("🟠 Ipermobilità MODERATA pettorale minore (<3.5cm). Rischio instabilità anteriore.")
        elif value < 4.5:
            warnings.append("🟡 Ipermobilità LIEVE pettorale minore (<4.5cm). Monitorare stabilità scapolare.")
    
    return warnings

# -----------------------------
# Clinical algorithms
# -----------------------------
# EBM Risk Assessment
# -----------------------------
def assess_injury_risk(df, sport, session_state):
    """Evidence-based injury risk assessment with prioritization"""
    risk_warnings = []
    
    for _, row in df.iterrows():
        test_name = row["Test"]
        score = row["Score"]
        region = row["Regione"]
        
        # Check if test is in risk database
        if test_name not in INJURY_RISK_DATABASE:
            continue
        
        risk_data = INJURY_RISK_DATABASE[test_name]
        
        # Check bilateral tests separately (Dx and Sx)
        dx_val = row.get("Dx", "")
        sx_val = row.get("Sx", "")
        ref = row.get("Rif", 10.0)
        
        # Get higher_is_better from row data
        higher_is_better = True
        for section_tests in TESTS.values():
            for test_tuple in section_tests:
                if test_tuple[0] == test_name:
                    higher_is_better = test_tuple[6] if len(test_tuple) > 6 else True
                    break
        
        # Function to evaluate a single side/value
        def evaluate_side(score_val, side_label=""):
            priority = "BASSO"
            priority_icon = "🟢"
            urgency_days = ">30"
            
            # Convert thresholds from 0-10 scale to 0-100 scale
            critical_threshold = risk_data.get("priority_critical", 3.0) * 10
            high_threshold = risk_data.get("priority_high", 5.0) * 10
            moderate_threshold = risk_data.get("risk_threshold", 7.0) * 10
            
            if score_val < critical_threshold:
                priority = "CRITICO"
                priority_icon = "🔴"
                urgency_days = "IMMEDIATO (0-7 giorni)"
                action = "STOP carichi >70% 1RM - Intervento immediato necessario"
            elif score_val < high_threshold:
                priority = "ALTO"
                priority_icon = "🟠"
                urgency_days = "7-14 giorni"
                action = "RIDURRE volume/intensità - Iniziare protocollo correttivo"
            elif score_val < moderate_threshold:
                priority = "MODERATO"
                priority_icon = "🟡"
                urgency_days = "14-30 giorni"
                action = "Monitorare e integrare lavoro accessorio"
            else:
                return None  # Score OK, no warning
            
            # Build warning
            test_display = f"{test_name} {side_label}".strip()
            warning = {
                "priority": priority,
                "priority_icon": priority_icon,
                "urgency": urgency_days,
                "test": test_display,
                "score": score_val,
                "region": region,
                "risk_injuries": risk_data["risk_injuries"],
                "mechanism": risk_data["mechanism"],
                "sport_specific": risk_data.get("sport_specific", {}).get(sport, ""),
                "evidence": risk_data["evidence"],
                "action": action,
                "note": risk_data.get("note", "")
            }
            return warning
        
        # Check if bilateral test with separate Dx/Sx values
        if dx_val != "" and sx_val != "" and pd.notna(dx_val) and pd.notna(sx_val):
            # Evaluate Dx side
            dx_score = round(ability_linear(float(dx_val), ref, higher_is_better, test_name), 2)
            warning_dx = evaluate_side(dx_score, "Dx")
            if warning_dx:
                # Add pain multiplier
                if row.get("DoloreDx", False):
                    warning_dx["pain_present"] = True
                    warning_dx["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                else:
                    warning_dx["pain_present"] = False
                risk_warnings.append(warning_dx)
            
            # Evaluate Sx side
            sx_score = round(ability_linear(float(sx_val), ref, higher_is_better, test_name), 2)
            warning_sx = evaluate_side(sx_score, "Sx")
            if warning_sx:
                # Add pain multiplier
                if row.get("DoloreSx", False):
                    warning_sx["pain_present"] = True
                    warning_sx["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                else:
                    warning_sx["pain_present"] = False
                risk_warnings.append(warning_sx)
        else:
            # Unilateral test or average score evaluation
            warning = evaluate_side(score)
            if warning:
                # Add pain multiplier
                if row.get("Dolore", False) or row.get("DoloreDx", False) or row.get("DoloreSx", False):
                    warning["pain_present"] = True
                    warning["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                else:
                    warning["pain_present"] = False
                risk_warnings.append(warning)
    
    # NEW: Check for HYPERMOBILITY (excessive range of motion)
    for _, row in df.iterrows():
        test_name = row["Test"]
        score = row["Score"]
        region = row["Regione"]
        unit = row.get("Unità", "")
        ref = row.get("Rif", 0)
        
        # Get higher_is_better from test definition
        higher_is_better = True
        for section_tests in TESTS.values():
            for test_tuple in section_tests:
                if test_tuple[0] == test_name:
                    higher_is_better = test_tuple[6] if len(test_tuple) > 6 else True
                    break
        
        # Function to evaluate hypermobility for a single side/value
        def evaluate_hypermobility(val, side_label=""):
            # Detect hypermobility
            is_hypermobile, severity, threshold = detect_hypermobility(val, ref, higher_is_better, unit)
            
            if not is_hypermobile:
                return None
            
            # Determine priority based on severity and region
            if severity == "SEVERE":
                priority = "ALTO"
                priority_icon = "🟠"
                urgency_days = "7-14 giorni"
            elif severity == "MODERATE":
                priority = "MODERATO"
                priority_icon = "🟡"
                urgency_days = "14-21 giorni"
            else:  # MILD
                priority = "MODERATO"
                priority_icon = "🟡"
                urgency_days = "21-30 giorni"
            
            # Region-specific risks
            if region in ["shoulder", "spalla"]:
                risk_injuries = ["Instabilità gleno-omerale", "Lussazione anteriore/posteriore", "Lesioni labrali", "Sindrome da impingement"]
                mechanism = "Lassità capsulo-legamentosa eccessiva riduce controllo neuromuscolare e aumenta traslazione omerale"
                evidence = "Liaghat et al. 2021: OR=3.25 per infortuni spalla in atleti ipermobili (meta-analisi)"
                action = "STABILIZZAZIONE: Lavoro eccentrico cuffia rotatori, controllo scapolare, evitare ROM estremi sotto carico"
            elif region in ["hip", "anca"]:
                risk_injuries = ["Impingement femoro-acetabolare", "Lesioni labrali", "Instabilità anca", "Tendinopatia flessori"]
                mechanism = "Eccessivo ROM può causare conflitto osseo o instabilità capsulare"
                evidence = "Pasanen et al. 2025: Ipermobilità anca associata a ↑rischio lesioni LCA in atlete"
                action = "CONTROLLO MOTORIO: Squat mono-podalico, copenhagen plank, limitare ROM fine corsa" 
            elif region in ["knee", "ginocchio"]:
                risk_injuries = ["Lesione LCA", "Instabilità rotulea", "Sublussazione patella"]
                mechanism = "Iperestensione ginocchio riduce stabilità passiva e sovraccarica legamenti"
                evidence = "Pasanen et al. 2025: Lassità articolare è fattore di rischio LCA prospettico"
                action = "RINFORZO: Focus eccentrico ischio-crurali, evitare iperestensione, taping correttivo"
            elif region in ["ankle", "caviglia"]:
                risk_injuries = ["Distorsioni ricorrenti", "Instabilità cronica caviglia", "Tendinopatia peroneale"]
                mechanism = "Lassità legamentosa laterale riduce stabilità propriocettiva"
                evidence = "Nathan et al. 2018: Ipermobili hanno maggior incidenza lussazioni vs distorsioni"
                action = "PROPRIOCEZIONE: Balance board, single-leg deadlift, rinforzo peroneali"
            elif region in ["thoracic", "toracica"]:
                risk_injuries = ["Instabilità scapolare", "Compensi posturali", "Disfunzione movimento overhead"]
                mechanism = "Eccessiva mobilità toracica può indicare instabilità segmentaria"
                evidence = "Joint hypermobility associato a pattern movimento alterati"
                action = "STABILIZZAZIONE: Core stability, controllo scapolare, rinforzo paravertebrali"
            else:
                risk_injuries = ["Instabilità articolare", "Pattern movimento compensatori"]
                mechanism = "Eccessivo ROM può indicare lassità legamentosa e ridotto controllo neuromuscolare"
                evidence = "Joint hypermobility associato a ↑durata infortuni (Nathan 2018)"
                action = "STABILIZZAZIONE: Lavoro isometrico end-range, controllo motorio, rinforzo specifico"
            
            # Sport-specific notes
            sport_specific = ""
            if sport in ["Powerlifting", "Weightlifting"]:
                if region == "shoulder":
                    sport_specific = "Rischio perdita transfer forza in panca e overhead press. Setup instabile."
                elif region == "hip":
                    sport_specific = "Squat instabile, 'folding' in buca, difficoltà a mantenere tensione"
            elif sport == "CrossFit":
                if region == "shoulder":
                    sport_specific = "Alto rischio in overhead (snatch, OHS, HSPU). Kipping instabile."
                elif region == "hip":  
                    sport_specific = "Rischio in olympic lifts, box jumps, wall balls con iperestensione"
            
            # Get score for this specific side
            side_score = ability_linear(val, ref, higher_is_better, test_name)
            
            test_display = f"{test_name} {side_label}".strip()
            warning = {
                "priority": priority,
                "priority_icon": priority_icon,
                "urgency": urgency_days,
                "test": test_display,
                "score": side_score,
                "region": region,
                "risk_injuries": risk_injuries,
                "mechanism": mechanism,
                "sport_specific": sport_specific,
                "evidence": evidence,
                "action": action,
                "note": f"⚖️ IPERMOBILITÀ {severity}: {val:.1f}{unit} (riferimento: {ref:.1f}{unit}, soglia rischio: >{threshold:.1f}{unit})",
                "pain_present": False,
                "hypermobility": True,
                "severity": severity
            }
            return warning
        
        # Get value - use Dx/Sx columns for bilateral tests, otherwise Valore
        try:
            dx_val = row.get("Dx", "")
            sx_val = row.get("Sx", "")
            
            # Check if this is a bilateral test (Dx/Sx columns populated)
            if dx_val != "" and sx_val != "" and pd.notna(dx_val) and pd.notna(sx_val):
                dx_val = float(dx_val)
                sx_val = float(sx_val)
                
                # Evaluate Dx side
                warning_dx = evaluate_hypermobility(dx_val, "Dx")
                if warning_dx:
                    warning_dx["pain_present"] = row.get("DoloreDx", False)
                    if warning_dx["pain_present"]:
                        warning_dx["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                    risk_warnings.append(warning_dx)
                
                # Evaluate Sx side
                warning_sx = evaluate_hypermobility(sx_val, "Sx")
                if warning_sx:
                    warning_sx["pain_present"] = row.get("DoloreSx", False)
                    if warning_sx["pain_present"]:
                        warning_sx["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                    risk_warnings.append(warning_sx)
            else:
                # Unilateral test - use Valore field
                val_str = row.get("Valore", "")
                val = float(val_str)
                
                warning = evaluate_hypermobility(val)
                if warning:
                    warning["pain_present"] = row.get("Dolore", False)
                    if warning["pain_present"]:
                        warning["action"] += " ⚠️ DOLORE PRESENTE - Priorità aumentata"
                    risk_warnings.append(warning)
        except:
            continue
    
    # Sort by priority (Critical > High > Moderate), then by score
    priority_order = {"CRITICO": 0, "ALTO": 1, "MODERATO": 2, "BASSO": 3}
    risk_warnings.sort(key=lambda x: (priority_order[x["priority"]], x["score"]))
    
    return risk_warnings

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
    """Generate detailed clinical conclusions with priorities and asymmetries"""
    recommendations = []
    priority_counter = 1
    
    # Priority 1: Critical scores (<40%) - URGENT
    critical = df[df["Score"] < 40].copy()
    if not critical.empty:
        for _, row in critical.iterrows():
            test_name = row["Test"]
            region = row["Regione"]
            score = row["Score"]
            
            # Calculate severity percentage
            deficit_percent = int(100 - score)
            
            # Determine impact
            impact = "Limitazione SEVERA che può causare compensi e aumentare il rischio di infortunio"
            recommendation = "Richiede intervento immediato e prioritario. Evitare carichi elevati (>85% 1RM) sulla regione fino a miglioramento."
            
            rec = {
                "priority_num": priority_counter,
                "priority": "🚨 CRITICA",
                "test": test_name,
                "region": region,
                "score": score,
                "deficit_percent": deficit_percent,
                "category": "Valutazione Urgente",
                "impact": impact,
                "recommendation": recommendation,
                "timeline": "2-4 settimane"
            }
            recommendations.append(rec)
            priority_counter += 1
    
    # Priority 2: Significant Asymmetries (SymScore < 60%)
    asymmetries = df[df["SymScore"] < 60].copy()
    if not asymmetries.empty:
        for _, row in asymmetries.iterrows():
            test_name = row["Test"]
            region = row["Regione"]
            sym_score = row["SymScore"]
            delta = row.get("Delta", 0)
            unit = row.get("Unità", "")
            
            # Determine side info
            dx_val = row.get("Dx", 0)
            sx_val = row.get("Sx", 0)
            
            if dx_val > sx_val:
                weaker_side = "sinistra"
                stronger_side = "destra"
            else:
                weaker_side = "destra"
                stronger_side = "sinistra"
            
            # Calculate asymmetry percentage
            higher_val = max(dx_val, sx_val) if max(dx_val, sx_val) > 0 else 1
            asym_percent = int((abs(delta) / higher_val) * 100)
            
            impact = f"Deficit significativo lato {weaker_side} rispetto al {stronger_side}. Rischio di sovraccarico compensatorio e pattern di movimento alterato."
            recommendation = f"Correggere l'asimmetria con lavoro specifico sul lato {weaker_side}. Monitorare bilateralità nei movimenti funzionali."
            
            rec = {
                "priority_num": priority_counter,
                "priority": "⚖️ ASIMMETRIA",
                "test": test_name,
                "region": region,
                "score": sym_score,
                "delta": f"{delta}{unit}",
                "asym_percent": asym_percent,
                "weaker_side": weaker_side,
                "category": "Deficit di Simmetria Bilaterale",
                "impact": impact,
                "recommendation": recommendation,
                "timeline": "3-6 settimane"
            }
            recommendations.append(rec)
            priority_counter += 1
    
    # Priority 3: Moderate scores (40-70%) - IMPORTANT
    moderate = df[(df["Score"] >= 40) & (df["Score"] < 70)].copy()
    if not moderate.empty:
        for _, row in moderate.iterrows():
            test_name = row["Test"]
            region = row["Regione"]
            score = row["Score"]
            
            deficit_percent = int(100 - score)
            
            impact = "Limitazione moderata che può ridurre la performance e potenzialmente evolvere in problematica se non gestita."
            recommendation = "Inserire lavoro mirato e progressivo. Monitorare durante carico crescente."
            
            rec = {
                "priority_num": priority_counter,
                "priority": "⚠️ MODERATA",
                "test": test_name,
                "region": region,
                "score": score,
                "deficit_percent": deficit_percent,
                "category": "Limitazione Moderata",
                "impact": impact,
                "recommendation": recommendation,
                "timeline": "4-8 settimane"
            }
            recommendations.append(rec)
            priority_counter += 1
    
    # Sort by priority: Critical first, then Asymmetries, then Moderate
    def sort_key(rec):
        if rec["priority"] == "🚨 CRITICA":
            return (0, rec["score"])  # Lower score = higher priority
        elif rec["priority"] == "⚖️ ASIMMETRIA":
            return (1, rec["score"])  # Lower sym score = higher priority
        else:
            return (2, rec["score"])  # Lower score = higher priority
    
    recommendations.sort(key=sort_key)
    
    # Renumber priorities after sorting
    for i, rec in enumerate(recommendations, 1):
        rec["priority_num"] = i
    
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
                
                # Special case for Wall Angel Test: allow negative values and set proper range
                if name == "Wall Angel Test":
                    min_val = -10.0
                    max_val = 25.0  # Allow up to 25cm positive (deficit)
                elif name == "Thomas Test (modified)":
                    min_val = -10.0
                    max_val = 20.0  # Allow from -10° (rigid) to 20° (hypermobile)
                elif name == "Pectoralis Minor Length":
                    min_val = 0.0
                    max_val = 13.0  # Allow from 0cm (hypermobile) to 13cm (rigid)
                else:
                    min_val = 0.0
                    max_val = rec.get("ref", ref) * 1.5 if rec.get("ref", ref) > 0 else 10.0
                
                if rec.get("bilat", False):
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        dx = st.number_input(f"Dx ({unit})", min_value=min_val, max_value=max_val, value=float(rec.get("Dx", 0.0)), step=0.1, key=f"{key}_Dx")
                        pdx = st.checkbox("Dolore Dx", value=bool(rec.get("DoloreDx", False)), key=f"{key}_pDx")
                    with c2:
                        sx = st.number_input(f"Sx ({unit})", min_value=min_val, max_value=max_val, value=float(rec.get("Sx", 0.0)), step=0.1, key=f"{key}_Sx")
                        psx = st.checkbox("Dolore Sx", value=bool(rec.get("DoloreSx", False)), key=f"{key}_pSx")
                    
                    rec.update({"Dx": dx, "Sx": sx, "DoloreDx": pdx, "DoloreSx": psx})
                    sc = ability_linear((dx + sx) / 2.0, rec.get("ref", ref), rec.get("higher_is_better", hib), name)
                    sym = symmetry_score(dx, sx, unit, rec.get("ref", ref), rec.get("higher_is_better", hib), name)
                    st.caption(f"Score: **{sc:.0f}%** — Δ {abs(dx - sx):.1f} {unit} — Sym: **{sym:.0f}%")
                    
                    # Validation
                    warnings_dx = validate_input(name, dx, "Dx")
                    warnings_sx = validate_input(name, sx, "Sx")
                    for w in warnings_dx + warnings_sx:
                        st.warning(w)
                
                else:
                    val = st.number_input(f"Valore ({unit})", min_value=min_val, max_value=max_val, value=float(rec.get("Val", 0.0)), step=0.1, key=f"{key}_Val")
                    p = st.checkbox("Dolore", value=bool(rec.get("Dolore", False)), key=f"{key}_p")
                    rec.update({"Val": val, "Dolore": p})
                    sc = ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib), name)
                    st.caption(f"Score: **{sc:.0f}%**")
                    
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
                sc = round(ability_linear(avg, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
                delta = round(abs(dx - sx), 2)
                sym = round(symmetry_score(dx, sx, unit, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
                dolore_dx = bool(rec.get("DoloreDx", False))
                dolore_sx = bool(rec.get("DoloreSx", False))
                dolore_any = dolore_dx or dolore_sx
                rows.append([
                    sec, name, unit, rec.get("ref", ref), f"{avg:.1f}", sc,
                    round(dx, 2), round(sx, 2), delta, sym, dolore_any, region,
                    dolore_dx, dolore_sx
                ])
            else:
                val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce")
                val = 0.0 if pd.isna(val) else float(val)
                sc = round(ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
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

def build_df_for_radar(section):
    """Build dataframe for radar chart with bilateral tests split into Dx/Sx rows"""
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
                # Create TWO rows for bilateral tests: one for Dx, one for Sx
                dx = pd.to_numeric(rec.get("Dx", 0.0), errors="coerce")
                sx = pd.to_numeric(rec.get("Sx", 0.0), errors="coerce")
                dx = 0.0 if pd.isna(dx) else float(dx)
                sx = 0.0 if pd.isna(sx) else float(sx)
                
                sc_dx = round(ability_linear(dx, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
                sc_sx = round(ability_linear(sx, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
                
                dolore_dx = bool(rec.get("DoloreDx", False))
                dolore_sx = bool(rec.get("DoloreSx", False))
                
                # Row for Dx
                rows.append([
                    sec, f"{name} Dx", unit, rec.get("ref", ref), f"{dx:.1f}", sc_dx,
                    round(dx, 2), "", "", "", dolore_dx, region,
                    dolore_dx, False
                ])
                
                # Row for Sx
                rows.append([
                    sec, f"{name} Sx", unit, rec.get("ref", ref), f"{sx:.1f}", sc_sx,
                    "", round(sx, 2), "", "", dolore_sx, region,
                    False, dolore_sx
                ])
            else:
                val = pd.to_numeric(rec.get("Val", 0.0), errors="coerce")
                val = 0.0 if pd.isna(val) else float(val)
                sc = round(ability_linear(val, rec.get("ref", ref), rec.get("higher_is_better", hib), name), 2)
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
def radar_plot_matplotlib(df, title="Punteggi (0–140%)"):
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

    # Create figure with space for legend
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Add colored background zones
    theta = np.linspace(0, 2 * np.pi, 100)
    
    # Yellow zone (0-90): Ipomobilità
    r_yellow = np.full_like(theta, 90)
    ax.fill_between(theta, 0, r_yellow, color="#FDE047", alpha=0.5, label="Ipomobilità (0-90%)")
    
    # Green zone (90-100): Normalità
    r_green_min = np.full_like(theta, 90)
    r_green_max = np.full_like(theta, 100)
    ax.fill_between(theta, r_green_min, r_green_max, color="#86EFAC", alpha=0.6, label="Normalità (90-100%)")
    
    # Blue zone (100-140): Ipermobilità
    r_blue_min = np.full_like(theta, 100)
    r_blue_max = np.full_like(theta, 140)
    ax.fill_between(theta, r_blue_min, r_blue_max, color="#93C5FD", alpha=0.5, label="Ipermobilità (>100%)")
    
    # Plot data
    ax.plot(angles, values, linewidth=2.5, linestyle="solid", color=PRIMARY, zorder=3)
    ax.fill(angles, values, alpha=0.25, color=PRIMARY, zorder=3)

    # Node colors based on value
    node_colors = []
    for v in values[:-1]:
        if v > 115:
            if v > 140:  # Ipermobilità severa
                node_colors.append("#DC2626")
            elif v > 125:  # Ipermobilità moderata
                node_colors.append("#F59E0B")
            else:  # Ipermobilità lieve
                node_colors.append("#3B82F6")
        elif v >= 71:
            node_colors.append("#16A34A")  # Verde
        elif v >= 51:
            node_colors.append("#EAB308")  # Giallo
        elif v >= 31:
            node_colors.append("#F59E0B")  # Arancione
        else:
            node_colors.append("#DC2626")  # Rosso
    
    node_angles = angles[:-1]
    ax.scatter(node_angles, values[:-1], c=node_colors, s=120, zorder=5, edgecolors="k", linewidths=2)

    ax.set_yticks([20, 40, 60, 80, 100, 120, 140])
    ax.set_ylim(0, 140)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10, fontweight="bold")
    ax.set_title(sanitize_text_for_plot(title), y=1.1, fontsize=16, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=9, 
             frameon=True, fancybox=True, shadow=True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

SIMPLE_TEST_LABELS = {
    "Weight Bearing Lunge Test": "Caviglia",
    "Passive Hip Flexion": "Flessione anca",
    "Hip Internal Rotation": "IR anca",
    "Hip External Rotation": "ER anca",
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
        if score >= 71:
            colors_map.append("#16A34A")  # Verde
        elif score >= 51:
            colors_map.append("#EAB308")  # Giallo
        elif score >= 31:
            colors_map.append("#F59E0B")  # Arancione
        else:
            colors_map.append("#DC2626")  # Rosso

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, scores, color=colors_map, edgecolor="black", linewidth=1.2)
    ax.set_xlabel("SymScore (0–100%)", fontsize=12, fontweight="bold")
    ax.set_title(sanitize_text_for_plot(title), fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
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
        polar=dict(radialaxis=dict(range=[0, 140], showticklabels=True, tickfont=dict(size=10))),
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
                 range_x=[0, 100])
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
                score = ability_linear(avg, ref, hib, test_name)
                scores.append(score)
                if test_data.get("DoloreDx") or test_data.get("DoloreSx"):
                    pains += 1
            else:
                val = test_data.get("Val", 0.0)
                score = ability_linear(val, ref, hib, test_name)
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
    ax1.set_ylabel("Score Medio (0-100%)", fontsize=12, fontweight="bold")
    ax1.set_title("Progressione Score nel Tempo", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    ax1.axhline(y=71, color='green', linestyle='--', alpha=0.5, label='Target (71%)')
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
    
    # Enhanced color palette
    COLOR_CRITICAL = colors.HexColor("#dc2626")
    COLOR_ASYMMETRY = colors.HexColor("#ea580c")
    COLOR_MODERATE = colors.HexColor("#ca8a04")
    COLOR_SUCCESS = colors.HexColor("#16a34a")
    COLOR_PRIMARY = colors.HexColor(PRIMARY)
    COLOR_BG_CRITICAL = colors.HexColor("#fee2e2")
    COLOR_BG_ASYMMETRY = colors.HexColor("#ffedd5")
    COLOR_BG_MODERATE = colors.HexColor("#fef9c3")
    COLOR_BG_INFO = colors.HexColor("#eff6ff")
    
    # Improved typography
    heading = ParagraphStyle("heading", parent=styles["Heading2"], alignment=TA_LEFT, 
                            textColor=COLOR_PRIMARY, fontSize=15, spaceAfter=14, fontName="Helvetica-Bold",
                            spaceBefore=8, borderWidth=0, leftIndent=0)
    heading_large = ParagraphStyle("heading_large", parent=heading, fontSize=16, 
                                   textColor=COLOR_PRIMARY, spaceAfter=16, spaceBefore=12)
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=9, leading=11)
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8)

    story = []
    
    # Enhanced Header with colored bar
    header_table = Table([
        [
            RLImage(io.BytesIO(logo_bytes), width=4.0 * cm, height=1.0 * cm),
            Paragraph(f"<b>Report Valutazione Completo</b><br/><font size=10>{sanitize_text_for_plot(section)}</font>", title_style),
            Paragraph(f"<b>Atleta:</b> {athlete}<br/><b>Valutatore:</b> {evaluator}<br/><b>Data:</b> {date_str}", body),
        ]
    ], colWidths=[4.2 * cm, 8.8 * cm, 4.0 * cm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_INFO),
        ("BOX", (0, 0), (-1, -1), 1.5, COLOR_PRIMARY),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 20))

    # Red Flags Section (if any) - Enhanced styling
    if session_state.get("red_flags"):
        red_flags_header = Table([[
            Paragraph("<b>⚠️ RED FLAGS IDENTIFICATE</b>", heading_large)
        ]], colWidths=[17*cm])
        red_flags_header.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_CRITICAL),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(red_flags_header)
        story.append(Spacer(1, 8))
        
        flags_list = "<br/>".join([f"• {RED_FLAGS.get(flag, flag)}" for flag in session_state["red_flags"]])
        flags_content = f"{flags_list}<br/><br/><b>AZIONE: Riferimento medico raccomandato prima di procedere.</b>"
        
        flags_box = Table([[Paragraph(flags_content, body)]], colWidths=[17*cm])
        flags_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_BG_CRITICAL),
            ("BOX", (0,0), (-1,-1), 2, COLOR_CRITICAL),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(flags_box)
        story.append(Spacer(1, 18))

    # Anamnesis with improved table style
    story.append(Paragraph("<b>Anamnesi</b>", heading))
    anamnesis_data = [
        ["Sport/Attività:", session_state.get("sport", "N/A")],
        ["Frequenza allenamento:", f"{session_state.get('training_frequency', 0)} giorni/settimana"],
        ["Storia infortuni:", session_state.get("injury_history", "Nessuna") or "Nessuna"],
        ["Sintomi attuali:", session_state.get("current_symptoms", "Nessuno") or "Nessuno"],
        ["Obiettivi:", session_state.get("goals", "N/A") or "N/A"],
    ]
    anamnesis_table = Table(anamnesis_data, colWidths=[4.5 * cm, 12.5 * cm])
    anamnesis_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f9fafb")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(anamnesis_table)
    story.append(Spacer(1, 18))

    # Functional Scales
    story.append(Paragraph("<b>Scale Funzionali</b>", heading))
    nprs = session_state.get("nprs", 0)
    story.append(Paragraph(f"<b>NPRS</b> (dolore medio ultima settimana): <b>{nprs}/10</b>", body))
    
    psfs_activities = session_state.get("psfs_activities", [])
    if psfs_activities:
        story.append(Paragraph("<b>PSFS</b> (Patient-Specific Functional Scale):", body))
        for act in psfs_activities:
            story.append(Paragraph(f"• {act['activity']}: <b>{act['score']}/10</b>", body))
    story.append(Spacer(1, 18))

    # Enhanced Metrics Summary with colored boxes
    avg_score = df["Score"].mean() if "Score" in df.columns and not df["Score"].isna().all() else 0.0
    n_dolore = int(df["Dolore"].sum()) if "Dolore" in df.columns else 0
    sym_mean = df["SymScore"].mean() if "SymScore" in df.columns else np.nan
    
    story.append(Paragraph("<b>Sintesi Metriche</b>", heading))
    
    # Use color-coded metric boxes
    metric_data = [[
        Paragraph("<b>Score medio</b><br/><font size=14><b>{:.0f}%</b></font>".format(avg_score), body),
        Paragraph("<b>Test con dolore</b><br/><font size=14><b>{}</b></font>".format(n_dolore), body),
        Paragraph("<b>Symmetry medio</b><br/><font size=14><b>{}</b></font>".format(f"{sym_mean:.0f}%" if not pd.isna(sym_mean) else "n/a"), body)
    ]]
    
    metrics_table = Table(metric_data, colWidths=[5.5*cm, 5.5*cm, 5.5*cm])
    metrics_table.setStyle(TableStyle([
        ("BOX", (0,0), (-1,-1), 2, COLOR_PRIMARY),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND", (0,0), (-1,-1), COLOR_BG_INFO),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("GRID", (0,0), (-1,-1), 1, colors.white)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 18))

    # Results table with improved styling - using radar dataframe for consistency
    story.append(Paragraph("<b>Risultati Test Oggettivi</b>", heading))
    story.append(Spacer(1, 8))
    
    # Use radar dataframe to show tests split by Dx/Sx like in the radar chart
    df_radar = build_df_for_radar(section)
    df_radar = df_radar[df_radar["Score"].notnull()].copy()
    
    df_radar["Status"] = df_radar["Score"].apply(lambda s: "✔" if s >= 71 else ("⚠" if s >= 31 else "✖"))
    
    # Add radar label in parentheses
    def format_test_with_radar_label(test_name):
        radar_label = SHORT_RADAR_LABELS.get(test_name, test_name)
        base_name = test_name.replace(" Dx", "").replace(" Sx", "")
        pdf_label = pdf_test_label(base_name)
        return f"{pdf_label} ({radar_label})"
    
    df_radar["TestPdf"] = df_radar["Test"].apply(format_test_with_radar_label)
    
    table_cols = ["Status", "Test", "Valore", "Unità", "Rif", "Score"]
    table_data = [table_cols]
    for _, r in df_radar.iterrows():
        table_data.append([r["Status"], r["TestPdf"], r["Valore"], r["Unità"], r["Rif"], f"{r['Score']:.1f}"])
    
    colWidths = [1.2*cm, 7.0*cm, 2.0*cm, 2.0*cm, 1.6*cm, 2.0*cm]
    result_table = Table(table_data, colWidths=colWidths, repeatRows=1)
    
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
    
    for i in range(1, len(table_data)):
        bg = colors.HexColor("#f9fafb") if i % 2 == 0 else colors.white
        style.add("BACKGROUND", (0, i), (-1, i), bg)
        try:
            score = float(table_data[i][5])
            if score >= 71:
                color = colors.HexColor("#d1fae5")  # Verde chiaro
                style.add("TEXTCOLOR", (5, i), (5, i), COLOR_SUCCESS)
            elif score >= 51:
                color = colors.HexColor("#fef9c3")  # Giallo chiaro
                style.add("TEXTCOLOR", (5, i), (5, i), colors.HexColor("#ca8a04"))
            elif score >= 31:
                color = colors.HexColor("#fed7aa")  # Arancione chiaro
                style.add("TEXTCOLOR", (5, i), (5, i), COLOR_MODERATE)
            else:
                color = colors.HexColor("#fee2e2")  # Rosso chiaro
                style.add("TEXTCOLOR", (5, i), (5, i), COLOR_CRITICAL)
            style.add("BACKGROUND", (5, i), (5, i), color)
            style.add("FONTNAME", (5, i), (5, i), "Helvetica-Bold")
        except Exception:
            pass
    
    result_table.setStyle(style)
    story.append(result_table)
    story.append(Spacer(1, 18))

    # Charts with section header
    if radar_buf or asym_buf:
        story.append(Paragraph("<b>Visualizzazioni Grafiche</b>", heading))
        story.append(Spacer(1, 10))
        
        chart_elements = []
        if radar_buf:
            chart_elements.append(Paragraph("<i>Grafico Radar - Profilo Mobilità</i>", small))
            chart_elements.append(Spacer(1, 4))
            chart_elements.append(RLImage(io.BytesIO(radar_buf.getvalue()), 
                                         width=9 * cm, height=9 * cm, hAlign="CENTER"))
        if asym_buf:
            chart_elements.append(Spacer(1, 12))
            chart_elements.append(Paragraph("<i>Analisi Asimmetrie Bilaterali</i>", small))
            chart_elements.append(Spacer(1, 4))
            chart_elements.append(RLImage(io.BytesIO(asym_buf.getvalue()), 
                                         width=14 * cm, height=6 * cm, hAlign="CENTER"))
        
        story.append(KeepTogether(chart_elements))
        story.append(Spacer(1, 18))

    # Progress over time (if available) with enhanced header
    if progress_buf:
        story.append(PageBreak())
        
        progress_header = Table([[
            Paragraph("<b>PROGRESSIONE TEMPORALE</b>", heading_large)
        ]], colWidths=[17*cm])
        progress_header.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_SUCCESS),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
            ("LEFTPADDING", (0,0), (-1,-1), 12),
            ("RIGHTPADDING", (0,0), (-1,-1), 12),
            ("TOPPADDING", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        story.append(progress_header)
        story.append(Spacer(1, 12))
        
        story.append(RLImage(io.BytesIO(progress_buf.getvalue()), 
                            width=16 * cm, height=10 * cm, hAlign="CENTER"))
        story.append(Spacer(1, 18))

    # NEW: Injury Risk Assessment (EBM-based)
    risk_warnings = assess_injury_risk(df, session_state.get("sport", "Powerlifting"), session_state)
    
    if risk_warnings:
        story.append(PageBreak())
        story.append(Paragraph("<b>⚠️ ANALISI RISCHIO INFORTUNI & IPERMOBILITÀ</b>", heading))
        story.append(Spacer(1, 12))
        
        # Separate hypermobility from hypomobility warnings
        hypermobile_warnings = [w for w in risk_warnings if w.get('hypermobility', False)]
        hypomobile_warnings = [w for w in risk_warnings if not w.get('hypermobility', False)]
        
        # Calculate metrics like in Streamlit UI
        critical_count = sum(1 for w in hypomobile_warnings if w["priority"] == "CRITICO" or w.get("score", 100) < 31)
        hypermobile_critical_count = sum(1 for w in hypermobile_warnings if w.get("score", 0) > 100)
        moderate_hypomobile_count = len([w for w in hypomobile_warnings if 31 <= w.get("score", 0) <= 70])
        asymmetry_count = len(df[(df["SymScore"].notnull()) & (df["SymScore"] < 90)])
        total_issues = critical_count + hypermobile_critical_count + moderate_hypomobile_count + asymmetry_count
        
        # Create summary table with metrics
        summary_data = [[
            Paragraph("<b>Ipomobili<br/>Critiche</b><br/><font size=14><b>{}</b></font><br/><font size=8>&lt;31%</font>".format(critical_count), body),
            Paragraph("<b>Ipermobili<br/>Critiche</b><br/><font size=14><b>{}</b></font><br/><font size=8>&gt;100%</font>".format(hypermobile_critical_count), body),
            Paragraph("<b>Ipomobilità<br/>Moderate</b><br/><font size=14><b>{}</b></font><br/><font size=8>31-70%</font>".format(moderate_hypomobile_count), body),
            Paragraph("<b>Asimmetrie</b><br/><font size=14><b>{}</b></font><br/><font size=8>Dx/Sx</font>".format(asymmetry_count), body),
            Paragraph("<b>Totale</b><br/><font size=14><b>{}</b></font><br/><font size=8>zone</font>".format(total_issues), body)
        ]]
        
        summary_table = Table(summary_data, colWidths=[3.2*cm, 3.2*cm, 3.2*cm, 3.2*cm, 3.2*cm])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), colors.HexColor("#fee2e2")),  # Red for critical hypomobility
            ("BACKGROUND", (1,0), (1,0), colors.HexColor("#fed7aa")),  # Orange for critical hypermobility
            ("BACKGROUND", (2,0), (2,0), colors.HexColor("#fef3c7")),  # Yellow for moderate
            ("BACKGROUND", (3,0), (3,0), colors.HexColor("#dbeafe")),  # Blue for asymmetries
            ("BACKGROUND", (4,0), (4,0), colors.HexColor("#e5e7eb")),  # Gray for total
            ("BOX", (0,0), (-1,-1), 1, colors.grey),
            ("INNERGRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 16))
        
        # Show hypomobility warnings (deficit)
        if hypomobile_warnings:
            story.append(Paragraph("<b>🔴 Deficit di Mobilità (Ipomobilità):</b>", body))
            story.append(Spacer(1, 6))
            
            for warning in hypomobile_warnings[:8]:
                priority_icon = warning.get('priority_icon', '⚠️')
                priority = warning.get('priority', 'MODERATO')
                test_name = warning.get('test', 'Test sconosciuto')
                score = warning.get('score', 0)
                
                risk_text = f"{priority_icon} <b>{test_name}</b> — Score: {score:.0f}% — Priorità: {priority}"
                story.append(Paragraph(risk_text, body))
                story.append(Spacer(1, 3))
            
            if len(hypomobile_warnings) > 8:
                story.append(Paragraph(f"<i>... e altri {len(hypomobile_warnings)-8} deficit.</i>", small))
            story.append(Spacer(1, 12))
        
        # Show hypermobility warnings (excessive ROM)
        if hypermobile_warnings:
            story.append(Paragraph("<b>⚖️ Eccessiva Mobilità (Ipermobilità - Rischio Instabilità):</b>", body))
            story.append(Spacer(1, 6))
            
            for warning in hypermobile_warnings[:8]:
                priority_icon = "⚖️"
                priority = warning.get('priority', 'MODERATO')
                test_name = warning.get('test', 'Test sconosciuto')
                score = warning.get('score', 0)
                note = warning.get('note', '')
                severity = warning.get('severity', 'MILD')
                
                # Color code by severity
                if severity == "SEVERE":
                    severity_color = "SEVERA"
                    icon = "🔴"
                elif severity == "MODERATE":
                    severity_color = "MODERATA"
                    icon = "🟠"
                else:
                    severity_color = "LIEVE"
                    icon = "🟡"
                
                risk_text = f"{icon} <b>{test_name}</b> — Ipermobilità {severity_color} — Priorità: {priority}"
                story.append(Paragraph(risk_text, body))
                if note:
                    story.append(Paragraph(f"<i>{note}</i>", small))
                story.append(Spacer(1, 5))
            
            if len(hypermobile_warnings) > 8:
                story.append(Paragraph(f"<i>... e altri {len(hypermobile_warnings)-8} casi di ipermobilità.</i>", small))
            story.append(Spacer(1, 12))
            
            # Add explanation for hypermobility
            hypermobility_note = (
                "<b>Nota sull'Ipermobilità:</b> L'eccessivo range di movimento può indicare lassità "
                "capsulo-legamentosa che riduce la stabilità articolare. Studi recenti (Liaghat 2021) "
                "dimostrano che atleti ipermobili hanno rischio 3.25x maggiore di infortuni alla spalla. "
                "Raccomandazione: focus su stabilizzazione, controllo motorio e rinforzo isometrico end-range."
            )
            note_box = Table([[Paragraph(hypermobility_note, small)]], colWidths=[16*cm])
            note_box.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#e0f2fe")),  # Light blue
                ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#0ea5e9")),
                ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10),
                ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ]))
            story.append(note_box)
        
        story.append(Spacer(1, 16))
    
    # Clinical Conclusions Section with enhanced styling
    story.append(PageBreak())
    
    # Section header with colored bar
    conclusion_header = Table([[
        Paragraph("<b>CONCLUSIONI CLINICHE E PRIORITÀ DI INTERVENTO</b>", heading_large)
    ]], colWidths=[17*cm])
    conclusion_header.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), COLOR_PRIMARY),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.white),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(conclusion_header)
    story.append(Spacer(1, 16))
    
    # Pain regions in styled box
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
        pain_list = "<br/>".join([f"• {pr.capitalize()}" for pr in pain_regions])
        pain_box = Table([[Paragraph(pain_list, body)]], colWidths=[17*cm])
        pain_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_BG_CRITICAL),
            ("BOX", (0,0), (-1,-1), 1.5, COLOR_CRITICAL),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(pain_box)
    else:
        story.append(Paragraph("✅ Nessuna regione dolorosa segnalata.", body))
    story.append(Spacer(1, 18))
    
    # Asymmetries detected section
    story.append(Paragraph("<b>Asimmetrie Rilevate</b>", heading))
    asymmetries = []
    for _, row in df.iterrows():
        if pd.notna(row.get("SymScore")) and row.get("SymScore") < 90:
            test_name = row.get("Test", "Test sconosciuto")
            sym_score = row.get("SymScore", 0)
            dx_val = row.get("Dx", 0)
            sx_val = row.get("Sx", 0)
            delta = row.get("Delta", 0)
            unit = row.get("Unità", "")
            
            # Determine severity
            if sym_score < 70:
                severity = "ACUTA"
                severity_color = COLOR_CRITICAL
                bg_color = COLOR_BG_CRITICAL
                icon = "🔴"
            else:
                severity = "MODERATA"
                severity_color = colors.HexColor("#F59E0B")
                bg_color = colors.HexColor("#FEF3C7")
                icon = "🟡"
            
            asymmetries.append({
                "test": test_name,
                "sym_score": sym_score,
                "dx": dx_val,
                "sx": sx_val,
                "delta": delta,
                "unit": unit,
                "severity": severity,
                "icon": icon,
                "bg_color": bg_color
            })
    
    if asymmetries:
        asym_items = []
        for asym in asymmetries:
            asym_text = (f"{asym['icon']} <b>{asym['test']}</b> — "
                        f"Asimmetria {asym['severity']} (SymScore: {asym['sym_score']:.0f}%) — "
                        f"Δ = {asym['delta']:.1f} {asym['unit']} "
                        f"(Dx: {asym['dx']:.1f}, Sx: {asym['sx']:.1f})")
            asym_items.append(asym_text)
        
        asym_list = "<br/>".join(asym_items)
        asym_box = Table([[Paragraph(asym_list, body)]], colWidths=[17*cm])
        asym_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#FEF9C3")),
            ("BOX", (0,0), (-1,-1), 1.5, colors.HexColor("#EAB308")),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(asym_box)
    else:
        story.append(Paragraph("✅ Nessuna asimmetria significativa rilevata (tutte SymScore ≥90%).", body))
    story.append(Spacer(1, 18))

    # Clinical notes with styled boxes
    if session_state.get("clinical_notes", "").strip():
        story.append(Paragraph("<b>Note Cliniche Aggiuntive</b>", heading))
        notes_box = Table([[Paragraph(session_state["clinical_notes"], body)]], colWidths=[17*cm])
        notes_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_BG_INFO),
            ("BOX", (0,0), (-1,-1), 1, COLOR_PRIMARY),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(notes_box)
        story.append(Spacer(1, 14))

    if session_state.get("postural_observations", "").strip():
        story.append(Paragraph("<b>Osservazioni Posturali</b>", heading))
        posture_box = Table([[Paragraph(session_state["postural_observations"], body)]], colWidths=[17*cm])
        posture_box.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), COLOR_BG_INFO),
            ("BOX", (0,0), (-1,-1), 1, COLOR_PRIMARY),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(posture_box)
        story.append(Spacer(1, 14))

    # Signature section with styled box
    story.append(Spacer(1, 24))
    signature_table = Table([
        [Paragraph("<b>Firma Fisioterapista:</b> ______________________", body)],
        [Paragraph(f"<b>Data:</b> {date_str}", body)]
    ], colWidths=[17*cm])
    signature_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f9fafb")),
        ("BOX", (0,0), (-1,-1), 1, colors.grey),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(signature_table)
    
    # Bibliografia scientifica (testo molto piccolo)
    story.append(PageBreak())
    story.append(Spacer(1, 12))
    
    bib_title_style = ParagraphStyle(
        "BibTitle",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor(PRIMARY),
        alignment=1,  # Center
        spaceAfter=8
    )
    
    bib_text_style = ParagraphStyle(
        "BibText",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7,
        fontName="Helvetica",
        textColor=colors.grey,
        alignment=0,  # Left
        leading=9,
        leftIndent=5,
        rightIndent=5,
        spaceAfter=3
    )
    
    story.append(Paragraph("BIBLIOGRAFIA SCIENTIFICA", bib_title_style))
    story.append(Paragraph(
        "<i>Il presente sistema di valutazione è basato sulle seguenti evidenze scientifiche:</i>",
        bib_text_style
    ))
    story.append(Spacer(1, 6))
    
    # Add each reference
    for i, ref in enumerate(BIBLIOGRAPHY, 1):
        story.append(Paragraph(f"{i}. {ref}", bib_text_style))
    
    story.append(Spacer(1, 10))
    
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=6,
        fontName="Helvetica-Oblique",
        textColor=colors.grey,
        alignment=1,  # Center
        leading=8
    )
    
    story.append(Paragraph(
        "Le raccomandazioni cliniche fornite in questo report sono basate sulle migliori evidenze disponibili "
        "e devono essere interpretate nel contesto clinico individuale. La pratica fisioterapica richiede sempre "
        "ragionamento clinico professionale e considerazione delle caratteristiche specifiche del paziente.",
        disclaimer_style
    ))

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    buf.seek(0)
    return buf

# -----------------------------
# Main UI
# -----------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

:root {{ --primary: {PRIMARY}; }}

* {{
    font-family: 'Roboto', sans-serif !important;
}}

[data-testid="stAppViewContainer"] {{
    background-color: #32373c;
}}
[data-testid="stHeader"] {{
    background-color: #32373c;
}}
[data-testid="stSidebar"] {{
    background-color: #23282d;
}}
.main {{
    background-color: #32373c;
}}
body {{ 
    background: #32373c;
    font-family: 'Roboto', sans-serif;
}}
h1, h2, h3, h4, h5, h6 {{
    color: #ffffff !important;
    font-family: 'Roboto', sans-serif !important;
}}
p, span, div {{
    color: #e8e8e8;
    font-family: 'Roboto', sans-serif;
}}
[data-testid="stMarkdownContainer"] p {{
    color: #e8e8e8 !important;
}}
[data-testid="stMetricLabel"] {{
    color: #ffffff !important;
}}
[data-testid="stMetricValue"] {{
    color: {PRIMARY} !important;
}}
label {{
    color: #ffffff !important;
}}
.stMarkdown {{
    color: #e8e8e8 !important;
}}
/* Tabelle */
[data-testid="stDataFrame"] {{
    background-color: #23282d;
}}
[data-testid="stTable"] {{
    color: #ffffff !important;
}}
table {{
    background-color: #23282d !important;
    color: #ffffff !important;
}}
thead tr th {{
    background-color: {PRIMARY} !important;
    color: #ffffff !important;
}}
tbody tr td {{
    color: #e8e8e8 !important;
    background-color: #23282d !important;
}}
/* Menu a tendina e input */
[data-baseweb="select"] {{
    background-color: #23282d !important;
}}
[data-baseweb="select"] > div {{
    background-color: #23282d !important;
    color: #ffffff !important;
}}
.stSelectbox label {{
    color: #ffffff !important;
}}
[data-baseweb="popover"] {{
    background-color: #23282d !important;
}}
[data-baseweb="menu"] li {{
    background-color: #23282d !important;
    color: #ffffff !important;
}}
[data-baseweb="menu"] li:hover {{
    background-color: #32373c !important;
}}
input, textarea, select {{
    background-color: #23282d !important;
    color: #ffffff !important;
    border: 1px solid #50575e !important;
}}
.header-card {{ 
    background: linear-gradient(135deg, #23282d 0%, #32373c 100%); 
    padding: 20px; 
    border-radius: 15px; 
    color: #ffffff;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    border: 2px solid #50575e;
}}
.header-card h1, .header-card p {{
    color: #ffffff !important;
}}
.card {{ 
    background: #23282d; 
    padding: 15px; 
    border-radius: 12px; 
    margin-bottom: 12px; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    border-left: 4px solid {PRIMARY};
    color: #e8e8e8;
}}
.small-muted {{ color: #a8a8a8; font-size: 0.9rem; }}
.metric-card {{
    background: #23282d;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    color: #ffffff;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
}}
.stTabs [data-baseweb="tab"] {{
    padding: 12px 24px;
    background-color: #23282d;
    color: #e8e8e8;
    border: 1px solid #50575e;
    border-radius: 8px;
}}
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY};
    color: #ffffff;
    border-color: {PRIMARY};
    font-weight: 600;
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
            
            # Load latest assessment when athlete changes
            if selected_athlete != st.session_state.get("athlete"):
                history = load_athlete_history(selected_athlete)
                if history:
                    latest = history[0]  # Most recent assessment
                    
                    # Load all data into session state
                    st.session_state["athlete"] = latest.get("athlete", selected_athlete)
                    st.session_state["evaluator"] = latest.get("evaluator", "")
                    st.session_state["date"] = latest.get("date", datetime.now().strftime("%Y-%m-%d"))
                    st.session_state["sport"] = latest.get("sport", "Powerlifting")
                    st.session_state["training_frequency"] = latest.get("training_frequency", 4)
                    st.session_state["injury_history"] = latest.get("injury_history", "")
                    st.session_state["current_symptoms"] = latest.get("current_symptoms", "")
                    st.session_state["goals"] = latest.get("goals", "")
                    st.session_state["red_flags"] = latest.get("red_flags", [])
                    st.session_state["nprs"] = latest.get("nprs", 0)
                    st.session_state["psfs_activities"] = latest.get("psfs_activities", [])
                    st.session_state["pain_behavior"] = latest.get("pain_behavior", [])
                    st.session_state["aggravating_factors"] = latest.get("aggravating_factors", [])
                    st.session_state["relieving_factors"] = latest.get("relieving_factors", [])
                    st.session_state["movement_quality"] = latest.get("movement_quality", {})
                    st.session_state["clinical_notes"] = latest.get("clinical_notes", "")
                    st.session_state["postural_observations"] = latest.get("postural_observations", "")
                    
                    # Load test data
                    test_data = latest.get("data", {})
                    for test_name, test_values in test_data.items():
                        if test_name in st.session_state["vals"]:
                            st.session_state["vals"][test_name].update(test_values)
                    
                    st.rerun()
                else:
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Anamnesi", 
    "🔍 Test oggettivi", 
    "📊 Risultati", 
    "📈 Progressione",
    "📄 Report pdf"
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
            {"activity": "Squat profondo", "score": 100},
            {"activity": "Corsa", "score": 100},
            {"activity": "Overhead press", "score": 100}
        ])
        
        for i in range(3):
            col_act, col_score = st.columns([2, 1])
            with col_act:
                psfs_activities[i]["activity"] = st.text_input(f"Attività {i+1}", 
                                                               value=psfs_activities[i]["activity"],
                                                               key=f"psfs_act_{i}")
            with col_score:
                psfs_activities[i]["score"] = st.number_input(f"Capacità (%)", min_value=0, max_value=100, value=psfs_activities[i].get("score", 100),
                                                        key=f"psfs_score_{i}",
                                                        help="0=impossibile, 100=come prima del problema")
        
        st.session_state["psfs_activities"] = psfs_activities
        
        avg_psfs = np.mean([a["score"] for a in psfs_activities])
        st.metric("PSFS Medio", f"{avg_psfs:.0f}%")
        
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
        critical_count = len(df_show[df_show["Score"] < 31])
        
        with col_metric1:
            st.metric("Score Medio", f"{avg_score:.0f}%",
                     delta="Buono" if avg_score >= 71 else "Da migliorare",
                     delta_color="normal" if avg_score >= 71 else "inverse")
        
        with col_metric2:
            st.metric("Test con Dolore", f"{painful}",
                     delta="OK" if painful == 0 else "Attenzione",
                     delta_color="normal" if painful == 0 else "inverse")
        
        with col_metric3:
            st.metric("Symmetry Medio", 
                     f"{sym_mean:.0f}%" if not pd.isna(sym_mean) else "n/a",
                     delta="Simmetrico" if sym_mean >= 71 else "Asimmetrico" if not pd.isna(sym_mean) else "",
                     delta_color="normal" if sym_mean >= 71 else "inverse")
        
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
                if v >= 71:
                    color = '#d4edda'  # Verde
                elif v >= 51:
                    color = '#fef9c3'  # Giallo
                elif v >= 31:
                    color = '#fed7aa'  # Arancione
                else:
                    color = '#f8d7da'  # Rosso
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
            # Use expanded dataframe for radar to show Dx/Sx separately
            df_radar_expanded = build_df_for_radar("Valutazione Generale")
            radar_fig = plotly_radar(df_radar_expanded)
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
        
        # Clinical Conclusions
        st.markdown("### 📋 Conclusioni Cliniche e Priorità di Intervento")
        
        st.info("""💡 **Analisi basata su valutazioni oggettive**: Le seguenti conclusioni identificano le aree di limitazione, 
        i deficit di simmetria bilaterale e le priorità di intervento basate sui risultati dei test eseguiti. 
        L'ordine di priorità considera la severità della limitazione, l'impatto funzionale e il potenziale rischio.""")
        
        recommendations = generate_recommendations(df_show, st.session_state.get("sport", "Powerlifting"), st.session_state)
        
        if recommendations:
            # Summary metrics
            summary_counts = {
                "critical": len([r for r in recommendations if r["priority"] == "🚨 CRITICA"]),
                "asymmetry": len([r for r in recommendations if r["priority"] == "⚖️ ASIMMETRIA"]),
                "moderate": len([r for r in recommendations if r["priority"] == "⚠️ MODERATA"])
            }
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Valutazioni Critiche", summary_counts["critical"])
            with col2:
                st.metric("⚖️ Deficit Simmetria", summary_counts["asymmetry"])
            with col3:
                st.metric("⚠️ Limitazioni Moderate", summary_counts["moderate"])
            
            st.markdown("---")
            
            # Display each recommendation
            for rec in recommendations:
                priority_label = f"PRIORITÀ {rec['priority_num']} — {rec['priority']}"
                
                # Different display based on type
                if rec["priority"] == "⚖️ ASIMMETRIA":
                    title = f"{priority_label} — {rec['test']} (Simmetria: {rec['score']:.0f}%)"
                else:
                    title = f"{priority_label} — {rec['test']} (Score: {rec['score']:.0f}%)"
                
                with st.expander(title, expanded=(rec['priority_num'] <= 3)):
                    st.markdown(f"**📍 Regione:** {rec['region']}")
                    
                    # Metrics based on type
                    if rec["priority"] == "⚖️ ASIMMETRIA":
                        st.markdown(f"**⚖️ Differenza bilaterale:** {rec['delta']} (~{rec.get('asym_percent', 0)}% asimmetria)")
                        st.markdown(f"**👉 Lato più debole:** {rec['weaker_side'].upper()}")
                    else:
                        st.markdown(f"**📊 Deficit stimato:** ~{rec.get('deficit_percent', 0)}% sotto il riferimento")
                    
                    st.markdown(f"**🎯 Categoria:** {rec['category']}")
                    st.markdown(f"**💥 Impatto clinico:** {rec['impact']}")
                    st.markdown(f"**✅ Raccomandazione:** {rec['recommendation']}")
                    st.markdown(f"**⏱️ Timeline suggerita:** {rec['timeline']}")
        else:
            st.success("✅ Nessuna limitazione critica o asimmetria significativa rilevata. Continuare monitoraggio regolare.")
        
        st.markdown("---")
        
        # NEW: Injury Risk Assessment (EBM-based)
        st.markdown("### 🚨 Analisi Rischio Infortuni & Ipermobilità (Evidence-Based)")
        
        st.info("""📚 **Analisi basata su evidenze scientifiche**: I seguenti warning identificano sia deficit biomeccanici 
        (ipomobilità) che eccessi di mobilità (ipermobilità), entrambi associati ad aumentato rischio di infortuni. 
        Studi recenti (Liaghat 2021, Pasanen 2025) dimostrano che l'ipermobilità aumenta il rischio di 3.25x per infortuni spalla.""")
        
        risk_warnings = assess_injury_risk(df_show, st.session_state.get("sport", "Powerlifting"), st.session_state)
        
        if risk_warnings:
            # Separate hypermobility from hypomobility
            hypermobile_warnings = [w for w in risk_warnings if w.get('hypermobility', False)]
            hypomobile_warnings = [w for w in risk_warnings if not w.get('hypermobility', False)]
            
            # Risk summary metrics
            critical_count = sum(1 for w in hypomobile_warnings if w["priority"] == "CRITICO")
            high_count = sum(1 for w in hypomobile_warnings if w["priority"] == "ALTO")
            moderate_count = sum(1 for w in hypomobile_warnings if w["priority"] == "MODERATO")
            
            # Count hypermobile zones with score >100%
            hypermobile_critical_count = sum(1 for w in hypermobile_warnings if w.get("score", 0) > 100)
            
            # Count moderate hypomobility (yellow/orange zones: 31-70%)
            moderate_hypomobile_count = len([w for w in hypomobile_warnings 
                                            if 31 <= w.get("score", 0) <= 70])
            
            # Count asymmetries from df
            asymmetry_count = len(df_show[(df_show["SymScore"].notnull()) & (df_show["SymScore"] < 90)])
            
            # Total problematic zones
            total_issues = critical_count + high_count + moderate_hypomobile_count + hypermobile_critical_count + asymmetry_count
            
            col_risk1, col_risk2, col_risk3, col_risk4, col_risk5 = st.columns(5)
            with col_risk1:
                st.metric("🔴 Ipomobili Critiche", critical_count, 
                         delta="Azione immediata" if critical_count > 0 else "OK",
                         delta_color="inverse" if critical_count > 0 else "normal")
            with col_risk2:
                st.metric("🟠 Ipermobili Critiche", hypermobile_critical_count,
                         delta=">100%" if hypermobile_critical_count > 0 else "OK",
                         delta_color="inverse" if hypermobile_critical_count > 0 else "normal")
            with col_risk3:
                st.metric("🟡 Ipomobilità Moderate", moderate_hypomobile_count,
                         delta="Giallo/Arancio" if moderate_hypomobile_count > 0 else "OK",
                         delta_color="normal")
            with col_risk4:
                st.metric("⚖️ Asimmetrie", asymmetry_count,
                         delta="Bilaterale" if asymmetry_count > 0 else "OK",
                         delta_color="inverse" if asymmetry_count > 0 else "normal")
            with col_risk5:
                st.metric("📊 Totale", total_issues,
                         delta="Zone a rischio" if total_issues > 0 else "OK",
                         delta_color="inverse" if total_issues > 0 else "normal")
            
            st.markdown("---")
            
            # Display hypomobility warnings first
            if hypomobile_warnings:
                st.markdown("#### 🔴 Deficit di Mobilità (Ipomobilità)")
                
                for idx, warning in enumerate(hypomobile_warnings, 1):
                    # Color code by priority
                    if warning["priority"] == "CRITICO":
                        border_color = "#dc2626"
                        bg_color = "#fee2e2"
                    elif warning["priority"] == "ALTO":
                        border_color = "#ea580c"
                        bg_color = "#fed7aa"
                    else:
                        border_color = "#ca8a04"
                        bg_color = "#fef3c7"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style='border-left: 4px solid {border_color}; padding-left: 15px; background-color: {bg_color}20; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                            <h4>{warning['priority_icon']} <b>{warning['priority']}</b> — {warning['test']}</h4>
                            <p><b>Score:</b> {warning['score']:.0f}% | <b>Urgenza intervento:</b> {warning['urgency']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_w1, col_w2 = st.columns([1, 1])
                        
                        with col_w1:
                            st.markdown(f"**🎯 Infortuni associati:**")
                            for injury in warning['risk_injuries']:
                                st.markdown(f"- {injury}")
                            
                            if warning['sport_specific']:
                                st.markdown(f"**⚡ Impatto sport-specifico:**")
                                st.markdown(f"*{warning['sport_specific']}*")
                        
                        with col_w2:
                            st.markdown(f"**🔬 Meccanismo biomeccanico:**")
                            st.markdown(f"{warning['mechanism']}")
                            
                            st.markdown(f"**📚 Evidenza scientifica:**")
                            st.markdown(f"*{warning['evidence']}*")
                        
                        st.markdown(f"**✅ Azione raccomandata:**")
                        st.markdown(f"➡️ {warning['action']}")
                        
                        if warning.get('note'):
                            st.info(f"ℹ️ {warning['note']}")
                        
                        if warning.get('pain_present'):
                            st.error("⚠️ **DOLORE PRESENTE** - Riferimento medico raccomandato prima di procedere con carico")
                        
                        st.markdown("---")
            
            # Display hypermobility warnings
            if hypermobile_warnings:
                st.markdown("#### ⚖️ Eccessiva Mobilità (Ipermobilità - Rischio Instabilità)")
                
                st.info("""💡 **Zona Ottimale**: Un ROM eccessivo può aumentare il rischio di instabilità articolare, lussazioni e 
                perdita di controllo motorio. La ricerca scientifica conferma che l'ipermobilità è un fattore di rischio significativo 
                per infortuni (OR=3.25 per spalla). Priorità: STABILIZZAZIONE, non stretching!""")
                
                for idx, warning in enumerate(hypermobile_warnings, 1):
                    severity = warning.get('severity', 'MILD')
                    
                    # Color code by severity
                    if severity == "SEVERE":
                        border_color = "#dc2626"
                        bg_color = "#fee2e2"
                        severity_icon = "🔴"
                    elif severity == "MODERATE":
                        border_color = "#f59e0b"
                        bg_color = "#fef3c7"
                        severity_icon = "🟡"
                    else:
                        border_color = "#3b82f6"
                        bg_color = "#dbeafe"
                        severity_icon = "🔵"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style='border-left: 4px solid {border_color}; padding-left: 15px; background-color: {bg_color}20; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                            <h4>⚖️ {severity_icon} <b>Ipermobilità {severity}</b> — {warning['test']}</h4>
                            <p><b>Score:</b> {warning['score']:.0f}% | <b>Priorità:</b> {warning['priority']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_w1, col_w2 = st.columns([1, 1])
                        
                        with col_w1:
                            st.markdown(f"**🎯 Rischi associati:**")
                            for injury in warning['risk_injuries']:
                                st.markdown(f"- {injury}")
                            
                            if warning['sport_specific']:
                                st.markdown(f"**⚡ Impatto sport-specifico:**")
                                st.markdown(f"*{warning['sport_specific']}*")
                        
                        with col_w2:
                            st.markdown(f"**🔬 Meccanismo biomeccanico:**")
                            st.markdown(f"{warning['mechanism']}")
                            
                            st.markdown(f"**📚 Evidenza scientifica:**")
                            st.markdown(f"*{warning['evidence']}*")
                        
                        st.markdown(f"**✅ Raccomandazione (STABILIZZAZIONE):**")
                        st.markdown(f"➡️ {warning['action']}")
                        
                        if warning.get('note'):
                            st.warning(f"⚖️ {warning['note']}")
                        
                        if warning.get('pain_present'):
                            st.error("⚠️ **DOLORE PRESENTE** - Possibile instabilità con sintomatologia, riferimento medico urgente")
                        
                        st.markdown("---")
        else:
            st.success("✅ Nessun deficit critico identificato. Tutti i test rientrano nei range di sicurezza basati su evidenze.")

# TAB 4: PROGRESSION (CORRECTED)
with tab4:
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
                for test_name, test_data in vals.items():
                    ref = test_data.get("ref", 10.0)
                    hib = test_data.get("higher_is_better", True)
                    if test_data.get("bilat", False):
                        dx = test_data.get("Dx", 0.0)
                        sx = test_data.get("Sx", 0.0)
                        avg = (float(dx) + float(sx)) / 2.0
                        score = ability_linear(avg, ref, hib, test_name)
                        scores.append(score)
                    else:
                        val = test_data.get("Val", 0.0)
                        score = ability_linear(val, ref, hib, test_name)
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
                                    score1 = ability_linear(avg1, ref, hib, test_name)
                                    
                                    dx2 = data2.get("Dx", 0.0)
                                    sx2 = data2.get("Sx", 0.0)
                                    avg2 = (float(dx2) + float(sx2)) / 2.0
                                    score2 = ability_linear(avg2, ref, hib, test_name)
                                else:
                                    val1 = data1.get("Val", 0.0)
                                    score1 = ability_linear(val1, ref, hib, test_name)
                                    
                                    val2 = data2.get("Val", 0.0)
                                    score2 = ability_linear(val2, ref, hib, test_name)
                                
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
            for test_name, test_data in vals.items():
                ref = test_data.get("ref", 10.0)
                hib = test_data.get("higher_is_better", True)
                if test_data.get("bilat", False):
                    dx = test_data.get("Dx", 0.0)
                    sx = test_data.get("Sx", 0.0)
                    avg = (float(dx) + float(sx)) / 2.0
                    score = ability_linear(avg, ref, hib, test_name)
                    scores.append(score)
                else:
                    val = test_data.get("Val", 0.0)
                    score = ability_linear(val, ref, hib, test_name)
                    scores.append(score)
            
            avg_score = np.mean(scores) if scores else 0
            
            col_score1, col_score2 = st.columns(2)
            with col_score1:
                st.metric("Score Medio", f"{avg_score:.0f}%")
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
                    score = ability_linear(avg, ref, hib, test_name)
                else:
                    val = test_data.get("Val", 0.0)
                    score = ability_linear(val, ref, hib, test_name)
                
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

# TAB 5: PDF REPORT
with tab5:
    st.markdown("### 📄 Generazione Report PDF")
    
    df_show = build_df("Valutazione Generale")
    
    if df_show.empty:
        st.warning("⚠️ Nessun dato disponibile per generare il report. Compilare prima i test.")
    else:
        st.info("Il report PDF includerà: anamnesi, red flags, scale funzionali, risultati test, grafici, raccomandazioni e note cliniche.")
        
        # Prepare data for PDF
        try:
            # Use expanded dataframe for radar to show Dx/Sx separately
            df_radar_expanded = build_df_for_radar("Valutazione Generale")
            df_radar = df_radar_expanded[df_radar_expanded["Score"].notnull()].copy()
            radar_buf = radar_plot_matplotlib(df_radar, title="Punteggi Test (0-140%)") if len(df_radar) >= 3 else None
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
