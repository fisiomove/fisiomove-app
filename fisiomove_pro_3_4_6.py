# Fisiomove Pro 3.4.6
# - Fix: StreamlitDuplicateElementKey (chiavi widget univoche per tutte le sezioni, anche in "Valutazione Generale")
# - Tema: blu coerente col logo, rimosso testo "Clinical Dark – Mobilità & Simmetria"
# - Mantiene: autosave/autoload, reset sezione, radar, simmetria, PDF con header logo, tutorial EBM

import io, os, json, sqlite3, datetime as dt, random, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- THEME / CONST -----------------
PRIMARY = "#1E6CF4"   # blu logo
BG = "#0E1117"        # dark neutro
CARD = "#171A21"
TEXT = "#EAEAEA"
OK = "#16a34a"; MID = "#f59e0b"; LOW = "#dc2626"

DB_PATH = "fisiomove.db"

# LOGHI (ordine di preferenza)
LOGOS = [
    "EAE1A98B-110F-4A45-A857-C67FFC148375.jpeg",  # immagine inviata
    "logo 2600x1000.jpg",
    "logo_fisiomove.png",
]

st.set_page_config(page_title="Fisiomove Pro 3.4.6", layout="wide")

# ----------------- GLOBAL CSS -----------------
DARK_CSS = f"""
<style>
html, body, .stApp {{ background: {BG} !important; color: {TEXT} !important; }}
.block-container {{ padding-top: 0.6rem; padding-bottom: 5.2rem; max-width: 860px; }}
h1, h2, h3, h4, h5 {{ color: {PRIMARY} !important; letter-spacing: 0.2px; }}
.stButton>button {{ width: 100%; border-radius: 12px; padding: 0.75rem 1rem; font-size: 1.05rem; background: {PRIMARY}22; color: {TEXT}; border: 1px solid {PRIMARY}55; }}
.stButton>button:hover {{ background: {PRIMARY}44; }}
.card {{ border: 1px solid #2A2F3A; border-radius: 14px; padding: 12px; background: {CARD}; box-shadow: 0 2px 10px rgba(0,0,0,0.25); }}
.badge {{ display:inline-block; padding: 6px 10px; border-radius: 10px; font-weight:700; }}
.stApp header {{ visibility: hidden; height: 0; }} footer {{ visibility: hidden; height: 0; }}
.dataframe td, .dataframe th {{ color: {TEXT}; }}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# ----------------- DATASETS -----------------
# Schema: (name, unit, ref, desc, bilateral: bool)
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10, "Dorsiflessione in carico (WB).", True),
        ("Passive Hip Flexion", "°", 120, "Flessione passiva anca.", True),
        ("Hip Rotation (flexed 90°)", "°", 40, "Rotazione anca a 90° flessione (IR/ER composite).", True),
        ("Thoracic Extension (T4- T12)", "°", 30, "Estensione toracica segmentale.", False),
        ("Shoulder ER (adducted, low-bar)", "°", 70, "Posizione low-bar bilanciere.", True),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180, "Flessione gleno-omerale in supino.", True),
        ("External Rotation (90 deg abd)", "°", 90, "ER spalla a 90° abduzione.", True),
        ("Thoracic Extension (T4- T12)", "°", 30, "Arco toracico per panca.", False),
        ("Pectoralis Minor Length", "cm", 10, "Spazio acromiale/posteriore da piano.", True),
        ("Wall Angel (distance)", "cm", 10, "Controllo scapolare a parete.", True),
        ("Thomas Test (modified)", "°", 10, "Estensione anca per postura lombo-pelvica.", True),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90, "Estensibilita' ischiocrurali.", True),
        ("Straight Leg Raise (SLR)", "°", 90, "Catena posteriore (neuro/muscolare).", True),
        ("Weight Bearing Lunge Test", "cm", 10, "Dorsiflessione in carico.", True),
        ("Modified Schober (lumbar)", "cm", 5, "Mobilita' lombare in flessione.", False),
        ("Sorensen Endurance", "sec", 180, "Endurance estensori lombari.", False),
    ],
    "Neurodinamica": [
        ("SLR (sensibile)", "°", 70, "Range con sintomi controllati.", True),
        ("PNLT tibiale (bias)", "°", 40, "Differenziazione sensibile tibiale.", True),
        ("PNLT peroneale (bias)", "°", 40, "Differenziazione sensibile peroneale.", True),
        ("ULNT1 (n. mediano)", "°", 100, "Range con differenziazione scapolare.", True),
    ],
}
ALL_ORDER = ["Squat", "Panca", "Deadlift", "Neurodinamica"]

TUTORIAL = {
    "Weight Bearing Lunge Test": "Affondo verso parete, tallone a terra; misura distanza punta-parete con ginocchio a contatto.",
    "Passive Hip Flexion": "Supino, bacino stabilizzato; arresto prima di compensi lombari.",
    "Hip Rotation (flexed 90°)": "Supino, anca/gomito 90°; IR/ER mantenendo bacino neutro.",
    "Thoracic Extension (T4- T12)": "Inclinometro tra T4 e T12; estensione attiva massima.",
    "Shoulder ER (adducted, low-bar)": "Braccio addotto, gomito 90°; ER per posizione low-bar.",
    "Shoulder Flexion (supine)": "Supino; elevazione completa, controlla coste.",
    "External Rotation (90 deg abd)": "Supino; spalla abdotta 90°, gomito 90°, ER fino a fine corsa.",
    "Pectoralis Minor Length": "Supino rilassato; distanza acromion-piano (cm).",
    "Wall Angel (distance)": "Schiena/avambracci a parete; misura distanza (cm).",
    "Thomas Test (modified)": "Bordo lettino; una coscia al petto, controlaterale pendola; misura estensione anca.",
    "Active Knee Extension (AKE)": "Supino, anca 90°; estendi ginocchio e misura deficit.",
    "Straight Leg Raise (SLR)": "Supino; elevazione arto esteso fino a tensione controllata.",
    "Modified Schober (lumbar)": "Segna L5 e 10 cm sopra; misura incremento in flessione.",
    "Sorensen Endurance": "Prono su bordo lettino; mantieni tronco orizzontale fino a cedimento.",
    "SLR (sensibile)": "SLR con differenziazione (dorsiflessione/piede).",
    "PNLT tibiale (bias)": "Dorsiflessione+eversione; differenziazione sintomi tibiale.",
    "PNLT peroneale (bias)": "Plantarflessione+inversione; differenziazione sintomi peroneale.",
    "ULNT1 (n. mediano)": "Abduzione+ER spalla, estensione gomito/polso; differenziazione scapolare."
}

# ----------------- UTILS -----------------
def to_ascii(s: str) -> str:
    if not isinstance(s, str): return s
    table = {"–":"-","—":"-","’":"'","‘":"'","“":'"',"”":'"',"•":"-","°":" gradi",
             "→":"->","↔":"<->","≥":">=","≤":"<=","Δ":"Delta","✓":"-","×":"x"}
    for k,v in table.items(): s = s.replace(k,v)
    return s

def is_degree_unit(unit: str) -> bool:
    u = (unit or "").lower()
    return ("°" in unit) or ("deg" in u) or ("degree" in u) or ("gradi" in u)

def score_linear(value: float, ref: float) -> float:
    try:
        v = float(value); r = float(ref)
        if r <= 0: return 0.0
        return float(np.clip((v/r) * 10.0, 0.0, 10.0))
    except Exception:
        return 0.0

def symmetry_score(dx: float, sx: float, unit: str) -> float:
    dx = float(dx); sx = float(sx); diff = abs(dx - sx)
    if unit == "%": scale = 20.0
    elif unit == "cm": scale = 8.0
    elif is_degree_unit(unit): scale = 20.0
    else: scale = 10.0
    return float(10.0 * max(0.0, 1.0 - min(diff, scale)/scale))

def color_by_score(s: float) -> str:
    if s < 4: return LOW
    if s <= 7: return MID
    return OK

def badge(text: str, color_hex: str) -> str:
    return f"<span class='badge' style='background:{color_hex}22;color:{color_hex};'>{text}</span>"

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "")

def key_id(sec: str, name: str, suffix: str = "") -> str:
    # Chiave UNIVOCA per ogni widget: sezione + nome test + suffisso (Dx/Sx/Val)
    return f"key_{sanitize(sec)}__{sanitize(name)}__{suffix}"

# Radar (plotly per UI)
def radar_plot(labels, scores, title):
    fig = go.Figure()
    if len(labels) == 0:
        fig.update_layout(title="Nessun test compilato", polar=dict(radialaxis=dict(visible=True, range=[0,10])), height=320,
                          paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT))
        return fig
    labels_c = list(labels) + [labels[0]]
    scores_c = list(scores) + [scores[0]]
    ideal = [10]*len(labels); ideal_c = ideal + [ideal[0]]
    fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target 10/10",
                                  line=dict(width=1, dash="dot", color="#9aa4b2")))
    fig.add_trace(go.Scatterpolar(r=scores_c, theta=labels_c, name="Atleta",
                                  fill="toself", line=dict(width=2, color=PRIMARY)))
    fig.update_layout(
        title=title, showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center"),
        polar=dict(bgcolor=BG, radialaxis=dict(visible=True, range=[0,10], tickmode="linear", dtick=1, gridcolor="#2A2F3A", tickfont=dict(color=TEXT)),
                   angularaxis=dict(tickfont=dict(color=TEXT))),
        margin=dict(l=10, r=10, t=40, b=10), height=460, paper_bgcolor=BG, font=dict(color=TEXT)
    )
    return fig

# Radar for PDF (matplotlib to PNG buffer)
def radar_png_buffer(labels, scores):
    N = len(labels)
    buf = io.BytesIO()
    if N == 0: return buf
    if N == 1:
        labels = labels * 2; scores = scores * 2; N = 2
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]
    scores_c = list(scores) + [scores[0]]
    ideal_c = [10]*N + [10]
    fig, ax = plt.subplots(figsize=(5.2,5.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, linewidth=0.5, color="#cccccc", alpha=0.9); ax.spines["polar"].set_color("#999999")
    ax.plot(angles_c, ideal_c, color="#777777", linewidth=1, linestyle="dotted", label="Target 10/10")
    ax.plot(angles_c, scores_c, color=PRIMARY, linewidth=2, label="Atleta")
    ax.fill(angles_c, scores_c, color="#3B82F644")
    ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10]); ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=8)
    for angle, r in zip(angles, scores):
        ax.text(angle, r + 0.25, f"{r:.1f}", ha="center", va="center", fontsize=7, color="#0b2758")
    fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf

# Symmetry bars (UI)
def symmetry_bar_chart(sym_df):
    fig = go.Figure()
    if sym_df is None or sym_df.empty:
        fig.update_layout(title="Nessun test bilaterale inserito", paper_bgcolor=BG, font=dict(color=TEXT), height=300)
        return fig
    fig.add_trace(go.Bar(
        x=sym_df["SymScore"], y=sym_df["Test"], orientation='h',
        marker=dict(color=[color_by_score(s) for s in sym_df["SymScore"]]),
        text=[f"{s:.1f}/10 (Delta {d})" for s, d in zip(sym_df["SymScore"], sym_df["DeltaStr"])],
        textposition="outside"
    ))
    fig.update_layout(xaxis=dict(range=[0,10], gridcolor="#2A2F3A", tickfont=dict(color=TEXT)),
                      yaxis=dict(tickfont=dict(color=TEXT)),
                      plot_bgcolor=BG, paper_bgcolor=BG, height=420, margin=dict(l=10,r=10,t=40,b=10),
                      title="Simmetria Dx/Sx (0–10)")
    return fig

# Symmetry bars for PDF (matplotlib) -> PNG buffer
def symmetry_png_buffer(sym_df):
    buf = io.BytesIO()
    if sym_df is None or sym_df.empty: return buf
    fig, ax = plt.subplots(figsize=(6.2,4.2))
    labels = list(sym_df["Test"])[::-1]
    scores = list(sym_df["SymScore"])[::-1]
    colors_bar = [("#16a34a" if s>7 else "#f59e0b" if s>=4 else "#dc2626") for s in scores]
    ax.barh(labels, scores, color=colors_bar)
    for i, s in enumerate(scores):
        ax.text(s+0.1, i, f"{s:.1f}", va='center', fontsize=8)
    ax.set_xlim(0,10); ax.set_xlabel("Simmetria 0-10")
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig); buf.seek(0); return buf

# EBM COMMENT ENGINE (ASCII-safe)
def ebm_explain_row(test_name, unit, ref, val_avg, dx=None, sx=None):
    lines = []
    if val_avg <= 0: return "Non disponibile."
    perf = "ottimo" if val_avg >= ref*0.9 else "nella norma" if val_avg >= ref*0.7 else "ridotto"
    if "Lunge" in test_name:
        lines += [
            f"- Dorsiflessione in carico: {perf}.",
            "- Valori ridotti spostano il carico sull'avampiede e anticipano la retroversione del bacino nello squat.",
            "- Possibile aumento di stress femoro-rotuleo e perdita di stabilita' del tallone.",
            "Riferimenti: Barton 2012 JOSPT; Rabin 2015 Phys Ther."
        ]
    elif "Thoracic Extension" in test_name:
        lines += [
            f"- Estensione toracica: {perf}.",
            "- Riduzioni favoriscono inclinazione del tronco e compensi scapolo-omerali in squat/panca.",
            "Riferimenti: Wilke 2020 Front Physiol."
        ]
    elif "AKE" in test_name or "SLR" in test_name:
        lines += [
            f"- Estensibilita' catena posteriore: {perf}.",
            "- Limiti riducono hinge efficiente e aumentano strain neurale/muscolare.",
            "Riferimenti: Malliaras 2015 Sports Med; Bohannon 1999 Phys Ther."
        ]
    elif "ER" in test_name and "Shoulder" in test_name:
        lines += [
            f"- Rotazione esterna spalla: {perf}.",
            "- Limitazioni in panca/low-bar aumentano stress anteriore e alterano controllo scapolare.",
            "Riferimenti: Ludewig 2009 Phys Ther; Green 2017 JSES."
        ]
    elif "Pectoralis Minor" in test_name:
        lines += [
            f"- Pettorale minore: {perf}.",
            "- Accorciamento associato a tilt anteriore scapola e riduzione spazio subacromiale.",
            "Riferimenti: Borstad 2006 JSES."
        ]
    elif "Schober" in test_name:
        lines += [
            f"- Mobilita' lombare: {perf}.",
            "- Valori bassi riducono adattabilita' e spostano carico su anche/posteriore.",
            "Riferimenti: Macrae 1990 Spine."
        ]
    elif "Sorensen" in test_name:
        lines += [
            f"- Endurance estensori lombari: {perf}.",
            "- Ridotta endurance = minore tolleranza a carichi prolungati in deadlift.",
            "Riferimenti: Biering-Sorensen 1984 Spine; McGill 2001 Arch PMR."
        ]
    elif "Thomas" in test_name:
        lines += [
            f"- Estensione d'anca: {perf}.",
            "- Riduzioni aumentano tilt anteriore o iperlordosi in panca.",
            "Riferimenti: Harvey 1998 PRI."
        ]
    else:
        lines += [f"- Prestazione {perf} rispetto al riferimento."]
    if dx is not None and sx is not None:
        d = abs(dx - sx)
        if (is_degree_unit(unit) and d >= 20) or (unit == "cm" and d >= 4):
            lines += ["- Nota: asimmetria significativa. Possibile compenso e rischio overuse (Fousekis 2010 BJSM; Bishop 2018 JSCR)."]
    return to_ascii("\n".join(lines))

# ----------------- STORAGE -----------------
def init_db():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        athlete TEXT, section TEXT, date TEXT, data_json TEXT
    );""")
    con.commit(); con.close()
init_db()

def save_session(athlete: str, section: str, date_str: str, df: pd.DataFrame):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("INSERT INTO sessions (athlete, section, date, data_json) VALUES (?, ?, ?, ?);",
                (athlete, section, date_str, json.dumps(df.to_dict(orient="records"), ensure_ascii=False)))
    con.commit(); con.close()

def load_sessions(athlete: str, section: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT date, data_json FROM sessions WHERE athlete=? AND section=? ORDER BY date DESC;", (athlete, section))
    rows = cur.fetchall(); con.close()
    if not rows: return pd.DataFrame(columns=["date","data_json"])
    return pd.DataFrame([{"date": r[0], "data_json": r[1]} for r in rows])

def latest_session_values(athlete: str, section: str):
    df = load_sessions(athlete, section)
    if df.empty: return {}
    rec = df.iloc[0]
    try:
        data = json.loads(rec["data_json"])
    except Exception:
        return {}
    out = {}
    for row in data:
        name = row.get("Test")
        if not name: continue
        out[name] = {
            "Val": float(row.get("Valore", 0) or 0),
            "Dx": float(row.get("Dx", 0) or 0),
            "Sx": float(row.get("Sx", 0) or 0)
        }
    return out

def export_session_json(athlete: str, section: str, date_str: str, df: pd.DataFrame) -> bytes:
    payload = {"athlete": athlete, "section": section, "date": date_str, "records": df.to_dict(orient="records")}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

# ----------------- STATE -----------------
SECTIONS = list(TESTS.keys()) + ["Valutazione Generale"]
st.session_state.setdefault("page", "Valutazione")
st.session_state.setdefault("athlete", "Mario Rossi")
st.session_state.setdefault("evaluator", "Dott. Alessandro Ferreri - Fisioterapista")
st.session_state.setdefault("date", dt.date.today().strftime("%Y-%m-%d"))
st.session_state.setdefault("section", "Squat")
st.session_state.setdefault("autosave", True)

# ----------------- HEADER -----------------
def load_logo_bytes():
    for p in LOGOS:
        try:
            with open(p, "rb") as f: return f.read()
        except Exception:
            continue
    return None
LOGO_BYTES = load_logo_bytes()

st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if LOGO_BYTES:
    st.image(LOGO_BYTES, use_container_width=True)
else:
    st.markdown("<div style='color:#9aa4b2;font-size:28px;font-weight:700;'>Fisiomove</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;margin-top:0'>Fisiomove Pro 3.4.6</h2>", unsafe_allow_html=True)

# ----------------- INPUT CARD -----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.session_state["athlete"] = st.text_input("Atleta", st.session_state["athlete"])
with col2:
    st.session_state["evaluator"] = st.text_input("Valutatore", st.session_state["evaluator"])
col3, col4 = st.columns(2)
with col3:
    st.session_state["date"] = st.date_input("Data", value=dt.date.fromisoformat(st.session_state["date"])).strftime("%Y-%m-%d")
with col4:
    st.session_state["section"] = st.selectbox("Movimento testato", options=SECTIONS, index=SECTIONS.index(st.session_state["section"]) if st.session_state["section"] in SECTIONS else 0)
st.caption("Suggerimento: imposta il movimento prima di inserire i valori.")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- LOGIC -----------------
def iterate_tests_for_section(section_name: str):
    if section_name != "Valutazione Generale":
        for t in TESTS[section_name]: yield section_name, t
    else:
        for sec in ALL_ORDER:
            for t in TESTS[sec]:
                yield sec, t

def rand_val(ref, unit):
    if unit == "sec":
        base = ref * random.uniform(0.8, 1.1)
    elif unit == "%":
        base = random.uniform(60, 95)
    else:
        base = ref * random.uniform(0.75, 1.15)
    return max(0.0, float(base))

def rand_pair(ref, unit):
    v = rand_val(ref, unit)
    if is_degree_unit(unit) or unit == "cm":
        delta = random.uniform(3, 10)
    else:
        delta = random.uniform(2, 6)
    if random.random() < 0.5:
        dx, sx = v + delta/2, v - delta/2
    else:
        dx, sx = v - delta/2, v + delta/2
    return max(0.0, dx), max(0.0, sx)

def build_dataframe_from_state(section, per_test_values):
    rows = []; sym_rows = []
    for sec_name, (name, unit, ref, desc, bilateral) in iterate_tests_for_section(section):
        if bilateral:
            dx = per_test_values[name]["Dx"]; sx = per_test_values[name]["Sx"]
            val_avg = (dx + sx) / 2.0
            ability = score_linear(val_avg, max(ref, 1))
            sym = symmetry_score(dx, sx, unit)
            delta = abs(dx - sx); delta_str = f"{delta:.1f}{unit}"
            comment = ebm_explain_row(name, unit, ref, val_avg, dx, sx)
            rows.append([name, unit, ref, val_avg, ability, dx, sx, delta, sym, comment, sec_name])
            sym_rows.append({"Test": name, "Dx": dx, "Sx": sx, "Unità": unit, "Delta": delta, "DeltaStr": delta_str, "SymScore": sym})
        else:
            val = per_test_values[name]["Val"]
            ability = score_linear(val, max(ref, 1))
            rows.append([name, unit, ref, val, ability, None, None, None, None, ebm_explain_row(name, unit, ref, val), sec_name])
    df_cols = ["Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM","Sezione"]
    df = pd.DataFrame(rows, columns=df_cols)
    sym_df = pd.DataFrame(sym_rows, columns=["Test","Dx","Sx","Unità","Delta","DeltaStr","SymScore"])
    return df, sym_df

def load_or_seed_values(section):
    athlete = st.session_state["athlete"]
    if section == "Valutazione Generale":
        agg = {}
        for sec in ALL_ORDER:
            saved = latest_session_values(athlete, sec)
            for _, (name, unit, ref, desc, bilateral) in iterate_tests_for_section(sec):
                if name in saved:
                    agg[name] = saved[name]
                else:
                    if bilateral:
                        dx, sx = rand_pair(ref, unit)
                        agg[name] = {"Val": 0.0, "Dx": dx, "Sx": sx}
                    else:
                        agg[name] = {"Val": rand_val(ref, unit), "Dx": 0.0, "Sx": 0.0}
        return agg
    saved = latest_session_values(athlete, section)
    result = {}
    for _, (name, unit, ref, desc, bilateral) in iterate_tests_for_section(section):
        if name in saved and any([saved[name].get("Val",0)>0, saved[name].get("Dx",0)>0, saved[name].get("Sx",0)>0]):
            result[name] = saved[name]
        else:
            if bilateral:
                dx, sx = rand_pair(ref, unit)
                result[name] = {"Val": 0.0, "Dx": dx, "Sx": sx}
            else:
                result[name] = {"Val": rand_val(ref, unit), "Dx": 0.0, "Sx": 0.0}
    return result

def page_valutazione():
    section = st.session_state["section"]
    st.markdown("### Valutazione")
    st.write("Per test bilaterali inserisci **Dx** e **Sx**. Il radar usa la media normalizzata (0–10). La sezione Simmetria calcola Delta e punteggio (0–10).")

    # --- load or seed current values in session ---
    key_seed = f"_seed_ready_{section}"
    if key_seed not in st.session_state or not st.session_state[key_seed]:
        st.session_state[f"per_test_{section}"] = load_or_seed_values(section)
        st.session_state[key_seed] = True
    per_test_values = st.session_state.get(f"per_test_{section}", {})

    rows = []; sym_rows = []

    for sec_name, (name, unit, ref, desc, bilateral) in iterate_tests_for_section(section):
        st.markdown(f"**{name}** · *{desc}* <span style='opacity:0.6'>(sezione: {sec_name})</span>", unsafe_allow_html=True)
        with st.expander("Standard del test", expanded=False):
            st.caption(TUTORIAL.get(name, "-"))
        maxv = float(ref) * 1.5 if unit == "sec" else (100.0 if unit == "%" else float(ref) * 1.3)
        if bilateral:
            c1, c2 = st.columns(2)
            with c1:
                dx_val = per_test_values[name]["Dx"]
                dx = st.slider(f"Dx ({unit}, rif. {ref})", 0.0, float(maxv), float(dx_val), 0.1, key=key_id(sec_name, name, "Dx"))
            with c2:
                sx_val = per_test_values[name]["Sx"]
                sx = st.slider(f"Sx ({unit}, rif. {ref})", 0.0, float(maxv), float(sx_val), 0.1, key=key_id(sec_name, name, "Sx"))
            # update in state
            per_test_values[name]["Dx"] = float(dx); per_test_values[name]["Sx"] = float(sx)
            val_avg = (dx + sx) / 2.0
            ability = score_linear(val_avg, max(ref, 1))
            st.markdown(badge(f"Abilita': {ability:.1f}/10", color_by_score(ability)), unsafe_allow_html=True)
            sym = symmetry_score(dx, sx, unit)
            delta = abs(dx - sx); delta_str = f"{delta:.1f}{unit}"
            st.caption(f"Delta Simmetria: {delta_str} -> Punteggio {sym:.1f}/10")
            sym_rows.append({"Test": name, "Dx": dx, "Sx": sx, "Unità": unit, "Delta": delta, "DeltaStr": delta_str, "SymScore": sym})
            comment = ebm_explain_row(name, unit, ref, val_avg, dx, sx)
            rows.append([name, unit, ref, val_avg, ability, dx, sx, delta, sym, comment, sec_name])
        else:
            val0 = per_test_values[name]["Val"]
            val = st.slider(f"{unit} (rif. {ref})", 0.0, float(maxv), float(val0), 0.1, key=key_id(sec_name, name, "Val"))
            per_test_values[name]["Val"] = float(val)
            ability = score_linear(val, max(ref, 1))
            st.markdown(badge(f"Punteggio: {ability:.1f}/10", color_by_score(ability)), unsafe_allow_html=True)
            rows.append([name, unit, ref, val, ability, None, None, None, None, ebm_explain_row(name, unit, ref, val), sec_name])
        st.divider()

    df_cols = ["Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM","Sezione"]
    df = pd.DataFrame(rows, columns=df_cols)
    sym_df = pd.DataFrame(sym_rows, columns=["Test","Dx","Sx","Unità","Delta","DeltaStr","SymScore"])

    df_meas = df[df["Valore"] > 0]
    pct = int((len(df_meas) / len(df) * 100)) if not df.empty else 0
    st.progress(pct); st.caption(f"Completati: {len(df_meas)} su {len(df)} test ({pct}%)")

    # Radar
    st.markdown("### Radar abilita' (media normalizzata)")
    if section == "Valutazione Generale":
        labels = [f"{r['Test']} [{r['Sezione']}]" for _, r in df_meas.iterrows()]
    else:
        labels = df_meas["Test"].tolist()
    scores = df_meas["Score"].tolist()
    st.plotly_chart(radar_plot(labels, scores, f"{section} – Abilita' (0–10)"), use_container_width=True)

    # Symmetry bars
    st.markdown("### Simmetria Dx/Sx")
    st.plotly_chart(symmetry_bar_chart(sym_df), use_container_width=True)

    # EBM Table
    st.markdown("### Spiegazione clinica (EBM)")
    show_cols = ["Sezione","Test","Unità","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM"]
    df_show = df[show_cols].copy()
    df_show["CommentoEBM"] = df_show["CommentoEBM"].apply(to_ascii)
    st.dataframe(df_show.style.format({"Score":"{:.1f}","SymScore":"{:.1f}","Valore":"{:.1f}","Dx":"{:.1f}","Sx":"{:.1f}","Delta":"{:.1f}"}), use_container_width=True)

    # Autosave
    if st.session_state.get("autosave", True) and not df_meas.empty and section != "Valutazione Generale":
        save_session(st.session_state["athlete"], section, st.session_state["date"], df_show)

    # Actions
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("Salva"):
            save_session(st.session_state["athlete"], section, st.session_state["date"], df_show)
            st.success("Sessione salvata (storico in 'Storico').")
    with c2:
        if st.button("JSON"):
            js = export_session_json(st.session_state["athlete"], section, st.session_state["date"], df_show)
            st.download_button("Scarica JSON", data=js, file_name=f"{st.session_state['athlete'].replace(' ','')}_{section}_{st.session_state['date']}.json", mime="application/json")
    with c3:
        if st.button("PDF", disabled=df_meas.empty):
            pdf = pdf_single_section(LOGO_BYTES, st.session_state["athlete"], st.session_state["evaluator"], st.session_state["date"], section, df_show, sym_df, labels, scores)
            st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{section}_{st.session_state['athlete'].replace(' ','')}_{st.session_state['date']}.pdf", mime="application/pdf")
    with c4:
        if st.button("Reset sezione"):
            st.session_state.pop(f"_seed_ready_{section}", None)
            st.session_state.pop(f"per_test_{section}", None)
            st.experimental_rerun()

# ----------------- PDF -----------------
def _header_canvas(logo_bytes):
    img = ImageReader(io.BytesIO(logo_bytes)) if logo_bytes else None
    def _draw(canvas, doc):
        if img:
            page_w, page_h = A4
            w = page_w - 2*cm
            h = w * 0.22
            x = (page_w - w) / 2.0
            y = page_h - h - 10
            canvas.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0]-36, 18, f"Fisiomove Pro – {dt.date.today().isoformat()}")
    return _draw

def pdf_single_section(logo_bytes, athlete, evaluator, date_str, section_name, df_show, sym_df, labels, scores):
    disp = df_show[df_show["Valore"] > 0].copy()
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf_pdf, pagesize=A4,
        rightMargin=1.6*cm, leftMargin=1.6*cm,
        topMargin=3.6*cm, bottomMargin=1.6*cm
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=14)
    story = []

    story.append(Paragraph("<b>Report Valutazione</b>", title)); story.append(Spacer(1, 6))
    meta_tbl = Table([["Atleta", to_ascii(athlete), "Data", date_str, "Valutatore", to_ascii(evaluator)]],
                     colWidths=[2*cm, 6*cm, 2*cm, 3*cm, 2.5*cm, 4*cm])
    meta_tbl.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                                  ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(meta_tbl); story.append(Spacer(1, 8))
    story.append(Paragraph(f"Sezione: <b>{to_ascii(section_name)}</b>", styles['Heading3']))

    if disp.empty:
        story.append(Paragraph("Nessun test compilato (valori > 0).", normal))
    else:
        cols_order = ["Sezione","Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore"]
        disp_pdf = disp[[c for c in cols_order if c in disp.columns]].copy()
        if "Unità" in disp_pdf.columns: disp_pdf.rename(columns={"Unità":"Unita'"}, inplace=True)
        for c in ["Score","SymScore","Valore","Dx","Sx","Delta"]:
            if c in disp_pdf.columns: disp_pdf[c] = pd.to_numeric(disp_pdf[c], errors="coerce").round(1)
        tdata = [list(disp_pdf.columns)] + disp_pdf.fillna("").values.tolist()
        tbl = Table(tdata, colWidths=[2.8*cm,5.4*cm,1.6*cm,1.6*cm,1.6*cm,1.6*cm,1.4*cm,1.4*cm,1.6*cm,2.0*cm][:len(disp_pdf.columns)])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("ALIGN",(3,1),(-1,-1),"CENTER")
        ]))
        story.append(tbl); story.append(Spacer(1, 8))

        rbuf = radar_png_buffer(labels, [float(s) for s in scores])
        try:
            story.append(Image(io.BytesIO(rbuf.getvalue()), width=13.0*cm, height=13.0*cm))
            story.append(Paragraph("Radar abilita' (0–10)", styles['Normal'])); story.append(Spacer(1, 8))
        except Exception as e:
            story.append(Paragraph(f"Radar non disponibile ({to_ascii(str(e))}).", normal)); story.append(Spacer(1, 6))

        if not sym_df.empty:
            sym_disp = sym_df[["Test","Dx","Sx","Delta","SymScore"]].copy()
            for c in ["Dx","Sx","Delta","SymScore"]:
                if c in sym_disp.columns: sym_disp[c] = pd.to_numeric(sym_disp[c], errors="coerce").round(1)
            t2 = [["Test","Dx","Sx","Delta","Simmetria (0–10)"]] + sym_disp.fillna("").values.tolist()
            tbl2 = Table(t2, colWidths=[7.0*cm,2.0*cm,2.0*cm,2.0*cm,3.0*cm])
            tbl2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),
                                      ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
            story.append(tbl2); story.append(Spacer(1, 6))

            sbuf = symmetry_png_buffer(sym_disp)
            try:
                story.append(Image(io.BytesIO(sbuf.getvalue()), width=14.0*cm, height=9.0*cm))
                story.append(Paragraph("Barre di simmetria Dx/Sx", styles['Normal']))
            except Exception:
                pass

        story.append(Spacer(1, 8))
        story.append(Paragraph("Feedback clinico (EBM)", styles['Heading3']))
        for _, r in disp.iterrows():
            bullets = [to_ascii(x) for x in str(r.get("CommentoEBM","")).split("\n") if x.strip()]
            flow = ListFlowable([ListItem(Paragraph(b, styles['Normal'])) for b in bullets],
                                bulletType='bullet', leftIndent=12)
            story.append(Paragraph(f"<b>{to_ascii(str(r.get('Test','')))}</b>", styles['Normal']))
            story.append(flow); story.append(Spacer(1, 4))

    header = _header_canvas(logo_bytes) if logo_bytes else None
    if header:
        doc.build(story, onFirstPage=header, onLaterPages=header)
    else:
        doc.build(story)
    buf_pdf.seek(0); return buf_pdf

# ----------------- RIEPILOGO / STORICO -----------------
def page_riepilogo():
    section = st.session_state["section"]
    athlete = st.session_state["athlete"]
    data_hist = load_sessions(athlete, section)
    if data_hist.empty:
        st.info("Nessuna sessione salvata per questo atleta/sezione."); return
    dates = data_hist["date"].tolist()
    pick = st.selectbox("Seleziona una valutazione salvata", dates, index=0)
    rec = data_hist[data_hist["date"]==pick].iloc[0]
    df = pd.DataFrame(json.loads(rec["data_json"]))
    df_meas = df[df.get("Valore",0) > 0]
    labels = [f"{r['Test']} [{r['Sezione']}]" for _, r in df_meas.iterrows()] if "Sezione" in df_meas.columns else df_meas["Test"].tolist()
    scores = pd.to_numeric(df_meas.get("Score", pd.Series()), errors="coerce").fillna(0).tolist()
    st.markdown("### Radar abilita'")
    st.plotly_chart(radar_plot(labels, scores, f"{section} – Abilita' (0–10) @ {pick}"), use_container_width=True)
    sym_df = df.dropna(subset=["Dx","Sx"], how="all").copy()
    if not sym_df.empty:
        sym_df["Dx"] = pd.to_numeric(sym_df.get("Dx", pd.Series()), errors="coerce").fillna(0)
        sym_df["Sx"] = pd.to_numeric(sym_df.get("Sx", pd.Series()), errors="coerce").fillna(0)
        unit_col = "Unità" if "Unità" in sym_df.columns else ("Unita'" if "Unita'" in sym_df.columns else None)
        units = sym_df[unit_col] if unit_col else ""
        sym_df["Delta"] = (sym_df["Dx"] - sym_df["Sx"]).abs()
        sym_df["DeltaStr"] = sym_df["Delta"].round(1).astype(str) + (units.astype(str) if unit_col else "")
        sym_df["SymScore"] = sym_df.apply(lambda r: symmetry_score(r["Dx"], r["Sx"], r.get(unit_col,"")), axis=1)
    st.markdown("### Barre di simmetria")
    st.plotly_chart(symmetry_bar_chart(sym_df), use_container_width=True)
    st.markdown("### Tabella EBM")
    show_cols = [c for c in ["Sezione","Test","Unità","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM"] if c in df.columns]
    df_show = df[show_cols].copy()
    df_show["CommentoEBM"] = df_show["CommentoEBM"].apply(to_ascii)
    st.dataframe(df_show.style.format({"Score":"{:.1f}","SymScore":"{:.1f}","Valore":"{:.1f}","Dx":"{:.1f}","Sx":"{:.1f}","Delta":"{:.1f}"}), use_container_width=True)

def page_storico():
    st.markdown("### Storico – confronta fino a 3 date")
    section = st.session_state["section"]; athlete = st.session_state["athlete"]
    data_hist = load_sessions(athlete, section)
    if data_hist.empty: st.info("Nessuna sessione salvata per questo atleta/sezione."); return
    dates = data_hist["date"].tolist()
    pick = st.multiselect("Seleziona fino a 3 valutazioni", dates, default=dates[:2])
    comp = []
    for d in pick[:3]:
        rec = data_hist[data_hist["date"]==d].iloc[0]
        df_rec = pd.DataFrame(json.loads(rec["data_json"])); df_rec = df_rec[df_rec.get("Valore",0) > 0]
        comp.append((d, df_rec))
    if not comp: st.warning("Seleziona almeno una data."); return
    def labels_for_df(dfc: pd.DataFrame):
        return (dfc["Test"] + " [" + dfc["Sezione"] + "]").tolist() if "Sezione" in dfc.columns else dfc["Test"].tolist()
    all_labels = sorted({lab for _, dfc in comp for lab in labels_for_df(dfc)})
    fig = go.Figure()
    if all_labels:
        labels_c = all_labels + [all_labels[0]]
        ideal = [10]*len(all_labels); ideal_c = ideal + [ideal[0]]
        fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target", line=dict(width=1, dash="dot", color="#9aa4b2")))
    for d, dfc in comp:
        labs = labels_for_df(dfc)
        scores_map = {l:s for l, s in zip(labs, pd.to_numeric(dfc.get("Score", pd.Series()), errors="coerce").fillna(0))}
        series = [scores_map.get(t, 0) for t in all_labels]
        series_c = series + ([series[0]] if all_labels else [])
        fig.add_trace(go.Scatterpolar(r=series_c, theta=(all_labels + [all_labels[0]] if all_labels else []), name=d))
    fig.update_layout(showlegend=True, title=f"Confronto – {section}", polar=dict(bgcolor=BG, radialaxis=dict(visible=True, range=[0,10], tickfont=dict(color=TEXT), gridcolor="#2A2F3A")), margin=dict(l=10,r=10,t=40,b=10), height=560, paper_bgcolor=BG, font=dict(color=TEXT))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- ROUTER -----------------
def render_page():
    page = st.radio("Navigazione", ["Valutazione", "Riepilogo", "Storico"], horizontal=True, label_visibility="collapsed")
    st.session_state["page"] = page
    if page == "Valutazione": page_valutazione()
    elif page == "Riepilogo": page_riepilogo()
    elif page == "Storico": page_storico()

render_page()
