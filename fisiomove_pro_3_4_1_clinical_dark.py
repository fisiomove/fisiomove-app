# Fisiomove Pro 3.4.1 – Clinical Dark (fix menu Movimento, commenti EBM puliti, simmetria affidabile)
# - "Movimento" selezionabile (Squat, Panca, Deadlift, Neurodinamica) con selectbox ben visibile
# - Commenti EBM senza emoji/simboli non ASCII; layout pulito (punti elenco) sia in UI che PDF
# - Simmetria calcolata per TUTTI i test bilaterali (es. Passive Hip Flexion)
# - PDF: niente caratteri speciali che creano quadratini; sostituzione Unicode -> ASCII
# - Restyling minore dark mode

import io, os, json, sqlite3, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- THEME / CONST -----------------
PRIMARY = "#3B82F6"  # accent blu
BG = "#0E1117"       # dark bg
CARD = "#1C1F26"     # dark card
TEXT = "#EAEAEA"     # testo chiaro
OK = "#16a34a"; MID = "#f59e0b"; LOW = "#dc2626"

DEFAULT_LOGO_PATH = "logo_fisiomove.png"
DB_PATH = "fisiomove.db"

st.set_page_config(page_title="Fisiomove Pro 3.4.1 – Clinical Dark", layout="wide")

# ----------------- GLOBAL CSS (dark) -----------------
DARK_CSS = f"""
<style>
html, body, .stApp {{ background: {BG} !important; color: {TEXT} !important; }}
.block-container {{ padding-top: 0.6rem; padding-bottom: 4.8rem; max-width: 760px; }}
h1, h2, h3, h4, h5 {{ color: {PRIMARY} !important; letter-spacing: 0.2px; }}
.stButton>button {{ width: 100%; border-radius: 12px; padding: 0.75rem 1rem; font-size: 1.05rem; background: {PRIMARY}22; color: {TEXT}; border: 1px solid {PRIMARY}55; }}
.stButton>button:hover {{ background: {PRIMARY}44; }}
.card {{ border: 1px solid #2A2F3A; border-radius: 14px; padding: 12px; background: {CARD}; box-shadow: 0 2px 10px rgba(0,0,0,0.25); }}
.navbar {{ position: fixed; left: 0; right: 0; bottom: 0; background: {CARD}; border-top: 1px solid #2A2F3A; padding: 8px 12px; z-index: 999; }}
.navgrid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; max-width: 760px; margin: 0 auto; }}
.navbtn {{ text-align: center; border-radius: 12px; padding: 8px 6px; border: 1px solid #2A2F3A; color: {TEXT}; background:{CARD}; font-weight:600; }}
.navbtn.active {{ border-color: {PRIMARY}; color: {PRIMARY}; background: {PRIMARY}15; }}
.badge {{ display:inline-block; padding: 6px 10px; border-radius: 10px; font-weight:700; }}
.stApp header {{ visibility: hidden; height: 0; }} footer {{ visibility: hidden; height: 0; }}
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
        ("External Rotation (90 deg abd)", "°", 90, "ER spalla a 90 gradi abduzione.", True),
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
        ("SLR (sensibile)", "°", 70, "Range con sintomatologia controllata.", True),
        ("PNLT tibiale (bias)", "°", 40, "Differenziazione sensibile tibiale.", True),
        ("PNLT peroneale (bias)", "°", 40, "Differenziazione sensibile peroneale.", True),
        ("ULNT1 (n. mediano)", "°", 100, "Range con differenziazione scapolare.", True),
    ],
}

TUTORIAL = {
    "Weight Bearing Lunge Test": "Affondo verso parete, tallone a terra; distanza punta-parete (cm) a ginocchio in contatto.",
    "Passive Hip Flexion": "Supino, bacino stabilizzato; arresto prima di compensi lombari.",
    "Hip Rotation (flexed 90°)": "Supino, anca/gomito 90 gradi; IR/ER mantenendo bacino neutro.",
    "Thoracic Extension (T4- T12)": "Inclinometro tra T4 e T12; estensione attiva massima.",
    "Shoulder ER (adducted, low-bar)": "In piedi, braccio addotto; ER utile per posizione bilanciere.",
    "Shoulder Flexion (supine)": "Supino, elevazione completa; controlla coste.",
    "External Rotation (90 deg abd)": "Supino, abduzione 90 gradi, gomito 90 gradi; ER controllata.",
    "Pectoralis Minor Length": "Supino rilassato; distanza acromion-piano (cm).",
    "Wall Angel (distance)": "Schiena/avambracci a parete; misura distanza (cm).",
    "Thomas Test (modified)": "Bordo lettino; una coscia in flessione, controlaterale pendola.",
    "Active Knee Extension (AKE)": "Supino, anca 90 gradi; estendi ginocchio; misura deficit in gradi.",
    "Straight Leg Raise (SLR)": "Supino; eleva arto in estensione fino a tensione controllata.",
    "Modified Schober (lumbar)": "Segni 10 cm sopra e 5 cm sotto L5; misura incremento.",
    "Sorensen Endurance": "Prono su bordo lettino, bacino fissato; mantieni tronco orizzontale.",
    "SLR (sensibile)": "SLR con monitoraggio sintomi; annota angolo e variazione con differenziazione.",
    "PNLT tibiale (bias)": "Dorsiflessione+eversione; valuta range e sintomi.",
    "PNLT peroneale (bias)": "Plantarflessione+inversione; valuta range e sintomi.",
    "ULNT1 (n. mediano)": "Sequenza standard; registra angolo e differenziazione scapolare.",
}

# ----------------- UTILS -----------------
def to_ascii(s: str) -> str:
    """Sostituisce simboli non ASCII per evitare quadratini nei PDF."""
    if not isinstance(s, str): return s
    repl = {
        "–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "•": "-", "°": " gradi",
        "→": "->", "↔": "<->", "≥": ">=", "≤": "<=", "Δ": "Delta", "✓": "-", "×": "x"
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s

def score_linear(value: float, ref: float) -> float:
    try:
        v = float(value); r = float(ref)
        if r <= 0: return 0.0
        return float(np.clip((v/r) * 10.0, 0.0, 10.0))
    except Exception:
        return 0.0

def symmetry_score(dx: float, sx: float, unit: str) -> float:
    """0-10: 10=perfetta simmetria. Soglie: 10 gradi / 4 cm / 20%."""
    dx = float(dx); sx = float(sx); diff = abs(dx - sx)
    if unit == "%": scale = 20.0
    elif unit == "cm": scale = 8.0  # 8 cm diff = 0 (piu' inclusivo per WBLT)
    else: scale = 10.0
    return float(10.0 * max(0.0, 1.0 - min(diff, scale)/scale))

def color_by_score(s: float) -> str:
    if s < 4: return LOW
    if s <= 7: return MID
    return OK

def badge(text: str, color_hex: str) -> str:
    return f"<span class='badge' style='background:{color_hex}22;color:{color_hex};'>{text}</span>"

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
        margin=dict(l=10, r=10, t=40, b=10), height=440, paper_bgcolor=BG, font=dict(color=TEXT)
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
    fig, ax = plt.subplots(figsize=(4.6,4.6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.grid(True, linewidth=0.5, color="#cccccc", alpha=0.9); ax.spines["polar"].set_color("#999999")
    ax.plot(angles_c, ideal_c, color="#777777", linewidth=1, linestyle="dotted", label="Target 10/10")
    ax.plot(angles_c, scores_c, color="#3B82F6", linewidth=2, label="Atleta")
    ax.fill(angles_c, scores_c, color="#3B82F644")
    ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10]); ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=8)
    for angle, r in zip(angles, scores):
        ax.text(angle, r + 0.3, f"{r:.1f}", ha="center", va="center", fontsize=7, color="#0b2758")
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
                      plot_bgcolor=BG, paper_bgcolor=BG, height=400, margin=dict(l=10,r=10,t=40,b=10),
                      title="Simmetria Dx/Sx (0–10)")
    return fig

# Symmetry bars for PDF (matplotlib) -> PNG buffer
def symmetry_png_buffer(sym_df):
    buf = io.BytesIO()
    if sym_df is None or sym_df.empty: return buf
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
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
    if val_avg <= 0:
        return "Non disponibile."
    perf = "ottimo" if val_avg >= ref*0.9 else "nella norma" if val_avg >= ref*0.7 else "ridotto"
    if "Lunge" in test_name:
        lines += [
            f"- Dorsiflessione in carico: {perf}.",
            "- Valori ridotti spostano il carico sull'avampiede e anticipano la retroversione del bacino nello squat.",
            "- Possibile aumento di stress femoro-rotuleo e perdita di stabilita' del tallone.",
            "Riferimenti: Barton CJ 2012 JOSPT; Rabin A 2015 Phys Ther."
        ]
    elif "Thoracic Extension" in test_name:
        lines += [
            f"- Estensione toracica: {perf}.",
            "- Riduzioni favoriscono eccessiva inclinazione del tronco e compensi scapolo-omerali in squat/panca.",
            "Riferimenti: Wilke J 2020 Front Physiol."
        ]
    elif "AKE" in test_name or "SLR" in test_name:
        lines += [
            f"- Estensibilita' catena posteriore: {perf}.",
            "- Limiti marcati riducono hinge efficiente e aumentano strain neurale/muscolare.",
            "Riferimenti: Malliaras P 2015 Sports Med; Bohannon 1999 Phys Ther."
        ]
    elif "ER" in test_name and "Shoulder" in test_name:
        lines += [
            f"- Rotazione esterna spalla: {perf}.",
            "- Limitazioni in panca/low-bar aumentano stress anteriore e alterano il controllo scapolare.",
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
            "- Valori bassi riducono adattabilita' in flessione e spostano carico su anche/posteriore.",
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
        if (unit in ["°","sec"] and d >= 10) or (unit == "cm" and d >= 4):
            lines += ["- Nota: asimmetria significativa. Potenziale compenso e rischio overuse (Fousekis 2010 BJSM; Bishop 2018 JSCR)."]
    # Join and ASCII sanitize
    return to_ascii("\n".join(lines))

# ----------------- STORAGE -----------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        athlete TEXT,
        section TEXT,
        date TEXT,
        data_json TEXT
    );""")
    con.commit(); con.close()

def save_session(athlete: str, section: str, date_str: str, df: pd.DataFrame):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    payload = df.to_dict(orient="records")
    cur.execute("INSERT INTO sessions (athlete, section, date, data_json) VALUES (?, ?, ?, ?);",
                (athlete, section, date_str, json.dumps(payload, ensure_ascii=False)))
    con.commit(); con.close()

def load_sessions(athlete: str, section: str) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("SELECT date, data_json FROM sessions WHERE athlete=? AND section=? ORDER BY date DESC;", (athlete, section))
    rows = cur.fetchall(); con.close()
    if not rows: return pd.DataFrame(columns=["date", "data_json"])
    return pd.DataFrame([{"date": r[0], "data_json": r[1]} for r in rows])

def export_session_json(athlete: str, section: str, date_str: str, df: pd.DataFrame) -> bytes:
    payload = {"athlete": athlete, "section": section, "date": date_str, "records": df.to_dict(orient="records")}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

init_db()

# ----------------- STATE -----------------
st.session_state.setdefault("page", "Valutazione")
st.session_state.setdefault("athlete", "Mario Rossi")
st.session_state.setdefault("evaluator", "Dott. Alessandro Ferreri - Fisioterapista")
st.session_state.setdefault("date", dt.date.today().strftime("%Y-%m-%d"))
st.session_state.setdefault("section", "Squat")

# ----------------- HEADER -----------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
try:
    st.image(DEFAULT_LOGO_PATH, width=120, use_container_width=False)
except Exception:
    st.markdown("<div style='color:#9aa4b2;'>Fisiomove</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;margin-top:0'>Fisiomove Pro 3.4.1</h2>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center;color:{PRIMARY};margin-top:-10px;'>Clinical Dark – Mobilita' & Simmetria</div>", unsafe_allow_html=True)

# ----------------- DATI ATLETA (card) -----------------
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
    # Menu a tendina ben visibile per selezionare il MOVIMENTO testato
    st.session_state["section"] = st.selectbox("Movimento testato", options=list(TESTS.keys()), index=list(TESTS.keys()).index(st.session_state["section"]))
st.caption("Suggerimento: imposta il movimento prima di inserire i valori.")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE: VALUTAZIONE -----------------
def page_valutazione():
    section = st.session_state["section"]
    st.markdown("### Valutazione")
    st.write("Per test bilaterali inserisci **Dx** e **Sx**. Il radar usa la media normalizzata (0-10). La sezione Simmetria calcola Delta e punteggio (0-10).")

    rows = []   # records for main DF
    sym_rows = []  # records for symmetry DF

    for name, unit, ref, desc, bilateral in TESTS[section]:
        st.markdown(f"**{name}** · *{desc}*")
        with st.expander("Standard del test", expanded=False):
            st.caption(TUTORIAL.get(name, "-"))

        # slider ranges
        if unit == "sec":
            maxv = float(ref) * 1.5
        elif unit == "%":
            maxv = 100.0
        else:
            maxv = float(ref) * 1.3

        if bilateral:
            c1, c2 = st.columns(2)
            with c1:
                dx = st.slider(f"Dx ({unit}, rif. {ref})", 0.0, float(maxv), 0.0, 0.1, key=f"{section}_{name}_dx")
            with c2:
                sx = st.slider(f"Sx ({unit}, rif. {ref})", 0.0, float(maxv), 0.0, 0.1, key=f"{section}_{name}_sx")

            val_avg = (dx + sx) / 2.0
            ability = score_linear(val_avg, max(ref, 1))
            st.markdown(badge(f"Abilita' media: {ability:.1f}/10", color_by_score(ability)), unsafe_allow_html=True)

            # sym
            sym = symmetry_score(dx, sx, unit)
            delta = abs(dx - sx)
            delta_str = f"{delta:.1f}{unit}"
            st.caption(f"Delta Simmetria: {delta_str} --> Punteggio {sym:.1f}/10")

            sym_rows.append({"Test": name, "Dx": dx, "Sx": sx, "Unità": unit, "Delta": delta, "DeltaStr": delta_str, "SymScore": sym})

            comment = ebm_explain_row(name, unit, ref, val_avg, dx, sx)
            rows.append([name, unit, ref, val_avg, ability, dx, sx, delta, sym, comment])
        else:
            val = st.slider(f"{unit} (rif. {ref})", 0.0, float(maxv), 0.0, 0.1, key=f"{section}_{name}")
            ability = score_linear(val, max(ref, 1))
            st.markdown(badge(f"Punteggio: {ability:.1f}/10", color_by_score(ability)), unsafe_allow_html=True)
            rows.append([name, unit, ref, val, ability, None, None, None, None, ebm_explain_row(name, unit, ref, val)])

        st.divider()

    df_cols = ["Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM"]
    df = pd.DataFrame(rows, columns=df_cols)
    sym_df = pd.DataFrame(sym_rows, columns=["Test","Dx","Sx","Unità","Delta","DeltaStr","SymScore"])

    # Completion
    df_meas = df[df["Valore"] > 0]
    pct = int((len(df_meas) / len(df) * 100)) if not df.empty else 0
    st.progress(pct); st.caption(f"Completati: {len(df_meas)} su {len(df)} test ({pct}%)")

    # Radar (ability only, tests with Valore>0)
    st.markdown("### Radar abilita' (media normalizzata)")
    fig_radar = radar_plot(df_meas["Test"].tolist(), df_meas["Score"].tolist(), f"{section} – Abilita' (0-10)")
    st.plotly_chart(fig_radar, use_container_width=True)

    # Symmetry bars
    st.markdown("### Simmetria Dx/Sx")
    fig_sym = symmetry_bar_chart(sym_df)  # mostro anche Delta=0 per coerenza visiva
    st.plotly_chart(fig_sym, use_container_width=True)

    # EBM Table (ASCII safe)
    st.markdown("### Spiegazione clinica (EBM)")
    show_cols = ["Test","Unità","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM"]
    df_show = df[show_cols].copy()
    df_show["CommentoEBM"] = df_show["CommentoEBM"].apply(to_ascii)
    st.dataframe(df_show.style.format({"Score":"{:.1f}", "SymScore":"{:.1f}", "Valore":"{:.1f}", "Dx":"{:.1f}", "Sx":"{:.1f}", "Delta":"{:.1f}"}), use_container_width=True)

    # Actions
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Salva"):
            save_session(st.session_state["athlete"], section, st.session_state["date"], df_show)
            st.success("Sessione salvata (storico disponibile in 'Storico').")
    with c2:
        if st.button("JSON"):
            js = export_session_json(st.session_state["athlete"], section, st.session_state["date"], df_show)
            st.download_button("Scarica JSON", data=js, file_name=f"{st.session_state['athlete'].replace(' ','')}_{section}_{st.session_state['date']}.json", mime="application/json")
    with c3:
        disabled_pdf = df_meas.empty
        if st.button("PDF", disabled=disabled_pdf):
            pdf = pdf_single_section(load_logo(), st.session_state["athlete"], st.session_state["evaluator"], st.session_state["date"], section, df_show, sym_df)
            st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{section}_{st.session_state['athlete'].replace(' ','')}_{st.session_state['date']}.pdf", mime="application/pdf")

def load_logo():
    try:
        with open(DEFAULT_LOGO_PATH, "rb") as f: return f.read()
    except Exception:
        return None

def pdf_single_section(logo_bytes, athlete, evaluator, date_str, section_name, df, sym_df):
    disp = df[df["Valore"] > 0].copy()
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4, rightMargin=1.2*cm, leftMargin=1.2*cm, topMargin=1.2*cm, bottomMargin=1.4*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=14)
    story = []
    story.append(Paragraph("Fisiomove Pro 3.4.1 – Clinical Dark", title))
    story.append(Spacer(1, 6))
    head = Table([["Atleta", to_ascii(athlete), "Data", date_str, "Valutatore", to_ascii(evaluator)]], colWidths=[2*cm, 6*cm, 2*cm, 3*cm, 2.5*cm, 4*cm])
    head.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(head); story.append(Spacer(1, 8))
    if logo_bytes:
        try:
            story.append(Image(io.BytesIO(logo_bytes), width=5.0*cm, height=5.0*cm)); story.append(Spacer(1, 6))
        except Exception:
            pass
    story.append(Paragraph(f"Sezione testata: <b>{to_ascii(section_name)}</b>", styles['Heading3']))
    if disp.empty:
        story.append(Paragraph("Nessun test compilato (valori > 0).", normal))
    else:
        disp_pdf = disp[["Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore"]].copy()
        disp_pdf["Test"] = disp_pdf["Test"].apply(to_ascii)
        disp_pdf["Unità"] = disp_pdf["Unità"].apply(to_ascii)
        disp_pdf["Score"] = disp_pdf["Score"].round(1); disp_pdf["SymScore"] = disp_pdf["SymScore"].round(1)
        table_data = [["Test","Unita'","Rif.","Valore","Score","Dx","Sx","Delta","Simmetria"]] + disp_pdf.fillna("").values.tolist()
        tbl = Table(table_data, colWidths=[6.5*cm,1.5*cm,1.8*cm,1.8*cm,1.8*cm,1.5*cm,1.5*cm,1.8*cm,2.2*cm])
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("ALIGN",(2,1),(-1,-1),"CENTER")]))
        story.append(tbl); story.append(Spacer(1, 8))
        # Radar image
        rbuf = radar_png_buffer(disp["Test"].tolist(), disp["Score"].tolist())
        try:
            story.append(Image(io.BytesIO(rbuf.getvalue()), width=10.0*cm, height=10.0*cm))
            story.append(Paragraph(f"Radar abilita' – {to_ascii(section_name)}", styles['Normal'])); story.append(Spacer(1, 6))
        except Exception as e:
            story.append(Paragraph(f"Radar non disponibile ({to_ascii(str(e))}).", normal)); story.append(Spacer(1, 6))
        # Symmetry table + bars
        sym_disp = sym_df[["Test","Dx","Sx","Delta","SymScore"]].copy()
        if not sym_disp.empty:
            sym_disp["Test"] = sym_disp["Test"].apply(to_ascii)
            sym_disp["SymScore"] = sym_disp["SymScore"].round(1); sym_disp["Delta"] = sym_disp["Delta"].round(1)
            t2 = [["Test","Dx","Sx","Delta","Simmetria (0-10)"]] + sym_disp.values.tolist()
            tbl2 = Table(t2, colWidths=[7.0*cm,2.0*cm,2.0*cm,2.0*cm,3.0*cm])
            tbl2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
            story.append(tbl2); story.append(Spacer(1, 6))
            sbuf = symmetry_png_buffer(sym_disp.rename(columns={"SymScore":"SymScore"}))
            try:
                story.append(Image(io.BytesIO(sbuf.getvalue()), width=12.0*cm, height=8.0*cm))
                story.append(Paragraph("Barre di simmetria Dx/Sx", styles['Normal']))
            except Exception:
                pass
        # EBM clinical comments – bullet list pulita
        story.append(Spacer(1, 8))
        story.append(Paragraph("Feedback clinico (EBM)", styles['Heading3']))
        for _, r in disp.iterrows():
            bullets = [to_ascii(x) for x in r["CommentoEBM"].split("\n") if x.strip()]
            flow = ListFlowable([ListItem(Paragraph(b, styles['Normal'])) for b in bullets], bulletType='bullet', leftIndent=12)
            story.append(Paragraph(f"<b>{to_ascii(r['Test'])}</b>", styles['Normal']))
            story.append(flow); story.append(Spacer(1, 4))
    doc.build(story); buf_pdf.seek(0); return buf_pdf

# ----------------- PAGES -----------------
def page_riepilogo():
    section = st.session_state["section"]
    athlete = st.session_state["athlete"]
    data_hist = load_sessions(athlete, section)
    if data_hist.empty:
        st.info("Nessuna sessione salvata per questo atleta/sezione.")
        return
    dates = data_hist["date"].tolist()
    pick = st.selectbox("Seleziona una valutazione salvata", dates, index=0)
    rec = data_hist[data_hist["date"]==pick].iloc[0]
    df = pd.DataFrame(json.loads(rec["data_json"]))
    df_meas = df[df["Valore"] > 0]
    st.markdown("### Radar abilita'")
    fig = radar_plot(df_meas["Test"].tolist(), df_meas["Score"].tolist(), f"{section} – Abilita' (0-10) @ {pick}")
    st.plotly_chart(fig, use_container_width=True)
    # symmetry
    sym_df = df.dropna(subset=["Dx","Sx"]).copy()
    sym_df["Delta"] = (sym_df["Dx"] - sym_df["Sx"]).abs()
    sym_df["DeltaStr"] = sym_df["Delta"].round(1).astype(str) + sym_df["Unità"].astype(str)
    sym_df["SymScore"] = sym_df.apply(lambda r: symmetry_score(r["Dx"], r["Sx"], r["Unità"]), axis=1)
    st.markdown("### Barre di simmetria")
    st.plotly_chart(symmetry_bar_chart(sym_df), use_container_width=True)
    st.markdown("### Tabella EBM")
    show_cols = ["Test","Unità","Valore","Score","Dx","Sx","Delta","SymScore","CommentoEBM"]
    df_show = df[show_cols].copy(); df_show["CommentoEBM"] = df_show["CommentoEBM"].apply(to_ascii)
    st.dataframe(df_show.style.format({"Score":"{:.1f}","SymScore":"{:.1f}","Valore":"{:.1f}","Dx":"{:.1f}","Sx":"{:.1f}","Delta":"{:.1f}"}), use_container_width=True)

def page_storico():
    st.markdown("### Storico – confronta fino a 3 date")
    section = st.session_state["section"]
    athlete = st.session_state["athlete"]
    data_hist = load_sessions(athlete, section)
    if data_hist.empty:
        st.info("Nessuna sessione salvata per questo atleta/sezione.")
        return
    dates = data_hist["date"].tolist()
    pick = st.multiselect("Seleziona fino a 3 valutazioni", dates, default=dates[:2])
    comp = []
    for d in pick[:3]:
        rec = data_hist[data_hist["date"]==d].iloc[0]
        df_rec = pd.DataFrame(json.loads(rec["data_json"])); df_rec = df_rec[df_rec["Valore"] > 0]
        comp.append((d, df_rec))
    if not comp:
        st.warning("Seleziona almeno una data."); return
    labels = sorted({t for _,dfc in comp for t in dfc["Test"].tolist()})
    fig = go.Figure()
    if labels:
        labels_c = labels + [labels[0]]
        ideal = [10]*len(labels); ideal_c = ideal + [ideal[0]]
        fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target", line=dict(width=1, dash="dot", color="#9aa4b2")))
    for d, dfc in comp:
        scores_map = {t:s for t,s in zip(dfc["Test"], dfc["Score"])}
        series = [scores_map.get(t, 0) for t in labels]
        series_c = series + ([series[0]] if labels else [])
        fig.add_trace(go.Scatterpolar(r=series_c, theta=(labels + [labels[0]] if labels else []), name=d))
    fig.update_layout(showlegend=True, title=f"Confronto – {section}", polar=dict(bgcolor=BG, radialaxis=dict(visible=True, range=[0,10], tickfont=dict(color=TEXT), gridcolor="#2A2F3A")), margin=dict(l=10,r=10,t=40,b=10), height=520, paper_bgcolor=BG, font=dict(color=TEXT))
    st.plotly_chart(fig, use_container_width=True)

# ----------------- ROUTER -----------------
def render_page():
    if st.session_state["page"] == "Valutazione": page_valutazione()
    elif st.session_state["page"] == "Riepilogo": page_riepilogo()
    elif st.session_state["page"] == "Storico": page_storico()

render_page()

# ----------------- NAVBAR -----------------
def nav_button(label, key, active_key):
    cls = "navbtn active" if (st.session_state['page'] == active_key) else "navbtn"
    st.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
    if st.button(" ", key=key):
        st.session_state['page'] = active_key
        st.rerun()

st.markdown("<div class='navbar'><div class='navgrid'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: nav_button("Valutazione", "nav_val", "Valutazione")
with c2: nav_button("Riepilogo", "nav_riep", "Riepilogo")
with c3: nav_button("Storico", "nav_sto", "Storico")
st.markdown("</div></div>", unsafe_allow_html=True)
