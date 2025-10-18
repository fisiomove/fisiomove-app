# Fisiomove Pro 3.3 – Mobile FIX (blu/bianco, no sidebar, navbar inferiore, animazione CSS sicura)
# - UI mobile-first, colonna singola, pulsanti grandi
# - Navbar inferiore: Valutazione · Riepilogo · Storico
# - Logo centrato (logo_fisiomove.png, opzionale)
# - Funzioni 3.2 invariate: radar fisso avanzato, PDF, SQLite, JSON, storico
# - Animazione fade-in via CSS su body/container (senza wrapper HTML)
# - Compatibile Streamlit Cloud (matplotlib Agg)

import io, os, json, sqlite3, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------- THEME / CONST -----------------
PRIMARY = "#004A9F"
BG = "#FFFFFF"
DEFAULT_LOGO_PATH = "logo_fisiomove.png"
DB_PATH = "fisiomove.db"

st.set_page_config(page_title="Fisiomove Pro 3.3 – Mobile FIX", layout="wide")

# ----------------- GLOBAL CSS (mobile-first + animazioni sicure) -----------------
MOBILE_CSS = f"""
<style>
/* Animazione globale sicura: applicata alla root Streamlit */
html, body, .stApp {{
  background: {BG} !important;
  animation: fadeIn 260ms ease-out;
}}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

/* Contenitore principale */
.block-container {{
  padding-top: 0.8rem;
  padding-bottom: 4.8rem; /* spazio per navbar */
  max-width: 720px;
}}
/* Tipografia */
h1, h2, h3, h4, h5 {{
  color: {PRIMARY} !important;
  letter-spacing: 0.2px;
}}
/* Pulsanti grandi */
.stButton>button {{
  width: 100%;
  border-radius: 12px;
  padding: 0.75rem 1rem;
  font-size: 1.05rem;
}}
/* Badge */
.badge {{
  display:inline-block;
  padding: 6px 10px;
  border-radius: 10px;
  font-weight: 700;
}}
/* Card */
.card {{
  border: 1px solid #e8e8e8;
  border-radius: 14px;
  padding: 12px;
  background: #fff;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}}
/* Navbar inferiore */
.navbar {{
  position: fixed;
  left: 0; right: 0; bottom: 0;
  background: #ffffff;
  border-top: 1px solid #eaeaea;
  padding: 8px 12px;
  z-index: 999;
}}
.navgrid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 10px;
  max-width: 720px;
  margin: 0 auto;
}}
.navbtn {{
  text-align: center;
  border-radius: 12px;
  padding: 8px 6px;
  border: 1px solid #e7e7e7;
  color: #333;
  background:#fff;
  font-weight:600;
}}
.navbtn.active {{
  border-color: {PRIMARY};
  color: {PRIMARY};
  background: {PRIMARY}10;
}}
/* Nascondi header/footer streamlit per look "app" */
.stApp header {{ visibility: hidden; height: 0; }}
footer {{ visibility: hidden; height: 0; }}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)

# ----------------- HELPERS -----------------
def score_linear(value: float, ref: float) -> float:
    try:
        v = float(value); r = float(ref)
        if r <= 0: return 0.0
        return max(0.0, min(10.0, (v/r)*10.0))
    except Exception:
        return 0.0

def score_color(s: float) -> str:
    if s < 4: return "#D32F2F"
    if s <= 7: return "#F9A825"
    return "#2E7D32"

def badge(text: str, color: str) -> str:
    return f"<span class='badge' style='background:{color}20;color:{color};'>{text}</span>"

def radar_plot(labels, scores, title):
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(title="Nessun test compilato", polar=dict(radialaxis=dict(visible=True, range=[0,10])), height=320)
        return fig
    labels_c = list(labels) + [labels[0]]
    scores_c = list(scores) + [scores[0]]
    ideal = [10]*len(labels); ideal_c = ideal + [ideal[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target 10/10",
                                  line=dict(width=1, dash="dot", color="#999")))
    fig.add_trace(go.Scatterpolar(r=scores_c, theta=labels_c, name="Atleta",
                                  fill="toself", line=dict(width=2, color=PRIMARY)))
    fig.update_layout(
        title=title,
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center"),
        polar=dict(radialaxis=dict(visible=True, range=[0,10], tickmode="linear", dtick=1, gridcolor="#cccccc")),
        margin=dict(l=10, r=10, t=40, b=10),
        height=440
    )
    return fig

def radar_png_buffer(labels, scores):
    N = len(labels)
    buf = io.BytesIO()
    if N == 0: return buf
    if N == 1:
        labels = labels * 2
        scores = scores * 2
        N = 2
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]
    scores_c = list(scores) + [scores[0]]
    ideal_c = [10]*N + [10]
    fig, ax = plt.subplots(figsize=(4.6,4.6), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.grid(True, linewidth=0.5, color="#cccccc", alpha=0.9)
    ax.spines["polar"].set_color("#999999")
    ax.plot(angles_c, ideal_c, color="#777777", linewidth=1, linestyle="dotted", label="Target 10/10")
    ax.plot(angles_c, scores_c, color="#004A9F", linewidth=2, label="Atleta")
    ax.fill(angles_c, scores_c, color="#2f6fb640")
    ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10]); ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=8)
    for angle, r in zip(angles, scores):
        ax.text(angle, r + 0.3, f"{r:.1f}", ha="center", va="center", fontsize=7, color="#003366")
    fig.savefig(buf, format="png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def ebm_comment(section_name, df):
    low = df[df["Score"] < 4]["Test"].tolist()
    mid = df[(df["Score"] >= 4) & (df["Score"] <= 7)]["Test"].tolist()
    high = df[df["Score"] > 7]["Test"].tolist()
    out = [f"Sintesi clinica – {section_name}:"]
    if low: out.append("• Deficit clinicamente rilevanti in: " + ", ".join(low))
    if mid: out.append("• Aree migliorabili: " + ", ".join(mid))
    if high: out.append("• Aree adeguate/ottimali: " + ", ".join(high))
    if section_name == "Squat":
        out.append("EBM: dorsiflessione in carico ↔ profondità/controllo; estensione toracica ↔ inclinazione tronco; ER spalla per low‑bar.")
    if section_name == "Panca":
        out.append("EBM: estensione toracica/ER spalla ↔ arco e stress gleno‑omerale; pettorale minore/controllo scapolare; Thomas modificato e postura.")
    if section_name == "Deadlift":
        out.append("EBM: AKE/SLR e mobilità lombare ↔ hinge e tolleranza al carico; dorsiflessione in carico; endurance tronco.")
    if section_name.startswith("Simmetrie"):
        out.append("EBM: asimmetrie >10% possono correlare con alterazioni di performance/overuse; interpretare nel contesto.")
    if section_name == "Neurodinamica":
        out.append("EBM: range sintomatico vs non sintomatico e differenziazione sensibile guidano progressione carichi.")
    out.append("Per l'atleta: lavora prima sui test <7/10; obiettivo portare tutto ≥8/10 con tecnica pulita.")
    return "\n".join(out)

# ----------------- DATASETS -----------------
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test","cm",10,"Dorsiflessione in carico (WB)."),
        ("Passive Hip Flexion","°",120,"Flessione passiva dell'anca."),
        ("Hip Rotation (flexed 90°)","°",40,"Rotazione anca a 90° flessione."),
        ("Thoracic Extension (T4–T12)","°",30,"Estensione toracica segmentale."),
        ("Shoulder ER (adducted, low‑bar)","°",70,"Posizione low‑bar bilanciere."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)","°",180,"Flessione gleno‑omerale in supino."),
        ("External Rotation (90° abd)","°",90,"ER spalla a 90° abduzione."),
        ("Thoracic Extension (T4–T12)","°",30,"Arco toracico per panca."),
        ("Pectoralis Minor Length","cm",10,"Lunghezza pettorale minore (spazio acromiale)."),
        ("Wall Angel (distance)","cm",10,"Controllo scapolare a parete."),
        ("Thomas Test (modified)","°",10,"Estensione d'anca per postura lombopelvica."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)","°",90,"Estensibilità ischiocrurali (catena aperta)."),
        ("Straight Leg Raise (SLR)","°",90,"Catena posteriore neural/muscolare."),
        ("Weight Bearing Lunge Test","cm",10,"Dorsiflessione in carico."),
        ("Modified Schober (lumbar)","cm",5,"Mobilità lombare in flessione."),
        ("Sorensen Endurance","sec",180,"Endurance estensori lombari."),
    ],
    "Simmetrie Superiori": [
        ("Grip Strength Δ% (Dx-Sx)","%",0,"Diff. % assoluta. 0%→10; ≥20%→0."),
        ("ER Spalla Δ° (90° abd)","°",0,"Differenza assoluta ER tra lati."),
        ("Elevazione Scapolare Δcm","cm",0,"Differenza distanza acromion-parete."),
    ],
    "Simmetrie Inferiori": [
        ("AKE Δ°","°",0,"Differenza assoluta AKE tra arti."),
        ("SLR Δ°","°",0,"Differenza assoluta SLR tra arti."),
        ("WBLT Δcm","cm",0,"Differenza assoluta WBLT tra arti."),
    ],
    "Neurodinamica": [
        ("SLR (sensibile)","°",70,"Range con sintomatologia controllata."),
        ("PNLT tibiale (dorsiflessione/eversione)","°",40,"Differenziazione sensibile tibiale."),
        ("PNLT peroneale (plantarflessione/inversione)","°",40,"Differenziazione sensibile peroneale."),
        ("ULNT1 (n. mediano)","°",100,"Range con differenziazione cingolo."),
    ],
}

TUTORIAL = {
    "Weight Bearing Lunge Test": "Affondo verso parete, tallone a terra. Avanza finché il ginocchio tocca la parete senza sollevare il tallone. Distanza punta-parete (cm).",
    "Passive Hip Flexion": "Supino, bacino stabilizzato. Flessione passiva d'anca senza compensi lombari.",
    "Hip Rotation (flexed 90°)": "Supino, anca e ginocchio a 90°. Ruota IR/ER mantenendo bacino stabile.",
    "Thoracic Extension (T4–T12)": "Inclinometro tra T4–T12. Estensione attiva massima, bacino neutro.",
    "Shoulder ER (adducted, low‑bar)": "In piedi, braccio addotto. ER come per low-bar, scapole addotte.",
    "Shoulder Flexion (supine)": "Supino, elevazione braccio. Stabilizza coste.",
    "External Rotation (90° abd)": "Supino, abd 90°, gomito 90°. ER controllata.",
    "Pectoralis Minor Length": "Supino rilassato. Distanza acromion-posteriore da tavolo/parete (cm).",
    "Wall Angel (distance)": "Schiena/avambracci a parete, braccia a W. Distanza avambraccio/polso (cm).",
    "Thomas Test (modified)": "Bordo lettino. Una coscia abbracciata, l’altra pendola. Estensione anca e compensi.",
    "Active Knee Extension (AKE)": "Supino, anca 90°, estendi ginocchio. Misura deficit in °.",
    "Straight Leg Raise (SLR)": "Supino. Eleva arto in estensione fino a tensione controllata.",
    "Modified Schober (lumbar)": "Segni 10 cm sopra e 5 cm sotto L5; flessione massima. Incremento cm.",
    "Sorensen Endurance": "Prono su bordo lettino, bacino fissato. Mantieni tronco orizzontale (sec).",
    "Grip Strength Δ% (Dx-Sx)": "Dinamometro, 3 prove/lato. Δ% = |Dx−Sx| / max × 100.",
    "ER Spalla Δ° (90° abd)": "Differenza assoluta ER tra arti a 90° abd.",
    "Elevazione Scapolare Δcm": "Differenza altezza/distanza scapolare.",
    "AKE Δ°": "Differenza assoluta AKE.",
    "SLR Δ°": "Differenza assoluta SLR.",
    "WBLT Δcm": "Differenza assoluta WBLT.",
    "SLR (sensibile)": "SLR con monitoraggio sintomi; annota angolo.",
    "PNLT tibiale (dorsiflessione/eversione)": "Bias tibiale; annota range.",
    "PNLT peroneale (plantarflessione/inversione)": "Bias peroneale; annota range.",
    "ULNT1 (n. mediano)": "Sequenza ULNT1; registra angolo con differenziazione scapolare.",
}

# ----------------- STORAGE (SQLite + JSON) -----------------
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
if "page" not in st.session_state: st.session_state.page = "Valutazione"
if "athlete" not in st.session_state: st.session_state.athlete = "Mario Rossi"
if "evaluator" not in st.session_state: st.session_state.evaluator = "Dott. Alessandro Ferreri – Fisioterapista"
if "date" not in st.session_state: st.session_state.date = dt.date.today().strftime("%Y-%m-%d")
if "section" not in st.session_state: st.session_state.section = "Squat"

# ----------------- HEADER -----------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
try:
    st.image(DEFAULT_LOGO_PATH, width=120, use_container_width=False)
except Exception:
    st.markdown("<div style='color:#777;'>Fisiomove</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;margin-top:0'>Fisiomove Pro 3.3</h2>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center;color:{PRIMARY};margin-top:-10px;'>Mobile Edition – Mobilità EBM</div>", unsafe_allow_html=True)

# ----------------- DATI ATLETA (card) -----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.session_state.athlete = st.text_input("Atleta", st.session_state.athlete)
with col2:
    st.session_state.evaluator = st.text_input("Valutatore", st.session_state.evaluator)
col3, col4 = st.columns(2)
with col3:
    st.session_state.date = st.date_input("Data", value=dt.date.fromisoformat(st.session_state.date)).strftime("%Y-%m-%d")
with col4:
    st.session_state.section = st.selectbox("Sezione", list(TESTS.keys()), index=list(TESTS.keys()).index(st.session_state.section))
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGES -----------------
def page_valutazione():
    section = st.session_state.section
    st.markdown("### Valutazione")
    st.write("Inserisci i valori (°/cm/sec/%). Punteggio 0–10 lineare rispetto al riferimento EBM. Tocca ℹ️ per lo standard del test.")
    rows = []
    for name, unit, ref, desc in TESTS[section]:
        st.markdown(f"**{name}** · *{desc}*")
        with st.expander("ℹ️ Standard", expanded=False):
            st.caption(TUTORIAL.get(name, "—"))
        maxv = float(ref)*1.2 if (unit not in ['%', 'sec']) else (float(ref)*1.5 if unit=='sec' else 30.0)
        if section.startswith("Simmetrie") and ref == 0:
            maxv = 30.0 if unit != "%" else 30.0
        val = st.slider(f"{unit} (rif. {ref})", min_value=0.0, max_value=float(maxv), value=0.0, step=0.1, key=f"{section}_{name}")
        if section.startswith("Simmetrie"):
            if unit == "%": score = max(0.0, 10.0 * (1.0 - min(val, 20.0)/20.0))
            else:          score = max(0.0, 10.0 * (1.0 - min(val, 10.0)/10.0))
        elif section == "Neurodinamica":
            score = score_linear(val, max(ref, 1))
        else:
            score = score_linear(val, ref)
        st.markdown(badge(f"Punteggio: {score:.1f}/10", score_color(score)), unsafe_allow_html=True)
        st.divider()
        rows.append([name, unit, ref, val, score])

    df = pd.DataFrame(rows, columns=["Test","Unità","Riferimento","Valore","Score"])
    df_meas = df[df["Valore"] > 0]
    pct = int((len(df_meas) / len(df) * 100)) if not df.empty else 0
    st.progress(pct); st.caption(f"Completati: {len(df_meas)} su {len(df)} test ({pct}%)")

    st.markdown("### Radar")
    fig = radar_plot(df_meas["Test"].tolist(), df_meas["Score"].tolist(), f"{section} – Punteggi (0–10)")
    st.plotly_chart(fig, use_container_width=True)
    mean_score = df_meas["Score"].mean() if not df_meas.empty else 0.0
    st.markdown(badge(f"Media sezione: {mean_score:.1f} / 10", score_color(mean_score)), unsafe_allow_html=True)

    st.markdown("")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("💾 Salva"):
            save_session(st.session_state.athlete, section, st.session_state.date, df)
            st.success("Sessione salvata.")
    with c2:
        if st.button("⬇️ JSON"):
            js = export_session_json(st.session_state.athlete, section, st.session_state.date, df)
            st.download_button("Scarica JSON", data=js, file_name=f"{st.session_state.athlete.replace(' ','')}_{section}_{st.session_state.date}.json", mime="application/json")
    with c3:
        disabled_pdf = df_meas.empty
        if st.button("🧾 PDF", disabled=disabled_pdf):
            pdf = pdf_single_section(load_logo(), st.session_state.athlete, st.session_state.evaluator, st.session_state.date, section, df)
            st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{section}_{st.session_state.athlete.replace(' ','')}_{st.session_state.date}.pdf", mime="application/pdf")

def load_logo():
    try:
        with open(DEFAULT_LOGO_PATH, "rb") as f: return f.read()
    except Exception:
        return None

def pdf_single_section(logo_bytes, athlete, evaluator, date_str, section_name, df):
    disp = df[df["Valore"] > 0].copy()
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4, rightMargin=1.2*cm, leftMargin=1.2*cm, topMargin=1.2*cm, bottomMargin=1.4*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=14)
    story = []
    story.append(Paragraph("Fisiomove Pro 3.3 – Mobile", title))
    story.append(Spacer(1, 6))
    head = Table([["Atleta", athlete, "Data", date_str, "Valutatore", evaluator]], colWidths=[2*cm, 6*cm, 2*cm, 3*cm, 2.5*cm, 4*cm])
    head.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(head); story.append(Spacer(1, 8))
    if logo_bytes:
        try:
            story.append(Image(ImageReader(io.BytesIO(logo_bytes)), width=5.5*cm, height=5.5*cm)); story.append(Spacer(1, 6))
        except Exception:
            pass
    story.append(Paragraph(f"Sezione testata: <b>{section_name}</b>", styles['Heading3']))
    if disp.empty:
        story.append(Paragraph("Nessun test compilato (valori > 0).", normal))
    else:
        disp["Score"] = disp["Score"].round(1)
        table_data = [["Test","Unità","Rif.","Valore","Punteggio (0–10)"]] + disp.values.tolist()
        tbl = Table(table_data, colWidths=[7*cm,1.5*cm,2.0*cm,2.2*cm,3.1*cm])
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("ALIGN",(2,1),(-1,-1),"CENTER")]))
        story.append(tbl); story.append(Spacer(1, 8))
        rbuf = radar_png_buffer(disp["Test"].tolist(), disp["Score"].tolist())
        try:
            img_reader = ImageReader(io.BytesIO(rbuf.getvalue()))
            story.append(Image(img_reader, width=10.5*cm, height=10.5*cm))
            story.append(Paragraph(f"<i>Radar clinico – {section_name}</i>", styles['Normal']))
            story.append(Spacer(1, 6))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Radar non disponibile ({str(e)}).", normal))
            story.append(Spacer(1, 6))
        story.append(Paragraph(ebm_comment(section_name, disp).replace("\n","<br/>"), normal))
    doc.build(story); buf_pdf.seek(0); return buf_pdf

def page_riepilogo():
    st.markdown("### Riepilogo (Semaforo)")
    section = st.session_state.section
    rows = []
    for name, unit, ref, _ in TESTS[section]:
        key = f"{section}_{name}"
        val = float(st.session_state.get(key, 0.0))
        if section.startswith("Simmetrie"):
            if unit == "%": score = max(0.0, 10.0 * (1.0 - min(val, 20.0)/20.0))
            else: score = max(0.0, 10.0 * (1.0 - min(val, 10.0)/10.0))
        elif section == "Neurodinamica":
            score = score_linear(val, max(ref, 1))
        else:
            score = score_linear(val, ref)
        rows.append([name, unit, ref, val, score])
    df = pd.DataFrame(rows, columns=["Test","Unità","Riferimento","Valore","Score"])
    df_meas = df[df["Valore"] > 0]
    if df_meas.empty:
        st.info("Compila almeno un test nella pagina Valutazione."); return
    df_sorted = df_meas.sort_values("Score")
    st.dataframe(df_sorted[["Test","Unità","Valore","Score"]].style.format({"Score":"{:.1f}"}), use_container_width=True)
    worst = df_sorted.head(3)
    if not worst.empty:
        st.markdown("### Priorità (Top 3)")
        for _, row in worst.iterrows():
            st.markdown(badge(f"{row['Test']} – {row['Score']:.1f}", score_color(row["Score"])), unsafe_allow_html=True)

def page_storico():
    st.markdown("### Storico e confronto radar")
    athlete = st.session_state.athlete; section = st.session_state.section
    data_hist = load_sessions(athlete, section)
    if data_hist.empty:
        st.info("Nessuna sessione salvata per questo atleta/sezione."); return
    dates = data_hist["date"].tolist()
    pick = st.multiselect("Seleziona fino a 3 date", dates, default=dates[:2])
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
        fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target", line=dict(width=1, dash="dot", color="#999")))
    for d, dfc in comp:
        scores_map = {t:s for t,s in zip(dfc["Test"], dfc["Score"])}
        series = [scores_map.get(t, 0) for t in labels]
        series_c = series + ([series[0]] if labels else [])
        fig.add_trace(go.Scatterpolar(r=series_c, theta=(labels + [labels[0]] if labels else []), name=d))
    fig.update_layout(showlegend=True, title=f"Confronto – {section}", polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                      margin=dict(l=10,r=10,t=40,b=10), height=520)
    st.plotly_chart(fig, use_container_width=True)

# ----------------- ROUTER -----------------
if "page" not in st.session_state: st.session_state.page = "Valutazione"
def render_page():
    if st.session_state.page == "Valutazione": page_valutazione()
    elif st.session_state.page == "Riepilogo": page_riepilogo()
    elif st.session_state.page == "Storico": page_storico()
render_page()

# ----------------- NAVBAR INFERIORE -----------------
def nav_button(label, key, active_key):
    cls = "navbtn active" if (st.session_state.page == active_key) else "navbtn"
    st.markdown(f"<div class='{cls}'>{label}</div>", unsafe_allow_html=True)
    if st.button(" ", key=key):
        st.session_state.page = active_key
        st.experimental_rerun()

st.markdown("<div class='navbar'><div class='navgrid'>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1: nav_button("🧍 Valutazione", "nav_val", "Valutazione")
with c2: nav_button("📊 Riepilogo", "nav_riep", "Riepilogo")
with c3: nav_button("🕓 Storico", "nav_sto", "Storico")
st.markdown("</div></div>", unsafe_allow_html=True)
