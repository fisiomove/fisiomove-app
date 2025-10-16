# Fisiomove Pro 2.0 – Streamlit app
import io
import datetime as dt
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
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fisiomove Pro 2.0 – Mobilità EBM", layout="wide")
PRIMARY = "#004A9F"
DEFAULT_LOGO_PATH = r"/mnt/data/fisio move 2.png"

def score_linear(value: float, ref: float) -> float:
    try:
        v = float(value); r = float(ref)
        if r <= 0: return 0.0
        return max(0.0, min(10.0, (v/r)*10.0))
    except Exception:
        return 0.0

def radar_plot(labels, scores, title):
    labels_c = list(labels) + [labels[0]]
    scores_c = list(scores) + [scores[0]]
    ideal = [10]*len(labels); ideal_c = ideal + [ideal[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ideal_c, theta=labels_c, name="Target", line=dict(width=1, dash="dot", color="#999")))
    fig.add_trace(go.Scatterpolar(r=scores_c, theta=labels_c, name="Atleta", fill="toself", line=dict(width=2, color=PRIMARY)))
    fig.update_layout(title=title, showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center"),
        polar=dict(radialaxis=dict(visible=True, range=[0,10], tickmode="linear", dtick=1)),
        margin=dict(l=10,r=10,t=40,b=10), height=460)
    return fig

def radar_png_buffer(labels, scores):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]
    scores_c = list(scores) + [scores[0]]
    ideal_c = [10]*N + [10]
    fig, ax = plt.subplots(figsize=(4.2,4.2), subplot_kw=dict(polar=True))
    ax.plot(angles_c, ideal_c, color="#777", linewidth=1, linestyle="dotted")
    ax.plot(angles_c, scores_c, color="#004A9F", linewidth=2)
    ax.fill(angles_c, scores_c, color="#2f6fb640")
    ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10]); ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout(); plt.savefig(buf, format="png", dpi=200); plt.close(fig); buf.seek(0)
    return buf

def ebm_comment_clinical(section_name, df):
    low = df[df["Score"] < 4]["Test"].tolist()
    mid = df[(df["Score"] >= 4) & (df["Score"] <= 7)]["Test"].tolist()
    high = df[df["Score"] > 7]["Test"].tolist()
    out = [f"Sintesi clinica – {section_name}:"]
    if low: out.append("• Deficit clinicamente rilevanti in: " + ", ".join(low))
    if mid: out.append("• Aree migliorabili: " + ", ".join(mid))
    if high: out.append("• Aree adeguate/ottimali: " + ", ".join(high))
    if section_name == "Squat":
        out.append("EBM: dorsiflessione in carico ↔ profondità/controllo (Bennell 1998); estensione toracica ↔ inclinazione tronco (Mansfield 2008); ER spalla rilevante per low‑bar (Wilk 2014).")
    if section_name == "Panca":
        out.append("EBM: estensione toracica e ER spalla influenzano arco e stress gleno‑omerale (Mansfield 2008; Riddle 1987). Pettorale minore e controllo scapolare modulano cinematica scapolo‑toracica (Borstad 2006; Hardwick 2014). Thomas modificato supporta la postura lombopelvica (Harvey 1998).")
    if section_name == "Deadlift":
        out.append("EBM: AKE/SLR e mobilità lombare (Schober) correlano al hinge e tolleranza al carico (Gajdosik 1983; Youdas 1993; Tousignant 2005). Dorsiflessione in carico supporta baricentro (Bennell 1998). Endurance del tronco predittiva (Biering‑Sørensen 1984).")
    out.append("Traduzione per l'atleta: concentrati sui test con punteggio <7; obiettivo portare tutti i valori ≥8/10 con tecnica pulita.")
    return "\\n".join(out)

def pdf_single_section(logo_bytes, athlete, evaluator, date_str, section_name, df):
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=14)

    story = []
    story.append(Paragraph("Fisiomove Pro 2.0 – Mobilità EBM", title))
    story.append(Spacer(1, 6))

    head = Table([["Atleta", athlete, "Data", date_str, "Valutatore", evaluator]], colWidths=[2*cm, 6*cm, 2*cm, 3*cm, 2.5*cm, 4*cm])
    head.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(head); story.append(Spacer(1, 8))

    # Logo
    used_logo = None
    if logo_bytes:
        used_logo = io.BytesIO(logo_bytes)
    else:
        try:
            used_logo = open(DEFAULT_LOGO_PATH, "rb")
        except Exception:
            used_logo = None
    if used_logo:
        try:
            story.append(Image(ImageReader(used_logo), width=6*cm, height=8*cm)); story.append(Spacer(1, 6))
        except Exception:
            pass

    story.append(Paragraph(f"Sezione testata: <b>{section_name}</b>", styles['Heading3']))

    disp = df.copy(); disp["Score"] = disp["Score"].round(1)
    table_data = [["Test","Unità","Riferimento","Valore","Punteggio (0–10)"]] + disp.values.tolist()
    tbl = Table(table_data, colWidths=[7*cm,1.5*cm,2.2*cm,2.2*cm,3*cm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.25,colors.grey),("ALIGN",(2,1),(-1,-1),"CENTER")]))
    story.append(tbl); story.append(Spacer(1, 8))

    rbuf = radar_png_buffer(disp["Test"].tolist(), disp["Score"].tolist())
    img_bytes = rbuf.getvalue()
    story.append(Image(ImageReader(io.BytesIO(img_bytes)), width=10*cm, height=10*cm)); story.append(Spacer(1, 6))

    comment = ebm_comment_clinical(section_name, disp).replace("\\n","<br/>")
    story.append(Paragraph(comment, normal))

    doc.build(story); buf_pdf.seek(0); return buf_pdf

# ---- DATA ----
tests_map = {
    "Squat": [
        ("Weight Bearing Lunge Test","cm",10,"Dorsiflessione in carico del complesso tibio‑tarsico."),
        ("Passive Hip Flexion","°",120,"Flessione passiva d'anca; profondità di accosciata."),
        ("Hip Rotation (flexed 90°)","°",40,"Rotazione anca a 90° di flessione."),
        ("Thoracic Extension (T4–T12)","°",30,"Estensione toracica segmentale."),
        ("Shoulder ER (adducted, low‑bar)","°",70,"Posizione low‑bar del bilanciere."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)","°",180,"Flessione gleno‑omerale; overhead ability."),
        ("External Rotation (90° abd)","°",90,"ER spalla a 90° abduzione."),
        ("Thoracic Extension (T4–T12)","°",30,"Arco toracico per la panca."),
        ("Pectoralis Minor Length","cm",10,"Lunghezza pettorale minore (spazio acromiale)."),
        ("Wall Angel (distance)","cm",10,"Controllo scapolare a parete."),
        ("Thomas Test (modified)","°",10,"Estensione d'anca per postura lombopelvica."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)","°",90,"Estensibilità ischiocrurali in catena aperta."),
        ("Straight Leg Raise (SLR)","°",90,"Catena posteriore; flessibilità neural/muscolare."),
        ("Weight Bearing Lunge Test","cm",10,"Dorsiflessione caviglia in carico."),
        ("Modified Schober (lumbar)","cm",5,"Mobilità lombare in flessione."),
        ("Sorensen Endurance","sec",180,"Endurance estensori lombari."),
    ],
}

# ---- UI ----
col_logo, col_title = st.columns([1,2])
with col_logo:
    try: st.image(DEFAULT_LOGO_PATH, use_container_width=True)
    except Exception: st.markdown("### Fisiomove")
with col_title:
    st.markdown("<h1 style='margin-bottom:0'>Fisiomove Pro 2.0</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{PRIMARY}; font-size:18px;'>Protocollo di valutazione della mobilità (EBM)</div>", unsafe_allow_html=True)
    st.caption("Analisi delle alzate (Squat, Panca, Deadlift) con punteggi normalizzati 0–10 e radar comparativo.")

st.markdown("---")

with st.sidebar:
    st.header("Dati atleta")
    athlete = st.text_input("Nome", "Mario Rossi")
    evaluator = st.text_input("Valutatore", "Dott. Alessandro Ferreri – Fisioterapista")
    date_str = st.date_input("Data", value=dt.date.today()).strftime("%Y-%m-%d")
    st.markdown("---")
    st.header("Sezione da valutare")
    section = st.selectbox("Scegli una sezione", ["Squat","Panca","Deadlift"])
    st.caption("Il PDF includerà solo la sezione selezionata.")

st.subheader(f"Sezione: {section}")
st.write("Inserisci i valori (°/cm/sec). Il punteggio 0–10 è lineare e normalizzato rispetto al riferimento EBM.")

rows = []
cols = st.columns(2)
for i,(name,unit,ref,desc) in enumerate(tests_map[section]):
    with cols[i % 2]:
        st.markdown(f"**{name}** · *{desc}*")
        maxv = float(ref)*1.2 if unit != "sec" else float(ref)*1.5
        val = st.slider(f"{unit} (rif. {ref})", min_value=0.0, max_value=float(maxv), value=0.0, step=0.1, key=f"{section}_{name}")
        score = score_linear(val, ref)
        st.progress(min(100, int(score*10)))
        st.caption(f"Punteggio: **{score:.1f}/10**")
        rows.append([name, unit, ref, val, score])

df = pd.DataFrame(rows, columns=["Test","Unità","Riferimento","Valore","Score"])
mean_score = df["Score"].mean() if not df.empty else 0.0

st.markdown("### Radar")
fig = radar_plot(df["Test"].tolist(), df["Score"].tolist(), f"{section} – Punteggi (0–10)")
st.plotly_chart(fig, use_container_width=True)
st.metric("Media sezione", f"{mean_score:.1f} / 10")

st.markdown("---")
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Referto PDF (sezione corrente)")
    if st.button("Genera PDF della sezione"):
        try:
            with open(DEFAULT_LOGO_PATH, "rb") as f:
                logo_bytes = f.read()
        except Exception:
            logo_bytes = None
        pdf = pdf_single_section(logo_bytes, athlete, evaluator, date_str, section, df)
        st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{section}_{athlete.replace(' ','')}_{date_str}.pdf", mime="application/pdf")
with col2:
    st.info("Obiettivo: portare tutti i punteggi ≥ **8/10**. Priorità clinica ai test < **7/10**.")
