
import io
import math
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

st.set_page_config(page_title="Fisiomove – Protocollo Clinico EBM", layout="wide")

# -------------------- Helpers --------------------
def score_linear(value: float, ref: float) -> float:
    """Linear normalization to 0–10 with cap at 10."""
    try:
        v = float(value)
        r = float(ref)
        if r <= 0:
            return 0.0
        return max(0.0, min(10.0, (v / r) * 10.0))
    except Exception:
        return 0.0

def radar_plotly(labels, scores, title):
    labels_c = list(labels) + [labels[0]]
    scores_c = list(scores) + [scores[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=scores_c,
        theta=labels_c,
        fill='toself',
        line=dict(width=2),
        name=title
    ))
    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0,10], tickmode="linear", dtick=1)),
        showlegend=False,
        height=450,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

def radar_matplotlib_image(labels, scores):
    """Create a radar chart using matplotlib and return it as PNG bytes for embedding in PDF."""
    # Prepare angles
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    scores_c = list(scores) + [scores[0]]
    angles_c = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.0, 4.0), subplot_kw=dict(polar=True))
    ax.plot(angles_c, scores_c, linewidth=2)
    ax.fill(angles_c, scores_c, alpha=0.25)
    ax.set_yticks([2,4,6,8,10])
    ax.set_ylim(0,10)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf

def ebm_comment(section_name: str, df_scores: pd.DataFrame) -> str:
    low = df_scores[df_scores["Score"] < 4]["Test"].tolist()
    mid = df_scores[(df_scores["Score"] >= 4) & (df_scores["Score"] <= 7)]["Test"].tolist()
    high = df_scores[df_scores["Score"] > 7]["Test"].tolist()
    parts = [f"Sintesi ({section_name}):"]
    if low:
        parts.append("• Limitazioni clinicamente rilevanti in: " + ", ".join(low))
    if mid:
        parts.append("• Aree migliorabili: " + ", ".join(mid))
    if high:
        parts.append("• Aree adeguate/ottimali: " + ", ".join(high))
    if "Panca" in section_name:
        parts.append("EBM: Mobilità toracica e rotazione esterna di spalla influenzano arco e stress gleno‑omerale (Mansfield 2008; Riddle 1987). "
                     "Lunghezza del pettorale minore e controllo scapolare (Wall Angel) modulano la cinematica scapolo‑toracica (Borstad 2006; Hardwick 2014). "
                     "Thomas modificato contribuisce alla postura lombopelvica (Harvey 1998).")
    if "Deadlift" in section_name:
        parts.append("EBM: Catena posteriore (AKE/SLR) e mobilità lombare (Schober) sono correlate alla qualità del hinge e tolleranza al carico (Gajdosik 1983; Youdas 1993; Tousignant 2005). "
                     "Dorsiflessione in carico supporta la gestione del baricentro (Bennell 1998). Endurance del tronco ha valore prognostico (Biering‑Sørensen 1984).")
    if "Squat" in section_name:
        parts.append("EBM: Dorsiflessione in carico associata a profondità/controllo (Bennell 1998). "
                     "Estensione toracica influenza l'inclinazione del tronco (Mansfield 2008). "
                     "Rotazione esterna di spalla rilevante per low‑bar (Wilk 2014).")
    return "\n".join(parts)

def section_block(section_name, tests):
    st.subheader(section_name)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Inserisci i valori misurati** (unità indicate)")
        rows = []
        for tname, unit, ref in tests:
            val = st.number_input(f"{tname} [{unit}] — riferimento {ref}", min_value=0.0, step=0.1, value=0.0, key=f"{section_name}_{tname}")
            score = score_linear(val, ref)
            rows.append([tname, unit, ref, val, score])
        df = pd.DataFrame(rows, columns=["Test", "Unità", "Riferimento", "Valore", "Score"])
        df_display = df.copy()
        df_display["Score"] = df_display["Score"].round(1)
        st.dataframe(df_display, use_container_width=True)
        mean_score = df["Score"].mean() if len(df) else 0.0
        st.metric("Punteggio medio", f"{mean_score:.1f} / 10")
    with cols[1]:
        fig = radar_plotly(df["Test"].tolist(), df["Score"].tolist(), f"Radar – {section_name}")
        st.plotly_chart(fig, use_container_width=True, key=f"plot_{section_name}")
    return df

def pdf_report(logo_file, athlete, evaluator, date_str, sections_dict):
    """Generate a multi-section PDF and return it as bytes."""
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4, rightMargin=1.5*cm, leftMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story = []
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor("#004A9F"))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=13)

    # Header
    story.append(Paragraph("Fisiomove – Protocollo di Valutazione Clinico EBM", title))
    story.append(Spacer(1, 6))
    head_tbl = Table([
        ["Atleta", athlete, "Valutatore", evaluator],
        ["Data", date_str, "", ""]
    ], colWidths=[2.5*cm, 8*cm, 2.5*cm, 3*cm])
    head_tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.25, colors.grey)]))
    story.append(head_tbl)
    story.append(Spacer(1, 10))

    if logo_file is not None:
        try:
            img_reader = ImageReader(logo_file)
            story.append(Image(img_reader, width=10*cm, height=3*cm))
            story.append(Spacer(1, 10))
        except Exception:
            pass

    # Sections
    for sec_name, df in sections_dict.items():
        story.append(Paragraph(f"Sezione: {sec_name}", styles['Heading3']))
        # Table of values
        display = df.copy()
        display["Score"] = display["Score"].round(1)
        table_data = [["Test", "Unità", "Riferimento", "Valore", "Score"]] + display.values.tolist()
        tbl = Table(table_data, colWidths=[7*cm, 1.5*cm, 2.2*cm, 2.2*cm, 2.1*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#004A9F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (2,1), (-1,-1), "CENTER")
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))
        # Radar image
        radar_buf = radar_matplotlib_image(display["Test"].tolist(), display["Score"].tolist())
        radar_img = radar_buf.getvalue()
        story.append(Image(io.BytesIO(radar_img), width=10*cm, height=10*cm))
        story.append(Spacer(1, 6))
        # Comment
        comment = ebm_comment(sec_name, display)
        story.append(Paragraph(comment.replace("\n", "<br/>"), normal))
        story.append(Spacer(1, 12))

    doc.build(story)
    buf_pdf.seek(0)
    return buf_pdf

# -------------------- App UI --------------------
st.title("Fisiomove – Protocollo Clinico EBM (Streamlit)")

with st.sidebar:
    st.header("Dati Atleta")
    athlete = st.text_input("Nome atleta", value="Mario Rossi")
    sport = st.text_input("Sport", value="CrossFit / PL")
    evaluator = st.text_input("Valutatore", value="Dott. Alessandro Ferreri – Fisioterapista")
    date_str = st.date_input("Data", value=dt.date.today()).strftime("%Y-%m-%d")
    st.markdown("---")
    logo_file = st.file_uploader("Logo Fisiomove (PNG/JPG)", type=["png", "jpg", "jpeg"])

tabs = st.tabs(["Squat", "Panca", "Deadlift"])

# ----- Squat tests -----
with tabs[0]:
    squat_tests = [
        ("Weight Bearing Lunge Test", "cm", 10),
        ("Passive Hip Flexion", "°", 120),
        ("Hip Rotation (flexed 90°)", "°", 40),
        ("Thoracic Extension (T4–T12)", "°", 30),
        ("Shoulder ER (adducted, low-bar)", "°", 70),
    ]
    df_squat = section_block("Squat", squat_tests)

# ----- Bench tests -----
with tabs[1]:
    bench_tests = [
        ("Shoulder Flexion (supine)", "°", 180),
        ("External Rotation (90° abd)", "°", 90),
        ("Thoracic Extension (T4–T12)", "°", 30),
        ("Pectoralis Minor Length", "cm", 10),
        ("Wall Angel (distance)", "cm", 10),
        ("Thomas Test (modified)", "°", 10),
    ]
    df_bench = section_block("Panca", bench_tests)

# ----- Deadlift tests -----
with tabs[2]:
    dead_tests = [
        ("Active Knee Extension (AKE)", "°", 90),
        ("Straight Leg Raise (SLR)", "°", 90),
        ("Weight Bearing Lunge Test", "cm", 10),
        ("Modified Schober (lumbar)", "cm", 5),
        ("Sorensen Endurance", "sec", 180),
    ]
    df_dead = section_block("Deadlift", dead_tests)

st.markdown("---")
col_pdf1, col_pdf2 = st.columns([1,2])
with col_pdf1:
    st.subheader("Referto PDF")
    st.write("Genera un PDF multipagina con logo, tabelle, radar e commento clinico EBM (leggibile per l’atleta).")
    if st.button("Genera PDF"):
        sections = {"Squat": df_squat, "Panca": df_bench, "Deadlift": df_dead}
        pdf_bytes = pdf_report(logo_file, athlete, evaluator, date_str, sections)
        st.download_button("Scarica Referto PDF", data=pdf_bytes, file_name=f"Referto_Fisiomove_{athlete.replace(' ','')}_{date_str}.pdf", mime="application/pdf")

with col_pdf2:
    st.info("Consiglio: inserisci i valori al **decimo** (es. 8.5 cm) per massima sensibilità; punteggio lineare normalizzato a 10 per ogni test. I grafici mostrano l’intervallo 0–10.")
