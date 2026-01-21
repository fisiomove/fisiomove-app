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
matplotlib.use("Agg")  # backend headless-safe
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
    # genera key breve e stabile per Streamlit
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
    "ULNT1A (Median nerve)": "Test neurodinamico mediano (ULNT1A)",
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
# Plots: Matplotlib (per PDF) kept; UI uses Plotly for nicer visuals
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


# -----------------------------
# UI Plotly (interactive, prettier)
# -----------------------------
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
# EBM comments
# -----------------------------
def ebm_from_df(df, friendly=False):
    notes = []
    ebm_library = {
        "Weight Bearing Lunge Test": {
            "ref": "Bennell KL et al., 1998; Konor MM et al., 2012",
            "low_score": "Mobilità della caviglia ridotta: rischio di compensi, sollevamento del tallone e stress femoro-rotuleo.",
            "friendly": "Il test della caviglia è ridotto: potresti avere difficoltà nello squat profondo.",
        },
        "Passive Hip Flexion": {
            "ref": "Reese NB, Bandy WD, 2020",
            "low_score": "Flessibilità anca ridotta: può limitare la profondità dello squat.",
            "friendly": "La flessione dell’anca è un po’ limitata.",
        },
        "Shoulder ER (adducted, low-bar)": {
            "ref": "Wilk KE et al., 2015",
            "low_score": "Rotazione esterna spalla ridotta: può influenzare la posizione low-bar.",
            "friendly": "La spalla ruota un po’ meno del normale.",
        },
        "Wall Angel Test": {
            "ref": "Kibler WB et al., 2013; Ludewig PM et al., 2009",
            "low_score": "Deficit nel controllo scapolare e nella mobilità toracica: possibili compensi nella postura o nei movimenti overhead.",
        },
        "ULNT1A (Median nerve)": {
            "ref": "Nee RJ et al., 2012",
            "low_score": "Test positivo: possibile irritazione o tensione del nervo mediano.",
            "friendly": "Test sul nervo del braccio positivo: potresti sentire tensione o fastidio.",
        },
    }

    problematic_tests = {}

    for _, r in df.iterrows():
        test = str(r["Test"])
        score = float(r["Score"]) if not pd.isna(r["Score"]) else 10.0
        pain = bool(r["Dolore"])
        sym = float(r.get("SymScore", 10.0) if not pd.isna(r.get("SymScore", np.nan)) else 10.0)

        comment_lines = []
        issue = False

        if score < 4:
            msg = ebm_library.get(test, {}).get("friendly" if friendly else "low_score",
                                                f"Deficit rilevato nel test '{test}'.")
            comment_lines.append(f"❗ {msg}")
            issue = True
        if sym < 7:
            comment_lines.append(f"Asimmetria significativa nel test '{test}' (SymScore: {sym:.1f}/10).")
            issue = True
        if pain:
            comment_lines.append("Dolore riportato durante il test.")

        if not issue:
            comment_lines.append(f"Il test '{test}' soddisfa la sufficienza.")

        notes.extend(comment_lines)

        if issue:
            problematic_tests[test] = ebm_library.get(test, {}).get("ref")

    if not friendly and problematic_tests:
        notes.append("")
        for test, ref in problematic_tests.items():
            if ref:
                notes.append(f"Riferimento: {ref}")

    return notes


# -----------------------------
# PDF generation (keeps matplotlib buffers)
# -----------------------------
def pdf_report_no_bodychart(logo_bytes, athlete, evaluator, date_str, section, df, ebm_notes, radar_buf=None, asym_buf=None):
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]

    story = []
    story.append(RLImage(io.BytesIO(logo_bytes), width=14 * cm, height=3.5 * cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Report Valutazione – {sanitize_text_for_plot(section)}</b>", title))
    story.append(Spacer(1, 6))

    info_data = [["Atleta", athlete, "Valutatore", evaluator, "Data", date_str]]
    info_table = Table(info_data, colWidths=[2.2 * cm, 5.0 * cm, 2.8 * cm, 5.0 * cm, 1.8 * cm, 2.0 * cm])
    info_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 8))

    disp = df[["Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]].copy()
    disp = disp[~disp["Test"].str.lower().str.contains("schober", na=False)]
    for col in ["Valore", "Score", "Dx", "Sx", "Delta", "SymScore"]:
        disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)

    table = Table([disp.columns.tolist()] + disp.values.tolist(), repeatRows=1,
                  colWidths=[2.2 * cm, 6.5 * cm, 1.2 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.6 * cm, 1.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 10))

    if radar_buf:
        story.append(Paragraph("<b>Radar – Punteggi (0–10)</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10 * cm, height=10 * cm))
        story.append(Spacer(1, 8))

    if asym_buf:
        story.append(Paragraph("<b>Grafico Asimmetrie Dx/Sx</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14 * cm, height=6 * cm))
        story.append(Spacer(1, 8))

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
    story.append(Paragraph("<b>Regioni dolorose riscontrate durante il test:</b>", normal))
    if pain_regions:
        for reg in pain_regions:
            story.append(Paragraph(f"• {reg.capitalize()}", normal))
    else:
        story.append(Paragraph("Nessuna regione segnalata come dolorosa.", normal))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Commento clinico (EBM)</b>", normal))
    story.append(Spacer(1, 4))
    if ebm_notes:
        for note in ebm_notes:
            if isinstance(note, str):
                story.append(Paragraph(f"• {sanitize_text_for_plot(note)}", normal))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("Nessun commento disponibile.", normal))

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
        buf, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]

    story = []
    story.append(RLImage(io.BytesIO(logo_bytes), width=14 * cm, height=3.5 * cm))
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
# UI: Styling & Header
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
    # placeholder metrics: will be updated after df_show is computed
    metric_container = st.container()


# -----------------------------
# Sidebar – dati atleta e sezione (solo Valutazione Generale)
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
# Rendering input: grouped by regione inside expanders, each test as small card
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
    # group by region for nicer UI
    region_map = {}
    for sec, name, unit, ref, bilat, region, desc in tests:
        region_map.setdefault(region or "other", []).append((sec, name, unit, ref, bilat, region, desc))

    for region, items in region_map.items():
        with st.expander(region.capitalize(), expanded=False):
            for sec, name, unit, ref, bilat, region, desc in items:
                rec = st.session_state["vals"].get(name)
                if not rec:
                    continue
                # card wrapper (HTML)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{name}**  \n*{desc}*  \n*Rif:* {ref} {unit}")
                key = short_key(name)
                # special max for ULNT1A
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
                    # status badge
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


# render inputs
render_inputs_for_section(st.session_state["section"])


# -----------------------------
# Build DF and main visualizations
# -----------------------------
df_show = build_df(st.session_state["section"])
st.markdown("### Risultati")
# styled table: color rows by score
def style_scores_df(df):
    def color_row(v):
        if pd.isna(v):
            return ""
        if v < 4:
            return "background-color:#fff1f2"  # redish
        if v < 7:
            return "background-color:#fffbeb"  # yellowish
        return "background-color:#ecfdf5"  # greenish

    sty = df.style.applymap(lambda x: color_row(x) if isinstance(x, (int, float)) else "", subset=["Score"])
    return sty

st.write(df_show.style.format(precision=1))


# update header metrics
with metric_container:
    avg_score = df_show["Score"].mean() if not df_show["Score"].isna().all() else 0.0
    painful = int(df_show["Dolore"].sum()) if "Dolore" in df_show.columns else 0
    sym_mean = df_show["SymScore"].mean() if "SymScore" in df_show.columns else np.nan
    st.metric("Score medio", f"{avg_score:.1f}/10")
    st.metric("Test con dolore", f"{painful}")
    st.metric("Symmetry medio", f"{sym_mean:.1f}/10" if not pd.isna(sym_mean) else "n/a")


# interactive plots
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
# Prepare PDF buffers (matplotlib) for download
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
# EBM notes (calcolate prima di PDF)
# -----------------------------
ebm_notes = ebm_from_df(df_show, friendly=False)


# -----------------------------
# Export PDF buttons
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
