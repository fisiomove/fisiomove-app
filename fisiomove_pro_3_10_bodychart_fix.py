
import io, os, random, math, base64
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Fisiomove Pro 3.10", layout="centered")

# -----------------------------
# Constants & Assets
# -----------------------------
APP_TITLE = "Fisiomove Pro 3.10 — BodyChart Fix"
PRIMARY = "#1E6CF4"

# Try loading assets from working dir
LOGO_PATHS = ["logo 2600x1000.jpg", "logo.png", "logo.jpg"]
BODYCHART_PATHS = ["8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png", "body_chart.png"]

def load_logo_bytes():
    for p in LOGO_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    # placeholder
    img = Image.new("RGB", (1000, 260), (30, 108, 244))
    d = ImageDraw.Draw(img)
    d.text((30, 100), "Fisiomove", fill=(255,255,255))
    bio = io.BytesIO(); img.save(bio, format="PNG"); return bio.getvalue()

def load_bodychart_image():
    for p in BODYCHART_PATHS:
        if os.path.exists(p):
            try:
                return Image.open(p).convert("RGBA")
            except Exception:
                pass
    # placeholder
    img = Image.new("RGBA", (1200, 800), (245,245,245,255))
    d = ImageDraw.Draw(img)
    d.text((20,20), "Aggiungi body_chart.png (anteriore a sinistra, posteriore a destra)", fill=(10,10,10))
    return img

LOGO = load_logo_bytes()
BODYCHART_BASE = load_bodychart_image()

# -----------------------------
# Tests database (EBM-like) — name, unit, ref, bilateral, region, desc
# -----------------------------
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10.0, True,  "ankle",    "Dorsiflessione in carico (WB lunge)."),
        ("Passive Hip Flexion",       "°",  120.0, True, "hip",      "Flessione d’anca passiva supina."),
        ("Hip Rotation (flexed 90°)", "°",   40.0, True, "hip",      "Rotazione anca a 90° flessione."),
        ("Thoracic Extension (T4-T12)","°",  30.0, False,"thoracic", "Estensione toracica globale."),
        ("Shoulder ER (adducted, low-bar)","°",70.0, True,"shoulder","ER spalla per low-bar."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Flessione spalla, scapole stabili."),
        ("External Rotation (90° abd)","°", 90.0, True, "shoulder", "ER a 90° abd (capsula anteriore)."),
        ("Thoracic Extension (T4-T12)","°", 30.0, False,"thoracic", "Estensione toracica per setup."),
        ("Pectoralis Minor Length",   "cm", 10.0, True, "shoulder", "Lunghezza piccolo pettorale."),
        ("Thomas Test (modified)",    "°",  10.0, True, "hip",      "Flesso-estensibilità flessori anca."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)","°", 90.0, True, "knee",     "Estensione attiva ginocchio."),
        ("Straight Leg Raise (SLR)",  "°", 90.0, True, "hip",      "SLR catena posteriore."),
        ("Weight Bearing Lunge Test", "cm", 10.0, True, "ankle",    "Dorsiflessione in carico."),
        ("Modified Schober (lumbar)", "cm", 5.0, False,"lumbar",    "Mobilità lombare in flessione."),
        ("Sorensen Endurance",        "sec",180.0,False,"lumbar",   "Endurance estensori lombari."),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)",  "°", 90.0, True,  "hip",      "Catena neurodinamica posteriore LE."),
        ("Popliteal Knee Bend (PKB)", "°", 90.0, True,  "knee",     "Scorrimento distale nervo sciatico."),
        ("ULNT1A (Median nerve)",     "°", 90.0, True,  "shoulder", "Upper Limb Neurodynamic Test 1A."),
    ],
}
ALL_SECTIONS = ["Squat", "Panca", "Deadlift", "Neurodinamica", "Valutazione Generale"]

# -----------------------------
# Utility scoring
# -----------------------------
def ability_linear(val, ref):
    try:
        if ref <= 0: return 0.0
        s = (float(val) / float(ref)) * 10.0
        return max(0.0, min(10.0, s))
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
        score = 10.0 * max(0.0, 1.0 - min(diff, scale)/scale)
        return max(0.0, min(10.0, score))
    except Exception:
        return 0.0

# -----------------------------
# Session state
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
        st.session_state["section"] = "Squat"

init_state()

# Seed initial values if empty
def seed_defaults():
    if st.session_state["vals"]:
        return
    for sec, items in TESTS.items():
        for (name, unit, ref, bilat, region, desc) in items:
            if name not in st.session_state["vals"]:
                if bilat:
                    st.session_state["vals"][name] = {
                        "Dx": ref*0.9 if unit!="sec" else ref*0.8,
                        "Sx": ref*0.88 if unit!="sec" else ref*0.78,
                        "DoloreDx": False, "DoloreSx": False,
                        "unit": unit, "ref": ref, "bilat": True,
                        "region": region, "desc": desc, "section": sec
                    }
                else:
                    st.session_state["vals"][name] = {
                        "Val": ref*0.85,
                        "Dolore": False,
                        "unit": unit, "ref": ref, "bilat": False,
                        "region": region, "desc": desc, "section": sec
                    }

seed_defaults()

# -----------------------------
# Build DataFrame for a section (or all)
# -----------------------------
def build_df(section):
    rows = []
    for sec, items in TESTS.items():
        if section != "Valutazione Generale" and sec != section:
            continue
        for (name, unit, ref, bilat, region, desc) in items:
            rec = st.session_state["vals"].get(name)
            if not rec:
                continue
            if bilat:
                dx = float(rec.get("Dx", 0))
                sx = float(rec.get("Sx", 0))
                sc = ability_linear((dx+sx)/2.0, ref)
                sym = symmetry_score(dx, sx, unit)
                rows.append([sec, name, unit, ref, f"{(dx+sx)/2:.1f}", sc, dx, sx, abs(dx-sx), sym,
                             bool(rec.get("DoloreDx", False) or rec.get("DoloreSx", False)), region])
            else:
                val = float(rec.get("Val", 0))
                sc = ability_linear(val, ref)
                rows.append([sec, name, unit, ref, f"{val:.1f}", sc, "", "", "", "", bool(rec.get("Dolore", False)), region])
    df = pd.DataFrame(rows, columns=["Sezione","Test","Unità","Rif","Valore","Score","Dx","Sx","Delta","SymScore","Dolore","Regione"])
    return df

# -----------------------------
# Body chart rendering (PIL) – aggregate by region over ALL tests
# -----------------------------
def bodychart_image_from_state(width=1200, height=800):
    base = BODYCHART_BASE.copy().resize((width, height))
    draw = ImageDraw.Draw(base)

    # normalized anchors (front on the left, back on the right)
    fx = 0.255; bx = 0.745
    points = {
        # Front (left panel)
        "shoulder_dx": (fx-0.058, 0.225),
        "shoulder_sx": (fx+0.058, 0.225),
        "hip_dx":      (fx-0.038, 0.58),
        "hip_sx":      (fx+0.038, 0.58),
        "knee_dx":     (fx-0.028, 0.78),
        "knee_sx":     (fx+0.028, 0.78),
        "ankle_dx":    (fx-0.025, 0.94),
        "ankle_sx":    (fx+0.025, 0.94),
        # Back (right panel)
        "thoracic":    (bx, 0.33),
        "lumbar":      (bx, 0.52),
    }

    # Compute region scores and pain from ALL tests in state
    df_all = build_df("Valutazione Generale")
    region_scores = {}
    region_pain = {}
    for region in ["shoulder","hip","knee","ankle","thoracic","lumbar"]:
        sub = df_all[df_all["Regione"]==region]
        if len(sub)==0:
            region_scores[region]=0.0; region_pain[region]=False; continue
        region_scores[region] = float(np.clip(sub["Score"].astype(float).mean(), 0, 10))
        region_pain[region] = bool(sub["Dolore"].any())

    def score_color(score):
        if score > 7: return (22,163,74,255)   # green
        if score >= 4: return (245,158,11,255) # yellow
        return (220,38,38,255)                 # red

    def draw_marker(xn, yn, score, pain):
        x = int(xn*width); y = int(yn*height)
        radius = int(10 + 6*(1 - min(max(score,0),10)/10))
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=score_color(score))
        # white check for good
        if score > 7:
            draw.line((x-4, y, x-2, y+6), fill=(255,255,255,255), width=3)
            draw.line((x-2, y+6, x+6, y-4), fill=(255,255,255,255), width=3)
        # red triangle for pain
        if pain:
            tri = [(x+radius+2, y-radius-2), (x+radius+12, y-radius-2), (x+radius+7, y-radius-12)]
            draw.polygon(tri, fill=(220,38,38,255))

    # Place markers
    draw_marker(*points["shoulder_dx"], region_scores["shoulder"], region_pain["shoulder"])
    draw_marker(*points["shoulder_sx"], region_scores["shoulder"], region_pain["shoulder"])
    draw_marker(*points["hip_dx"],      region_scores["hip"],      region_pain["hip"])
    draw_marker(*points["hip_sx"],      region_scores["hip"],      region_pain["hip"])
    draw_marker(*points["knee_dx"],     region_scores["knee"],     region_pain["knee"])
    draw_marker(*points["knee_sx"],     region_scores["knee"],     region_pain["knee"])
    draw_marker(*points["ankle_dx"],    region_scores["ankle"],    region_pain["ankle"])
    draw_marker(*points["ankle_sx"],    region_scores["ankle"],    region_pain["ankle"])
    draw_marker(*points["thoracic"],    region_scores["thoracic"], region_pain["thoracic"])
    draw_marker(*points["lumbar"],      region_scores["lumbar"],   region_pain["lumbar"])

    bio = io.BytesIO(); base.save(bio, format="PNG"); bio.seek(0)
    return bio

# -----------------------------
# Radar plot (matplotlib) for a given dataframe
# -----------------------------
def radar_plot(df, title="Punteggi (0–10)"):
    labels = df["Test"].tolist()
    values = df["Score"].astype(float).tolist()

    if len(labels) == 0:
        raise ValueError("No data for radar.")

    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]
    angles = np.linspace(0, 2*np.pi, len(values_c), endpoint=False)

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 10)
    ax.plot(angles, values_c, linewidth=2, linestyle='solid')
    ax.fill(angles, values_c, alpha=0.25)
    ax.set_title(title)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="PNG", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------
# PDF report
# -----------------------------
def pdf_report(logo_bytes, athlete, evaluator, date_str, section, df, body_buf, ebm_notes, radar_buf=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.4*cm, rightMargin=1.4*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    title = styles["Title"]

    story = []
    story.append(RLImage(io.BytesIO(logo_bytes), width=16*cm, height=4*cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Report Valutazione – {section}</b>", title))
    story.append(Spacer(1, 6))

    info_data = [["Atleta", athlete, "Valutatore", evaluator, "Data", date_str]]
    info_table = Table(info_data, colWidths=[2.2*cm, 5.0*cm, 2.8*cm, 5.0*cm, 1.8*cm, 2.0*cm])
    info_table.setStyle(TableStyle([
        ("BOX",(0,0),(-1,-1),0.6,colors.lightgrey),
        ("INNERGRID",(0,0),(-1,-1),0.3,colors.lightgrey),
        ("BACKGROUND",(0,0),(-1,-1), colors.whitesmoke),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    disp = df[["Sezione","Test","Unità","Rif","Valore","Score","Dx","Sx","Delta","SymScore","Dolore"]].copy()
    table = Table([disp.columns.tolist()] + disp.values.tolist(), repeatRows=1, colWidths=[2.2*cm,6.5*cm,1.2*cm,1.2*cm,1.6*cm,1.6*cm,1.4*cm,1.4*cm,1.2*cm,1.6*cm,1.6*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR",(0,0),(-1,0), colors.white),
        ("ALIGN",(0,0),(-1,0),"CENTER"),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),8),
        ("GRID",(0,0),(-1,-1),0.25,colors.lightgrey),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    if radar_buf is not None:
        story.append(Paragraph("<b>Radar</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10*cm, height=10*cm))
        story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Body Chart – Sintesi</b>", normal))
    story.append(Spacer(1, 4))
    story.append(RLImage(io.BytesIO(body_buf.getvalue()), width=16*cm, height=12*cm))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Legenda: rosso=deficit; giallo=parziale; verde=buono; triangolo=Dolore.", normal))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Commento clinico (EBM)</b>", normal))
    for note in ebm_notes:
        story.append(Paragraph(f"• {note}", normal))

    doc.build(story)
    buf.seek(0)
    return buf

# -----------------------------
# EBM comment generator (rule-based)
# -----------------------------
def ebm_from_df(df):
    notes = set()
    for _, r in df.iterrows():
        score = float(r["Score"])
        pain  = bool(r["Dolore"])
        name  = (str(r["Test"]) + " " + str(r["Regione"])).lower()
        if score < 4:
            if "lunge" in name or "ankle" in name:
                notes.add("Deficit di dorsiflessione: rischio sollevamento del tallone, sovraccarico dell’avampiede e aumento stress rotuleo nello squat.")
            elif "hip" in name:
                notes.add("Ridotta flessione/rotazione d’anca: profondità ridotta e maggiori compensi lombari in squat/stacco.")
            elif "thoracic" in name:
                notes.add("Scarsa estensione toracica: setup della panca limitato e peggior allineamento nello squat.")
            elif "shoulder" in name:
                notes.add("Limitata rotazione esterna di spalla: stabilità scapolo-omerale ridotta in low-bar/panca (più stress anteriori).")
            elif "lumbar" in name or "knee" in name:
                notes.add("Rigidità posteriore/lombare: tolleranza al carico ridotta in deadlift (attenzione a butt-wink).")
        if pain:
            notes.add("Test doloroso: considerare irritabilità tissutale e progressione graduata del carico.")
    if not notes:
        notes.add("Nessun deficit clinicamente rilevante: profilo di mobilità adeguato ai compiti.")
    return sorted(list(notes))

# -----------------------------
# UI
# -----------------------------
st.markdown(f"<h2 style='color:{PRIMARY};margin-bottom:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("### Dati atleta")
    st.session_state["athlete"] = st.text_input("Atleta", st.session_state["athlete"])
    st.session_state["evaluator"] = st.text_input("Valutatore", st.session_state["evaluator"])
    st.session_state["date"] = st.date_input("Data", datetime.strptime(st.session_state["date"], "%Y-%m-%d")).strftime("%Y-%m-%d")
    st.markdown("---")
    st.session_state["section"] = st.selectbox("Sezione", ALL_SECTIONS, index=ALL_SECTIONS.index(st.session_state["section"]))
    colb1,colb2 = st.columns(2)
    with colb1:
        if st.button("Reset valori", use_container_width=True):
            st.session_state["vals"].clear()
            seed_defaults()
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
                    rec["Val"] = max(0.0, ref * random.uniform(0.5, 1.2))
                    rec["Dolore"] = random.random() < 0.15
            st.success("Valori random impostati.")

# Inputs for current section
sec = st.session_state["section"]
st.markdown(f"#### Sezione: <span style='color:{PRIMARY}'>{sec}</span>", unsafe_allow_html=True)

def render_inputs_for_section(section):
    items = []
    if section == "Valutazione Generale":
        for s in ["Squat","Panca","Deadlift","Neurodinamica"]:
            items += TESTS[s]
    else:
        items = TESTS.get(section, [])

    for (name, unit, ref, bilat, region, desc) in items:
        rec = st.session_state["vals"].get(name)
        if rec is None:
            if bilat:
                st.session_state["vals"][name] = {"Dx":0.0,"Sx":0.0,"DoloreDx":False,"DoloreSx":False,"unit":unit,"ref":ref,"bilat":True,"region":region,"desc":desc,"section":section}
            else:
                st.session_state["vals"][name] = {"Val":0.0,"Dolore":False,"unit":unit,"ref":ref,"bilat":False,"region":region,"desc":desc,"section":section}
            rec = st.session_state["vals"][name]

        with st.container():
            st.markdown(f"**{name}** — {desc}  \n*Rif:* {ref} {unit}")
            if bilat:
                c1,c2 = st.columns(2)
                with c1:
                    dx = st.slider(f"Dx ({unit})", 0.0, float(ref*1.5 if unit!='sec' else ref*1.2), float(rec.get('Dx',0.0)), 0.1, key=f"{name}_Dx_{section}")
                    pdx = st.checkbox("Dolore Dx", value=bool(rec.get("DoloreDx", False)), key=f"{name}_pDx_{section}")
                with c2:
                    sx = st.slider(f"Sx ({unit})", 0.0, float(ref*1.5 if unit!='sec' else ref*1.2), float(rec.get('Sx',0.0)), 0.1, key=f"{name}_Sx_{section}")
                    psx = st.checkbox("Dolore Sx", value=bool(rec.get("DoloreSx", False)), key=f"{name}_pSx_{section}")
                if name not in st.session_state["vals"]:
                    st.session_state["vals"][name] = {}
                st.session_state["vals"][name].update({"Dx":dx,"Sx":sx,"DoloreDx":pdx,"DoloreSx":psx,"unit":unit,"ref":ref,"bilat":True,"region":region,"desc":desc,"section":section})
                sc = ability_linear((dx+sx)/2.0, ref); sym = symmetry_score(dx, sx, unit)
                st.caption(f"Score: **{sc:.1f}/10** — Δ {abs(dx-sx):.1f} {unit} — Sym: **{sym:.1f}/10**")
            else:
                val = st.slider(f"Valore ({unit})", 0.0, float(ref*1.5 if unit!='sec' else ref*1.2), float(rec.get('Val',0.0)), 0.1, key=f"{name}_Val_{section}")
                p = st.checkbox("Dolore", value=bool(rec.get("Dolore", False)), key=f"{name}_p_{section}")
                if name not in st.session_state["vals"]:
                    st.session_state["vals"][name] = {}
                st.session_state["vals"][name].update({"Val":val,"Dolore":p,"unit":unit,"ref":ref,"bilat":False,"region":region,"desc":desc,"section":section})
                sc = ability_linear(val, ref)
                st.caption(f"Score: **{sc:.1f}/10**")

render_inputs_for_section(sec)

# Dataframe for current section (or overall)
df_show = build_df(sec)
st.markdown("#### Tabella risultati")
st.dataframe(df_show, use_container_width=True)

# Radar for current section
try:
    df_radar = df_show.copy()
    if len(df_radar) > 0:
        rbuf = radar_plot(df_radar, title=f"{sec} – Punteggi (0–10)")
        st.image(rbuf.getvalue(), use_container_width=True)
    else:
        rbuf = None
except Exception as e:
    rbuf = None
    st.warning(f"■ Radar non disponibile ({e}).")

# Body chart (always from full state)
bbuf = bodychart_image_from_state()
st.image(bbuf.getvalue(), use_container_width=True, caption="Body Chart – Sintesi (verde=buono, giallo=parziale, rosso=deficit; triangolo=Dolore)")

# EBM comments (from current section df)
ebm_notes = ebm_from_df(df_show)

# PDF generate
colp1,colp2 = st.columns(2)
with colp1:
    if st.button("Genera PDF", use_container_width=True):
        athlete = st.session_state["athlete"]
        evaluator = st.session_state["evaluator"]
        date_str = st.session_state["date"]
        try:
            pdf = pdf_report(LOGO, athlete, evaluator, date_str, sec, df_show, bbuf, ebm_notes, radar_buf=rbuf)
            st.success("PDF creato.")
            st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{sec}_{date_str}.pdf", mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(f"Errore PDF: {e}")
with colp2:
    if st.button("Esporta CSV", use_container_width=True):
        st.download_button("Scarica CSV", data=df_show.to_csv(index=False).encode("utf-8"), file_name=f"Fisiomove_{sec}_{st.session_state['date']}.csv", mime="text/csv", use_container_width=True)
