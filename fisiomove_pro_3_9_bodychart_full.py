# Fisiomove Pro 3.9 – BodyChart FULL
# Basato sulla 3.4.6_fix, esteso con:
# - Body chart anatomica nel PDF (overlay su immagine locale)
# - Flag "Dolore durante il test" (Dx/Sx o singolo) e propagazione nel PDF
# - Commenti clinici EBM automatici (in base ai deficit rilevati)
# - Radar, simmetria Dx/Sx, reset valutazione, export JSON
# Requisiti: streamlit, pandas, numpy, matplotlib, pillow, reportlab, plotly

import io, os, json, datetime as dt, random, re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from PIL import Image as PILImage, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

# ---------------- THEME ----------------
PRIMARY = "#1E6CF4"; BG = "#0E1117"; CARD = "#171A21"; TEXT = "#EAEAEA"
OK = "#16a34a"; MID = "#f59e0b"; LOW = "#dc2626"

st.set_page_config(page_title="Fisiomove Pro 3.9 – BodyChart FULL", layout="wide")

CSS = f"""
<style>
html, body, .stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ max-width: 980px; padding-top: .5rem; }}
h1,h2,h3,h4 {{ color: {PRIMARY} !important; }}
.card {{ border:1px solid #2A2F3A; border-radius:12px; padding:12px; background:{CARD}; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:10px; font-weight:700; }}
.dataframe td, .dataframe th {{ color: {TEXT}; }}
.stButton>button {{ background:{PRIMARY}33; border:1px solid {PRIMARY}66; color:{TEXT}; }}
.stButton>button:hover {{ background:{PRIMARY}55; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------- FILES --------------
def load_bytes(cands):
    for p in cands:
        try:
            with open(p, "rb") as f: 
                return f.read()
        except Exception:
            pass
    return None

LOGO = load_bytes(["logo 2600x1000.jpg","EAE1A98B-110F-4A45-A857-C67FFC148375.jpeg","logo_fisiomove.png"])
BODY = load_bytes(["8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png","body_chart.png"])

st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
if LOGO: st.image(LOGO)
else: st.markdown("<div style='font-size:28px;color:#9aa4b2'>Fisiomove</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;margin-top:0'>Fisiomove Pro 3.9 – BodyChart FULL</h2>", unsafe_allow_html=True)

# -------------- TESTS (EBM mapping) --------------
# (name, unit, ref, bilateral, region_key, description)
TESTS = {
    "Squat":[
        ("Weight Bearing Lunge Test","cm",10,True,"ankle","Dorsiflessione in carico."),
        ("Passive Hip Flexion","°",120,True,"hip","Flessione passiva anca."),
        ("Hip Rotation (flexed 90°)","°",40,True,"hip","IR/ER composite in flessione 90°."),
        ("Thoracic Extension (T4- T12)","°",30,False,"thoracic","Estensione toracica segmentale."),
        ("Shoulder ER (adducted, low-bar)","°",70,True,"shoulder","Rotazione esterna per low-bar."),
    ],
    "Panca":[
        ("Shoulder Flexion (supine)","°",180,True,"shoulder","Flessione gleno-omerale in supino."),
        ("External Rotation (90 deg abd)","°",90,True,"shoulder","ER a 90° abd."),
        ("Thoracic Extension (T4- T12)","°",30,False,"thoracic","Arco toracico per setup panca."),
        ("Pectoralis Minor Length","cm",10,True,"shoulder","Distanza spalla-piano."),
        ("Thomas Test (modified)","°",10,True,"hip","Estensione anca (ileopsoas)."),
    ],
    "Deadlift":[
        ("Active Knee Extension (AKE)","°",90,True,"knee","Ischiocrurali (AKE)."),
        ("Straight Leg Raise (SLR)","°",90,True,"hip","Catena posteriore (SLR)."),
        ("Weight Bearing Lunge Test","cm",10,True,"ankle","Dorsiflessione in carico."),
        ("Modified Schober (lumbar)","cm",5,False,"lumbar","Mobilità lombare (flessione)."),
        ("Sorensen Endurance","sec",180,False,"lumbar","Endurance estensori lombari."),
    ]
}
ALL_SECS = ["Squat","Panca","Deadlift"]

# -------------- HELPERS --------------
def score_linear(v, ref):
    try:
        v=float(v); ref=float(ref)
        if ref<=0: return 0.0
        return float(np.clip((v/ref)*10.0,0,10))
    except: 
        return 0.0

def sym_score(dx, sx, unit):
    diff = abs(float(dx)-float(sx))
    if "°" in unit or "deg" in unit.lower(): sc=20.0
    elif unit=="cm": sc=8.0
    else: sc=10.0
    return float(10.0*max(0.0,1.0-min(diff,sc)/sc))

def keyify(*parts):
    s = "_".join(parts)
    return re.sub(r'[^A-Za-z0-9]+','_',s)

import re

# -------------- PATIENT CARD --------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
with c1:
    athlete = st.text_input("Atleta", st.session_state.get("athlete","Mario Rossi"))
    evaluator = st.text_input("Valutatore", st.session_state.get("evaluator","Dott. Alessandro Ferreri – Fisioterapista"))
    age = st.number_input("Età", 10, 90, int(st.session_state.get("age",30)), 1)
with c2:
    height = st.number_input("Altezza (cm)", 100, 220, int(st.session_state.get("height",180)), 1)
    weight = st.number_input("Peso (kg)", 35.0, 200.0, float(st.session_state.get("weight",80.0)), 0.5)
    date_str = st.date_input("Data", value=dt.date.fromisoformat(st.session_state.get("date", dt.date.today().isoformat()))).strftime("%Y-%m-%d")
st.markdown("</div>", unsafe_allow_html=True)

# persist
st.session_state["athlete"]=athlete; st.session_state["evaluator"]=evaluator
st.session_state["age"]=age; st.session_state["height"]=height; st.session_state["weight"]=weight; st.session_state["date"]=date_str

section = st.selectbox("Sezione", ALL_SECS + ["Valutazione Generale"], index=st.session_state.get("sec_idx",0))
st.session_state["sec_idx"]= (ALL_SECS + ["Valutazione Generale"]).index(section)

# -------------- STATE for values --------------
if "vals" not in st.session_state: st.session_state["vals"]={}

def seed_vals(current_section):
    vals={}
    secs = ALL_SECS if current_section=="Valutazione Generale" else [current_section]
    random.seed(7)
    for sec in secs:
        for (name,unit,ref,bilat,region,desc) in TESTS[sec]:
            if bilat:
                vals[name]={"Dx":ref*random.uniform(0.75,1.1),
                            "Sx":ref*random.uniform(0.75,1.1),
                            "DoloreDx":False,"DoloreSx":False,
                            "unit":unit,"ref":ref,"bilat":True,"region":region,"desc":desc}
            else:
                vals[name]={"Val":ref*random.uniform(0.75,1.1),
                            "Dolore":False,"unit":unit,"ref":ref,"bilat":False,"region":region,"desc":desc}
    st.session_state["vals"]=vals

if not st.session_state["vals"]:
    seed_vals(section)

# -------------- INPUTS --------------
st.markdown("### Valutazione")
rows=[]; sym_rows=[]
secs = ALL_SECS if section=="Valutazione Generale" else [section]
for sec in secs:
    st.subheader(sec)
    for (name,unit,ref,bilat,region,desc) in TESTS[sec]:
        st.markdown(f"**{name}** · *{desc}*")
        rec = st.session_state["vals"].get(name,{})
        maxv = (ref*1.5 if unit!="%" else 100.0)
        if bilat:
            c1,c2 = st.columns(2)
            with c1:
                dx = st.slider(f"Dx ({unit}, rif {ref})", 0.0, float(maxv), float(rec.get("Dx",0.0)), 0.1, key=keyify(sec,name,"Dx"))
                pdx = st.checkbox("Dolore durante il test (Dx)", value=bool(rec.get("DoloreDx",False)), key=keyify(sec,name,"DolDx"))
            with c2:
                sx = st.slider(f"Sx ({unit}, rif {ref})", 0.0, float(maxv), float(rec.get("Sx",0.0)), 0.1, key=keyify(sec,name,"Sx"))
                psx = st.checkbox("Dolore durante il test (Sx)", value=bool(rec.get("DoloreSx",False)), key=keyify(sec,name,"DolSx"))
            st.session_state["vals"][name].update({"Dx":dx,"Sx":sx,"DoloreDx":pdx,"DoloreSx":psx,"unit":unit,"ref":ref,"bilat":True,"region":region,"desc":desc})
            val_avg=(dx+sx)/2.0; ability=score_linear(val_avg,ref); sym=sym_score(dx,sx,unit)
            rows.append([sec,name,unit,ref,val_avg,ability,dx,sx,abs(dx-sx),sym,region,(pdx or psx)])
            sym_rows.append({"Test":name,"Dx":dx,"Sx":sx,"Unità":unit,"Delta":abs(dx-sx),"SymScore":sym})
        else:
            val = st.slider(f"Valore ({unit}, rif {ref})", 0.0, float(maxv), float(rec.get("Val",0.0)), 0.1, key=keyify(sec,name,"Val"))
            pain = st.checkbox("Dolore durante il test", value=bool(rec.get("Dolore",False)), key=keyify(sec,name,"Dol"))
            st.session_state["vals"][name].update({"Val":val,"Dolore":pain,"unit":unit,"ref":ref,"bilat":False,"region":region,"desc":desc})
            ability = score_linear(val,ref)
            rows.append([sec,name,unit,ref,val,ability,None,None,None,None,region,pain])
        st.divider()

cols = ["Sezione","Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore","Regione","Dolore"]
df = pd.DataFrame(rows, columns=cols)
sym_df = pd.DataFrame(sym_rows)

# -------------- Radar chart (UI) --------------
def radar_plot(labels, scores, title):
    fig = go.Figure()
    if not labels:
        fig.update_layout(title="Nessun dato", paper_bgcolor=BG, font=dict(color=TEXT)); return fig
    lc = labels+[labels[0]]; sc = scores+[scores[0]]
    fig.add_trace(go.Scatterpolar(r=[10]*len(lc), theta=lc, name="Target", line=dict(dash="dot", color="#9aa4b2")))
    fig.add_trace(go.Scatterpolar(r=sc, theta=lc, name="Atleta", fill="toself", line=dict(color=PRIMARY)))
    fig.update_layout(paper_bgcolor=BG, font=dict(color=TEXT), polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                      title=title, height=420, margin=dict(l=10,r=10,t=40,b=10))
    return fig

labels = (df["Test"]+" ["+df["Sezione"]+"]").tolist() if section=="Valutazione Generale" else df["Test"].tolist()
scores = pd.to_numeric(df["Score"].fillna(0), errors="coerce").fillna(0).tolist()
st.plotly_chart(radar_plot(labels, scores, f"{section} – Abilità (0–10)"), use_container_width=True)

# -------------- Body chart overlay (PIL) --------------
def build_bodychart_image(body_bytes, joint_scores, joint_pain):
    if not body_bytes: return None
    img = PILImage.open(io.BytesIO(body_bytes)).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    def pt(nx, ny): return (int(nx*W), int(ny*H))

    # approx coords for provided image: front (left) and back (right)
    fx = 0.255; bx = 0.745  # centers
    points = {
        # Front
        "shoulder_dx": (fx-0.055, 0.235), "shoulder_sx": (fx+0.055, 0.235),
        "hip_dx": (fx-0.045, 0.58), "hip_sx": (fx+0.045, 0.58),
        "knee_dx": (fx-0.035, 0.78), "knee_sx": (fx+0.035, 0.78),
        "ankle_dx": (fx-0.03, 0.94), "ankle_sx": (fx+0.03, 0.94),
        "thoracic": (fx, 0.33),
        # Back
        "lumbar": (bx, 0.52),
    }

    def draw_marker(xy, score, pain=False):
        if score is None: return
        r = int(10 + 6*(1 - min(max(score,0),10)/10))
        x,y = pt(*xy)
        if score > 7:
            draw.ellipse((x-r,y-r,x+r,y+r), fill=OK, outline="#0f6d2a", width=2)
            draw.line((x-3, y, x-1, y+4), fill="white", width=3)
            draw.line((x-1, y+4, x+5, y-4), fill="white", width=3)
        elif score >= 4:
            draw.ellipse((x-r,y-r,x+r,y+r), fill=MID, outline="#9a6105", width=2)
        else:
            draw.ellipse((x-r,y-r,x+r,y+r), fill=LOW, outline="#8b0d0d", width=2)
        if pain:
            tri = [(x+r+2,y-r-2),(x+r+10,y-r-2),(x+r+6,y-r-10)]
            draw.polygon(tri, fill=LOW)

    for side in ["dx","sx"]:
        draw_marker(points[f"shoulder_{side}"], joint_scores.get(f"shoulder_{side}"), joint_pain.get(f"shoulder_{side}", False))
        draw_marker(points[f"hip_{side}"], joint_scores.get(f"hip_{side}"), joint_pain.get(f"hip_{side}", False))
        draw_marker(points[f"knee_{side}"], joint_scores.get(f"knee_{side}"), joint_pain.get(f"knee_{side}", False))
        draw_marker(points[f"ankle_{side}"], joint_scores.get(f"ankle_{side}"), joint_pain.get(f"ankle_{side}", False))

    draw_marker(points["thoracic"], joint_scores.get("thoracic"), joint_pain.get("thoracic", False))
    draw_marker(points["lumbar"], joint_scores.get("lumbar"), joint_pain.get("lumbar", False))

    out = io.BytesIO()
    img.save(out, format="PNG"); out.seek(0)
    return out

def aggregate_joint_scores(df: pd.DataFrame):
    joints = ["shoulder_dx","shoulder_sx","hip_dx","hip_sx","knee_dx","knee_sx","ankle_dx","ankle_sx","thoracic","lumbar"]
    acc = {k: [] for k in joints}; pain = {k: [] for k in joints}
    for _, r in df.iterrows():
        region = r["Regione"]; score = float(r["Score"] or 0.0); pain_flag = bool(r["Dolore"])
        if region == "shoulder":
            acc["shoulder_dx"].append(score); acc["shoulder_sx"].append(score)
            pain["shoulder_dx"].append(pain_flag); pain["shoulder_sx"].append(pain_flag)
        elif region == "hip":
            acc["hip_dx"].append(score); acc["hip_sx"].append(score)
            pain["hip_dx"].append(pain_flag); pain["hip_sx"].append(pain_flag)
        elif region == "knee":
            acc["knee_dx"].append(score); acc["knee_sx"].append(score)
            pain["knee_dx"].append(pain_flag); pain["knee_sx"].append(pain_flag)
        elif region == "ankle":
            acc["ankle_dx"].append(score); acc["ankle_sx"].append(score)
            pain["ankle_dx"].append(pain_flag); pain["ankle_sx"].append(pain_flag)
        elif region == "thoracic":
            acc["thoracic"].append(score); pain["thoracic"].append(pain_flag)
        elif region == "lumbar":
            acc["lumbar"].append(score); pain["lumbar"].append(pain_flag)
    def avg_or_none(arr): 
        arr = [a for a in arr if a is not None]
        return float(np.mean(arr)) if arr else None
    joint_scores = {k: avg_or_none(v) for k,v in acc.items()}
    joint_pain = {k: any(v) if v else False for k,v in pain.items()}
    return joint_scores, joint_pain

joint_scores, joint_pain = aggregate_joint_scores(df)
body_buf = build_bodychart_image(BODY, joint_scores, joint_pain) if BODY else None

if body_buf:
    st.image(body_buf, caption="Body Chart – Sintesi visiva (verde=ok, giallo=parziale, rosso=deficit, triangolo=Dolore)")
else:
    st.warning("Body chart non trovata. Aggiungi '8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png' nella cartella dell'app.")

# -------------- EBM COMMENTS --------------
def ebm_comment(df):
    out=[]
    for _,r in df.iterrows():
        t=r["Test"].lower(); sc=float(r["Score"] or 0.0); pain=bool(r["Dolore"])
        if sc<4:
            if "lunge" in t or "caviglia" in t or "ankle" in t:
                out.append("Deficit di dorsiflessione può indurre sollevamento del tallone, sovraccarico dell’avampiede e stress rotuleo nello squat.")
            elif "hip" in t or "anca" in t:
                out.append("Ridotta flessione/rotazione d’anca altera la profondità e aumenta i compensi lombari in squat e stacco.")
            elif "thoracic" in t or "toracic" in t:
                out.append("Scarsa estensione toracica limita il setup in panca e l’allineamento del carico nello squat.")
            elif "shoulder" in t or "spalla" in t:
                out.append("Limitata ER riduce la stabilità scapolo-omerale (low-bar/panca) e può aumentare stress anteriori.")
            elif "knee" in t or "ginocchio" in t:
                out.append("Rigidità posteriore può anticipare flessione lombare nello stacco (butt wink).")
            elif "lumbar" in t or "lomb" in t:
                out.append("Bassa mobilità/endurance lombare può ridurre tolleranza al carico ripetuto in deadlift.")
        if pain:
            out.append("Il test è risultato doloroso: considerare irritabilità tissutale e progressione graduata del carico.")
    if not out:
        out.append("Nessun deficit clinicamente rilevante: profilo di mobilità adeguato ai compiti.")
    seen=set(); uniq=[]
    for c in out:
        if c not in seen: uniq.append(c); seen.add(c)
    return uniq

# -------------- PDF --------------
def header_canvas(logo_bytes):
    img = ImageReader(io.BytesIO(logo_bytes)) if logo_bytes else None
    def _draw(canvas, doc):
        if img:
            page_w, page_h = A4
            w = page_w - 2*cm; h = w * 0.22; x = (page_w - w) / 2.0; y = page_h - h - 10
            canvas.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
        canvas.setFont("Helvetica", 9); canvas.setFillColor(colors.grey)
        canvas.drawRightString(A4[0]-36, 18, f"Fisiomove Pro – {dt.date.today().isoformat()}")
    return _draw

def radar_png(labels, scores):
    if len(labels)==0:
        buf=io.BytesIO(); return buf
    N=len(labels); angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()
    angles_c=angles+[angles[0]]; scores_c=list(scores)+[scores[0]]; ideal=[10]*N+[10]
    fig,ax=plt.subplots(figsize=(5.2,5.2),subplot_kw=dict(polar=True))
    ax.grid(True,linewidth=0.5,color="#cccccc"); ax.spines["polar"].set_color("#999")
    ax.plot(angles_c,ideal,color="#777",linestyle="dotted",linewidth=1)
    ax.plot(angles_c,scores_c,color=PRIMARY,linewidth=2); ax.fill(angles_c,scores_c,color="#1E6CF433")
    ax.set_ylim(0,10); ax.set_xticks(angles); ax.set_xticklabels(labels,fontsize=8)
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=200,facecolor="white",bbox_inches="tight"); plt.close(fig); buf.seek(0); return buf

def pdf_report(logo_bytes, athlete, evaluator, date_str, section, df, body_buf, ebm_notes):
    disp = df.copy()
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(buf_pdf, pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=3.6*cm, bottomMargin=1.6*cm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))

    story = []
    story.append(Paragraph("<b>Report Valutazione</b>", title)); story.append(Spacer(1,6))
    meta = [["Atleta", athlete, "Età", str(st.session_state.get("age","")), "Data", date_str],
            ["Valutatore", evaluator, "Altezza (cm)", str(st.session_state.get("height","")), "Peso (kg)", str(st.session_state.get("weight",""))]]
    mt = Table(meta, colWidths=[2.6*cm,4.8*cm,2.6*cm,2.0*cm,2.6*cm,3.0*cm])
    mt.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.25,colors.grey), ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)]))
    story.append(mt); story.append(Spacer(1,8))
    story.append(Paragraph(f"Sezione: <b>{section}</b>", styles["Heading3"]))

    # Tabella risultati (include Dolore)
    cols = ["Sezione","Test","Unità","Riferimento","Valore","Score","Dx","Sx","Delta","SymScore","Dolore"]
    disp_tbl = disp[cols].copy()
    for c in ["Valore","Score","Dx","Sx","Delta","SymScore"]:
        if c in disp_tbl.columns: disp_tbl[c] = pd.to_numeric(disp_tbl[c], errors="coerce").round(1)
    tdata = [list(disp_tbl.columns)] + disp_tbl.fillna("").astype(str).values.tolist()
    tbl = Table(tdata, colWidths=[2.1*cm,5.5*cm,1.1*cm,1.4*cm,1.4*cm,1.1*cm,1.1*cm,1.1*cm,1.1*cm,1.6*cm,1.2*cm])
    tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),
                             ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                             ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                             ("ALIGN",(2,1),(-1,-1),"CENTER")]))
    story.append(tbl); story.append(Spacer(1,8))

    # Radar
    labels = (disp["Test"]+" ["+disp["Sezione"]+"]").tolist() if section=="Valutazione Generale" else disp["Test"].tolist()
    scores = pd.to_numeric(disp["Score"].fillna(0), errors="coerce").fillna(0).tolist()
    rbuf = radar_png(labels, scores)
    story.append(RLImage(io.BytesIO(rbuf.getvalue()), width=12.5*cm, height=12.5*cm)); story.append(Spacer(1,8))

    # Body chart section
    story.append(Paragraph("Body Chart – Sintesi visiva", styles["Heading3"]))
    if body_buf:
        story.append(RLImage(ImageReader(io.BytesIO(body_buf.getvalue())), width=16*cm, height=12*cm))
        legend = "Legenda: rosso = deficit marcato; giallo = parziale; verde = adeguato; triangolo = test doloroso."
        story.append(Paragraph(legend, styles["Normal"]))
    else:
        story.append(Paragraph("Body chart non disponibile.", styles["Normal"]))
    story.append(Spacer(1,8))

    # EBM comments
    story.append(Paragraph("Commento clinico (EBM)", styles["Heading3"]))
    for c in ebm_notes:
        story.append(Paragraph(f"– {c}", styles["Normal"]))

    header = (lambda c,d: None)
    if logo_bytes:
        header = header_canvas(logo_bytes)
        doc.build(story, onFirstPage=header, onLaterPages=header)
    else:
        doc.build(story)
    buf_pdf.seek(0); return buf_pdf

# -------------- ACTIONS --------------
st.markdown("### Esporta / Utility")
c1,c2,c3 = st.columns(3)
with c1:
    if st.button("Genera PDF"):
        ebm_notes = ebm_comment(df)
        pdf = pdf_report(LOGO, athlete, evaluator, date_str, section, df, body_buf, ebm_notes)
        st.download_button("Scarica PDF", data=pdf.getvalue(),
                           file_name=f"Fisiomove_{section}_{athlete.replace(' ','')}_{date_str}.pdf",
                           mime="application/pdf")
with c2:
    if st.button("Esporta JSON"):
        payload = {"athlete":athlete,"evaluator":evaluator,"date":date_str,"records":df.to_dict(orient="records")}
        st.download_button("Scarica JSON", data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                           file_name=f"{athlete.replace(' ','')}_{section}_{date_str}.json", mime="application/json")
with c3:
    if st.button("Azzera valutazione"):
        st.session_state["vals"] = {}
        st.experimental_rerun()
