# Fisiomove Pro 3.0 – Streamlit app (user-friendly, robust, pro features)
# - SQLite session storage (save/load) + JSON export
# - History & radar comparison per athlete
# - Summary page with traffic-light priorities
# - Sections: Simmetrie Sup/Inf, Neurodinamica (0–10 scaling)
# - Tutorials (text) for every test
# - Robust PDF creation (only measured tests) and safe radar rendering
# - Uses local logo file 'logo_fisiomove.png' if present (optional)

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
import matplotlib.pyplot as plt

PRIMARY = "#004A9F"
DEFAULT_LOGO_PATH = "logo_fisiomove.png"  # carica un PNG/JPG nel repo con questo nome (opzionale)
DB_PATH = "fisiomove.db"

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Fisiomove Pro 3.0 – Mobilità EBM", layout="wide")

# ---------- HELPERS ----------
def score_linear(value: float, ref: float) -> float:
    try:
        v = float(value); r = float(ref)
        if r <= 0: return 0.0
        return max(0.0, min(10.0, (v/r)*10.0))
    except Exception:
        return 0.0

def score_color(s: float) -> str:
    if s < 4: return "#D32F2F"     # red
    if s <= 7: return "#F9A825"    # yellow
    return "#2E7D32"               # green

def badge(text: str, color: str) -> str:
    return f"<span style='background:{color}20;color:{color};padding:4px 8px;border-radius:8px;font-weight:700;'>{text}</span>"

def radar_plot(labels, scores, title):
    if len(labels) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Nessun test compilato",
            polar=dict(radialaxis=dict(visible=True, range=[0,10])),
            margin=dict(l=10, r=10, t=40, b=10),
            height=320
        )
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
        polar=dict(radialaxis=dict(visible=True, range=[0,10], tickmode="linear", dtick=1)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=460
    )
    return fig

def radar_png_buffer(labels, scores):
    import matplotlib
    matplotlib.use("Agg")  # ✅ forza backend compatibile con Streamlit Cloud

    N = len(labels)
    if N == 0:
        return io.BytesIO()

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles_c = angles + [angles[0]]
    scores_c = list(scores) + [scores[0]]
    ideal_c = [10]*N + [10]

    fig, ax = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
    ax.plot(angles_c, ideal_c, color="#777", linewidth=1, linestyle="dotted")
    ax.plot(angles_c, scores_c, color="#004A9F", linewidth=2)
    ax.fill(angles_c, scores_c, color="#2f6fb640")
    ax.set_ylim(0,10)
    ax.set_yticks([2,4,6,8,10])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True)

    buf = io.BytesIO()
    try:
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=200)
    except Exception as e:
        print("Errore nel salvataggio radar:", e)
    finally:
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

def pdf_single_section(logo_bytes, athlete, evaluator, date_str, section_name, df):
    disp = df[df["Valore"] > 0].copy()
    buf_pdf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf_pdf,
        pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], textColor=colors.HexColor(PRIMARY))
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontSize=10, leading=14)

    story = []
    story.append(Paragraph("Fisiomove Pro 3.0 – Mobilità EBM", title))
    story.append(Spacer(1, 6))
    head = Table([["Atleta", athlete, "Data", date_str, "Valutatore", evaluator]],
                 colWidths=[2*cm, 6*cm, 2*cm, 3*cm, 2.5*cm, 4*cm])
    head.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke)
    ]))
    story.append(head)
    story.append(Spacer(1, 8))

    # Logo
    if logo_bytes:
        try:
            story.append(Image(ImageReader(io.BytesIO(logo_bytes)), width=6*cm, height=6*cm))
            story.append(Spacer(1, 6))
        except Exception:
            pass
    else:
        try:
            with open(DEFAULT_LOGO_PATH, "rb") as f:
                story.append(Image(ImageReader(f), width=6*cm, height=6*cm))
                story.append(Spacer(1, 6))
        except Exception:
            pass

    story.append(Paragraph(f"Sezione testata: <b>{section_name}</b>", styles['Heading3']))

    if disp.empty:
        story.append(Paragraph("Nessun test compilato (valori > 0).", normal))
    else:
        disp["Score"] = disp["Score"].round(1)
        table_data = [["Test","Unità","Rif.","Valore","Punteggio (0–10)"]] + disp.values.tolist()
        tbl = Table(table_data, colWidths=[7*cm,1.5*cm,2.0*cm,2.2*cm,3.1*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(PRIMARY)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("ALIGN",(2,1),(-1,-1),"CENTER")
        ]))
        story.append(tbl)
        story.append(Spacer(1, 8))

        # Grafico radar con controllo robusto
        rbuf = radar_png_buffer(disp["Test"].tolist(), disp["Score"].tolist())
        try:
            img = rbuf.getvalue()
            if img and len(img) > 100:
                story.append(Image(ImageReader(io.BytesIO(img)), width=10*cm, height=10*cm))
                story.append(Spacer(1, 6))
            else:
                raise ValueError("Radar vuoto")
        except Exception:
            story.append(Paragraph("⚠️ Grafico radar non disponibile (nessun test valido o errore grafico).", normal))
            story.append(Spacer(1, 6))

        # Commento EBM
        story.append(Paragraph(ebm_comment(section_name, disp).replace("\n","<br/>"), normal))

    # ✅ Chiusura corretta della funzione
    doc.build(story)
    buf_pdf.seek(0)
    return buf_pdf


# ---------- DATASETS ----------
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
    # Simmetrie: calcolo 0–10 su differenze assolute (0% = 10/10; 20% o 10 unità = 0/10)
    "Simmetrie Superiori": [
        ("Grip Strength Δ% (Dx-Sx)","%",0,"Dinamometro: differenza % (assoluta). Punteggio=10 se 0%; 0 se ≥20%."),
        ("ER Spalla Δ° (90° abd)","°",0,"Differenza assoluta di gradi ER tra lati."),
        ("Elevazione Scapolare Δcm","cm",0,"Differenza distanza acromion-parete o misura analoga."),
    ],
    "Simmetrie Inferiori": [
        ("AKE Δ°","°",0,"Differenza assoluta AKE tra arti."),
        ("SLR Δ°","°",0,"Differenza assoluta SLR tra arti."),
        ("WBLT Δcm","cm",0,"Differenza assoluta WBLT tra arti."),
    ],
    # Neurodinamica: range sintomatico con target clinici
    "Neurodinamica": [
        ("SLR (sensibile)","°",70,"Range con sintomatologia controllata."),
        ("PNLT tibiale (dorsiflessione/eversione)","°",40,"Differenziazione sensibile tibiale."),
        ("PNLT peroneale (plantarflessione/inversione)","°",40,"Differenziazione sensibile peroneale."),
        ("ULNT1 (n. mediano)","°",100,"Range con differenziazione cingolo."),
    ],
}

# Tutorials standardizzati (test -> breve standardizzazione pratica)
TUTORIAL = {
    "Weight Bearing Lunge Test": "Atleta in affondo verso parete, tallone a terra. Avanza finché il ginocchio tocca la parete senza sollevare il tallone. Misura distanza punta-parete (cm).",
    "Passive Hip Flexion": "Supino, bacino stabilizzato. Flessione passiva d'anca evitando compensi lombari. Misura l’angolo libero da compensi.",
    "Hip Rotation (flexed 90°)": "Supino, anca e ginocchio a 90°. Ruota internamente/esternamente mantenendo bacino stabile. Misura range per lato.",
    "Thoracic Extension (T4–T12)": "Seduto/stazione, inclinometro tra T4–T12. Estensione attiva massima, bacino neutro.",
    "Shoulder ER (adducted, low‑bar)": "In piedi, braccio addotto. Ruota esternamente come per low‑bar tenendo scapole addotte. Misura angolo.",
    "Shoulder Flexion (supine)": "Supino, braccio in elevazione. Stabilizza coste. Misura il massimo angolo senza iperlordosi.",
    "External Rotation (90° abd)": "Supino, 90° abduzione, gomito 90°. Ruota ER mantenendo scapola controllata.",
    "Pectoralis Minor Length": "Supino rilassato. Misura distanza acromion-posteriore dal piano tavolo/parete: maggiore distanza = tightness.",
    "Wall Angel (distance)": "Schiena e avambracci a parete, braccia a 'W' in salita. Misura la distanza dell’avambraccio/polso dalla parete (cm).",
    "Thomas Test (modified)": "Bordo lettino. Abbraccia una coscia, lascia l’altra pendere. Valuta estensione d’anca (°) e compensi.",
    "Active Knee Extension (AKE)": "Supino, anca 90°, estendi il ginocchio. Misura deficit di estensione (°).",
    "Straight Leg Raise (SLR)": "Supino. Eleva l’arto in estensione fino a tensione controllata. Misura l’angolo.",
    "Modified Schober (lumbar)": "Segna 10 cm sopra e 5 cm sotto L5; flessione massima. Misura incremento (cm).",
    "Sorensen Endurance": "Prono su bordo lettino, bacino fissato. Mantieni tronco orizzontale (sec).",
    "Grip Strength Δ% (Dx-Sx)": "Dinamometro. Tre prove per lato, prendi il migliore. Δ% = |Dx−Sx| / max(Dx,Sx) × 100.",
    "ER Spalla Δ° (90° abd)": "Differenza assoluta di ER tra arti a 90° abduzione.",
    "Elevazione Scapolare Δcm": "Differenza di altezza/distanza tra scapole in elevazione.",
    "AKE Δ°": "Differenza assoluta di AKE tra arti.",
    "SLR Δ°": "Differenza assoluta di SLR tra arti.",
    "WBLT Δcm": "Differenza assoluta di WBLT tra arti.",
    "SLR (sensibile)": "SLR con monitoraggio dei sintomi; registra l’angolo tollerabile.",
    "PNLT tibiale (dorsiflessione/eversione)": "Bias tibiale con differenziazione prossimale; annota range sintomatico.",
    "PNLT peroneale (plantarflessione/inversione)": "Bias peroneale; annota range sintomatico.",
    "ULNT1 (n. mediano)": "Sequenza ULNT1; registra l’angolo/range con differenziazione scapolare.",
}

# ---------- STORAGE (SQLite + JSON export) ----------
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
    payload = {
        "athlete": athlete,
        "section": section,
        "date": date_str,
        "records": df.to_dict(orient="records")
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

init_db()

# ---------- UI HEADER ----------
col_logo, col_title = st.columns([1,2])
with col_logo:
    try: st.image(DEFAULT_LOGO_PATH, use_container_width=True)
    except Exception: st.markdown("### Fisiomove")
with col_title:
    st.markdown("<h1 style='margin-bottom:0'>Fisiomove Pro 3.0</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{PRIMARY}; font-size:18px;'>Mobilità EBM · Storico · Riepilogo · Simmetrie · Neuro</div>", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.header("Dati atleta")
    athlete = st.text_input("Nome", "Mario Rossi")
    evaluator = st.text_input("Valutatore", "Dott. Alessandro Ferreri – Fisioterapista")
    date_str = st.date_input("Data", value=dt.date.today()).strftime("%Y-%m-%d")
    st.markdown("---")
    st.header("Sezione")
    section = st.selectbox("Scegli una sezione", list(TESTS.keys()))
    st.caption("Il PDF include solo la sezione selezionata e i test compilati (>0).")

tabs = st.tabs(["Valutazione", "Riepilogo", "Storico"])

# ---------- TAB: VALUTAZIONE ----------
with tabs[0]:
    st.subheader(f"Sezione: {section}")
    st.write("Inserisci i valori (°/cm/sec/%). Punteggio 0–10 lineare rispetto al riferimento EBM. Tocca ℹ️ per lo standard del test.")

    rows = []
    cols = st.columns(2)
    for i, (name, unit, ref, desc) in enumerate(TESTS[section]):
        with cols[i % 2]:
            col_a, col_b = st.columns([5,1])
            with col_a:
                st.markdown(f"**{name}** · *{desc}*")
            with col_b:
                with st.expander("ℹ️", expanded=False):
                    st.caption(TUTORIAL.get(name, "—"))
            maxv = float(ref)*1.2 if (unit not in ["%", "sec"]) else (float(ref)*1.5 if unit=="sec" else 30.0)
            # For symmetry deltas (ref=0) choose a practical max range
            if section.startswith("Simmetrie") and ref == 0:
                maxv = 30.0 if unit != "%" else 30.0
            val = st.slider(f"{unit} (rif. {ref})", min_value=0.0, max_value=float(maxv), value=0.0, step=0.1, key=f"{section}_{name}")
            # scoring
            if section.startswith("Simmetrie"):
                if unit == "%":
                    score = max(0.0, 10.0 * (1.0 - min(val, 20.0)/20.0))  # 0%->10, 20%->0
                else:
                    score = max(0.0, 10.0 * (1.0 - min(val, 10.0)/10.0))  # 0->10, 10 unità->0
            elif section == "Neurodinamica":
                score = score_linear(val, max(ref, 1))
            else:
                score = score_linear(val, ref)

            color = score_color(score)
            st.markdown(badge(f"Punteggio: {score:.1f}/10", color), unsafe_allow_html=True)
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

    st.markdown("---")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        if st.button("💾 Salva sessione (SQLite)"):
            save_session(athlete, section, date_str, df)
            st.success("Sessione salvata.")
    with c2:
        if st.button("⬇️ Esporta JSON"):
            js = export_session_json(athlete, section, date_str, df)
            st.download_button("Scarica JSON", data=js, file_name=f"{athlete.replace(' ','')}_{section}_{date_str}.json", mime="application/json")
    with c3:
        st.subheader("PDF sezione")
        disabled_pdf = df_meas.empty
        if st.button("Genera PDF", disabled=disabled_pdf):
            try:
                with open(DEFAULT_LOGO_PATH, "rb") as f:
                    logo_bytes = f.read()
            except Exception:
                logo_bytes = None
            pdf = pdf_single_section(logo_bytes, athlete, evaluator, date_str, section, df)
            st.download_button("Scarica PDF", data=pdf.getvalue(), file_name=f"Fisiomove_{section}_{athlete.replace(' ','')}_{date_str}.pdf", mime="application/pdf")

# ---------- TAB: RIEPILOGO ----------
with tabs[1]:
    st.subheader("Riepilogo rapido (Semaforo clinico)")
    st.caption("Legenda: 🔴 <4 deficit · 🟡 4–7 migliorabile · 🟢 >7 buono")
    if 'df' not in locals() or df_meas.empty:
        st.warning("Compila almeno un test nella scheda Valutazione per vedere il riepilogo.")
    else:
        df_sorted = df_meas.sort_values("Score")
        st.dataframe(df_sorted[["Test","Unità","Valore","Score"]].style.format({"Score":"{:.1f}"}), use_container_width=True)
        worst = df_sorted.head(3)
        if not worst.empty:
            st.markdown("### Priorità (Top 3)")
            cols = st.columns(len(worst))
            for k, (_, row) in enumerate(worst.iterrows()):
                with cols[k]:
                    st.markdown(badge(f"{row['Test']}", score_color(row["Score"])), unsafe_allow_html=True)
                    st.caption(f"Score {row['Score']:.1f} – {TUTORIAL.get(row['Test'],'')}")

# ---------- TAB: STORICO ----------
with tabs[2]:
    st.subheader("Storico e confronto radar")
    data_hist = load_sessions(athlete, section)
    if data_hist.empty:
        st.info("Nessuna sessione salvata per questo atleta/sezione.")
    else:
        dates = data_hist["date"].tolist()
        pick = st.multiselect("Seleziona fino a 3 date da confrontare", dates, default=dates[:2])
        comp = []
        for d in pick[:3]:
            rec = data_hist[data_hist["date"]==d].iloc[0]
            df_rec = pd.DataFrame(json.loads(rec["data_json"]))
            df_rec = df_rec[df_rec["Valore"] > 0]
            comp.append((d, df_rec))

        if not comp:
            st.warning("Seleziona almeno una data.")
        else:
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
