# fisiomove_app.py — Fisiomove Pro 3.10 Refactor Completo

# 1. Import
import io, os, random
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

# 2. Config
st.set_page_config(page_title="Fisiomove Pro v. 3.10", layout="centered")

# 3. Costanti
APP_TITLE = "Fisiomove Pro 3.10 — Body mobility screening"
PRIMARY = "#1E6CF4"
LOGO_PATHS = ["logo.png", "logo 2600x1000.jpg"]
BODYCHART_PATHS = ["body_chart.png", "8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png"]
ALL_SECTIONS = ["Squat", "Panca", "Deadlift", "Neurodinamica", "Valutazione Generale"]

# 4. Caricamento asset
def load_logo_bytes():
    for path in LOGO_PATHS:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    img = Image.new("RGB", (1000, 260), (30, 108, 244))
    d = ImageDraw.Draw(img)
    d.text((30, 100), "Fisiomove", fill=(255, 255, 255))
    bio = io.BytesIO(); img.save(bio, format="PNG")
    return bio.getvalue()

def load_bodychart_image():
    for path in BODYCHART_PATHS:
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGBA")
            except:
                pass
    img = Image.new("RGBA", (1200, 800), (245,245,245,255))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "Aggiungi body_chart.png", fill=(0, 0, 0))
    return img

LOGO = load_logo_bytes()
BODYCHART_BASE = load_bodychart_image()

# 5. Stato Streamlit
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
# 6. Funzioni di scoring
def ability_linear(val, ref):
    try:
        if ref <= 0: return 0.0
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
        return 10.0 * max(0.0, 1.0 - min(diff, scale)/scale)
    except:
        return 0.0

# 7. Dizionario TESTS
TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10.0, True,  "ankle",    "Dorsiflessione in carico (WB lunge)."),
        ("Passive Hip Flexion",       "°",  120.0, True, "hip",      "Flessione d’anca passiva supina."),
        ("Hip Rotation (flexed 90°)", "°",   40.0, True, "hip",      "Rotazione anca a 90° flessione."),
        ("Thoracic Extension (T4-T12)", "°", 30.0, False,"thoracic", "Estensione toracica globale."),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "ER spalla per low-bar."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Flessione spalla, scapole stabili."),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "ER a 90° abd (capsula anteriore)."),
        ("Thoracic Extension (T4-T12)", "°", 30.0, False, "thoracic", "Estensione toracica per setup."),
        ("Pectoralis Minor Length", "cm", 10.0, True, "shoulder", "Lunghezza piccolo pettorale."),
        ("Thomas Test (modified)", "°", 10.0, True, "hip", "Flesso-estensibilità flessori anca."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90.0, True, "knee", "Estensione attiva ginocchio."),
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "SLR catena posteriore."),
        ("Weight Bearing Lunge Test", "cm", 10.0, True, "ankle", "Dorsiflessione in carico."),
        ("Modified Schober (lumbar)", "cm", 5.0, False, "lumbar", "Mobilità lombare in flessione."),
        ("Sorensen Endurance", "sec", 180.0, False, "lumbar", "Endurance estensori lombari."),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Catena neurodinamica posteriore LE."),
        ("Popliteal Knee Bend (PKB)", "°", 90.0, True, "knee", "Scorrimento distale nervo sciatico."),
        ("ULNT1A (Median nerve)", "°", 90.0, True, "shoulder", "Upper Limb Neurodynamic Test 1A."),
    ],
}

# 8. Seed valori di default
def seed_defaults():
    if st.session_state["vals"]:
        return
    for sec, items in TESTS.items():
        for (name, unit, ref, bilat, region, desc) in items:
            if name not in st.session_state["vals"]:
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
                        "section": sec
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
                        "section": sec
                    }

seed_defaults()

# 9. Costruzione DataFrame
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
                sc = ability_linear((dx + sx) / 2.0, ref)
                sym = symmetry_score(dx, sx, unit)
                rows.append([
                    sec, name, unit, ref, f"{(dx+sx)/2:.1f}", sc,
                    dx, sx, abs(dx - sx), sym,
                    bool(rec.get("DoloreDx", False) or rec.get("DoloreSx", False)),
                    region
                ])
            else:
                val = float(rec.get("Val", 0))
                sc = ability_linear(val, ref)
                rows.append([
                    sec, name, unit, ref, f"{val:.1f}", sc,
                    "", "", "", "",
                    bool(rec.get("Dolore", False)),
                    region
                ])
    return pd.DataFrame(rows, columns=[
        "Sezione", "Test", "Unità", "Rif", "Valore", "Score",
        "Dx", "Sx", "Delta", "SymScore", "Dolore", "Regione"
    ])
# 10. Bodychart rendering
def bodychart_image_from_state(width=1200, height=800):
    base = BODYCHART_BASE.copy().resize((width, height))
    draw = ImageDraw.Draw(base)

    fx, bx = 0.255, 0.745
    points = {
        "shoulder_dx": (fx-0.135, 0.225),
        "shoulder_sx": (fx+0.058, 0.225),
        "hip_dx":      (fx-0.112, 0.50),
        "hip_sx":      (fx+0.038, 0.50),
        "knee_dx":     (fx-0.106, 0.68),
        "knee_sx":     (fx+0.030, 0.68),
        "ankle_dx":    (fx-0.070, 0.94),
        "ankle_sx":    (fx+0.025, 0.94),
        "thoracic":    (bx, 0.30),
        "lumbar":      (bx, 0.40),
    }

    df_all = build_df("Valutazione Generale")
    region_scores = {}
    region_pain = {}
    for region in ["shoulder", "hip", "knee", "ankle", "thoracic", "lumbar"]:
        sub = df_all[df_all["Regione"] == region]
        # Nuovo calcolo: penalizza asimmetria nella media regionale
        mean_score = sub["Score"].astype(float).mean()
        penalty = pd.to_numeric(sub["SymScore"], errors="coerce").dropna().apply(lambda x: 1 - x / 10).mean()
        final_score = np.clip(mean_score * (1 - penalty), 0, 10)
        region_scores[region] = float(final_score)
        region_pain[region] = bool(sub["Dolore"].any()) if len(sub) else False

    def score_color(score):
        if score > 7: return (22, 163, 74, 255)     # verde
        if score >= 4: return (245, 158, 11, 255)   # giallo
        return (220, 38, 38, 255)                   # rosso

    def draw_marker(xn, yn, score, pain):
        try:
            # Fallback: se score non è un numero valido, forzalo a 0
            score = float(score)
            if np.isnan(score):
                score = 0.0
        except:
            score = 0.0

        x = int(xn * width)
        y = int(yn * height)
        radius = int(10 + 6 * (1 - min(max(score, 0), 10) / 10))
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=score_color(score))
    
        if score > 7:
            draw.line((x-4, y, x-2, y+6), fill=(255,255,255,255), width=3)
            draw.line((x-2, y+6, x+6, y-4), fill=(255,255,255,255), width=3)
        if pain:
            tri = [(x+radius+2, y-radius-2), (x+radius+12, y-radius-2), (x+radius+7, y-radius-12)]
            draw.polygon(tri, fill=(220,38,38,255))

    for region, coord in points.items():
        base_region = region.split("_")[0] if "_" in region else region
        draw_marker(*coord, region_scores.get(base_region, 0), region_pain.get(base_region, False))

    bio = io.BytesIO(); base.save(bio, format="PNG"); bio.seek(0)
    return bio

# 11. Radar plot (score)
def radar_plot(df, title="Radar – Punteggi (0–10)"):
    labels = df["Test"].tolist()
    values = df["Score"].astype(float).tolist()
    if len(labels) < 3:
        raise ValueError("Servono almeno 3 test per il radar.")

    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]
    angles = np.linspace(0, 2 * np.pi, len(values_c), endpoint=False)

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
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

# 12. Asymmetry bar plot
def asymmetry_bar_plot(df, title="Asimmetria Dx–Sx"):
    import matplotlib.pyplot as plt
    import io

    # Filtra solo test con Delta numerico
    df_bilat = df[df["Delta"].notnull()].copy()

    try:
        df_bilat["Delta"] = pd.to_numeric(df_bilat["Delta"], errors="coerce")
        df_bilat["SymScore"] = pd.to_numeric(df_bilat["SymScore"], errors="coerce")
    except Exception as e:
        st.warning(f"Errore nel cast dei valori numerici: {e}")
        return None

    # Rimuovi righe dove i dati non sono numerici
    df_bilat = df_bilat.dropna(subset=["Delta", "SymScore"])


    if df_bilat.empty:
        return None

    labels = df_bilat["Test"].tolist()
    
    # Trasforma SymScore in asimmetria su scala 0-10
    df_bilat["AsymValue"] = (1 - df_bilat["SymScore"] / 10) * 10
    asym_values = df_bilat["AsymValue"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, asym_values, color="#FF6B6B")

    ax.set_xlabel("Indice di Asimmetria (0–10)")
    ax.set_xlim(0, 10)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va='center')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# 13. Intestazione app
st.markdown(f"<h2 style='color:{PRIMARY};margin-bottom:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)

# 14. Sidebar – dati atleta e opzioni
with st.sidebar:
    st.markdown("### Dati atleta")
    st.session_state["athlete"] = st.text_input("Atleta", st.session_state["athlete"])
    st.session_state["evaluator"] = st.text_input("Valutatore", st.session_state["evaluator"])
    st.session_state["date"] = st.date_input("Data", datetime.strptime(st.session_state["date"], "%Y-%m-%d")).strftime("%Y-%m-%d")

    st.markdown("---")
    st.session_state["section"] = st.selectbox("Sezione", ALL_SECTIONS, index=ALL_SECTIONS.index(st.session_state["section"]))

    colb1, colb2 = st.columns(2)
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

# 15. Input dinamico test
def render_inputs_for_section(section):
    items = []
    if section == "Valutazione Generale":
        for s in ["Squat", "Panca", "Deadlift", "Neurodinamica"]:
            items += TESTS[s]
    else:
        items = TESTS.get(section, [])

    for (name, unit, ref, bilat, region, desc) in items:
        rec = st.session_state["vals"].get(name)
        if not rec:
            continue

        with st.container():
            st.markdown(f"**{name}** — {desc}  \n*Rif:* {ref} {unit}")
            if bilat:
                c1, c2 = st.columns(2)
                with c1:
                    dx = st.slider(
                        f"Dx ({unit})",
                        0.0, ref * 1.5,
                        float(rec.get("Dx", 0.0)),
                        0.1,
                        key=f"{name}_Dx_{section}"
                    )
                    pdx = st.checkbox(
                        "Dolore Dx",
                        value=bool(rec.get("DoloreDx", False)),
                        key=f"{name}_pDx_{section}"
                    )
                with c2:
                    sx = st.slider(
                        f"Sx ({unit})",
                        0.0, ref * 1.5,
                        float(rec.get("Sx", 0.0)),
                        0.1,
                        key=f"{name}_Sx_{section}"
                    )
                    psx = st.checkbox(
                        "Dolore Sx",
                        value=bool(rec.get("DoloreSx", False)),
                        key=f"{name}_pSx_{section}"
                    )

                rec.update({
                    "Dx": dx, "Sx": sx,
                    "DoloreDx": pdx,
                    "DoloreSx": psx
                })
                sc = ability_linear((dx + sx) / 2.0, ref)
                sym = symmetry_score(dx, sx, unit)
                st.caption(f"Score: **{sc:.1f}/10** — Δ {abs(dx - sx):.1f} {unit} — Sym: **{sym:.1f}/10**")

            else:
                val = st.slider(
                    f"Valore ({unit})",
                    0.0, ref * 1.5,
                    float(rec.get("Val", 0.0)),
                    0.1,
                    key=f"{name}_Val_{section}"
                )
                p = st.checkbox(
                    "Dolore",
                    value=bool(rec.get("Dolore", False)),
                    key=f"{name}_p_{section}"
                )
                rec.update({"Val": val, "Dolore": p})
                sc = ability_linear(val, ref)
                st.caption(f"Score: **{sc:.1f}/10**")

# Render input dinamici
render_inputs_for_section(st.session_state["section"])
# 16. Visualizzazione risultati
df_show = build_df(st.session_state["section"])
st.markdown("#### Tabella risultati")
st.dataframe(df_show, use_container_width=True)

# 17. Radar plot
try:
    if len(df_show) > 0:
        radar_buf = radar_plot(df_show, title=f"{st.session_state['section']} – Punteggi (0–10)")
        radar_img = Image.open(radar_buf)
        st.image(radar_img)
        st.caption("Radar – Punteggi (0–10)")
    else:
        radar_buf = None
except Exception as e:
    radar_buf = None
    st.warning(f"■ Radar non disponibile ({e})")
#def ebm
#def ebm_from_df(df):

# 18. Body Chart
bbuf = bodychart_image_from_state()
body_img = Image.open(io.BytesIO(bbuf.getvalue()))
st.image(body_img, caption="Body Chart – Sintesi (verde=buono, giallo=parziale, rosso=deficit; triangolo=Dolore)")

# 19. Asymmetry bar plot
try:
    if len(df_show) > 0 and "Delta" in df_show.columns:
        asym_buf = asymmetry_bar_plot(df_show, title=f"Asimmetrie – {st.session_state['section']}")
        if asym_buf:
            asym_img = Image.open(asym_buf)
            st.image(asym_img)
            st.caption("Grafico delle asimmetrie tra Dx e Sx per i test bilaterali")
except Exception as e:
    st.warning(f"■ Grafico asimmetrie non disponibile ({e})")
# 20. Commenti EBM (Evidence-Based Message)
def ebm_from_df(df):
    notes = []

    ebm_library = {
        # Squat
        "Weight Bearing Lunge Test": {
            "ref": "Bennell KL et al., 1998; Konor MM et al., 2012",
            "low_score": "Deficit dorsiflessione caviglia: rischio compensi, sollevamento tallone e stress femoro-rotuleo nello squat."
        },
        "Passive Hip Flexion": {
            "ref": "Reese NB, Bandy WD, 2020",
            "low_score": "Flessione d’anca ridotta: può limitare la profondità dello squat e aumentare i compensi lombari."
        },
        "Hip Rotation (flexed 90°)": {
            "ref": "Gajdosik RL et al., 1983; Norkin & White, 2016",
            "low_score": "Limitata rotazione d’anca: associata a compensi pelvici e rischio di impingement."
        },
        "Thoracic Extension (T4-T12)": {
            "ref": "Edmonston SJ et al., 2011",
            "low_score": "Estensione toracica ridotta: compromette il setup in panca e l’allineamento nello squat."
        },
        "Shoulder ER (adducted, low-bar)": {
            "ref": "Wilk KE et al., 2015",
            "low_score": "Limitata ER spalla: può compromettere la posizione low-bar e aumentare lo stress anteriore."
        },
        # Panca
        "Shoulder Flexion (supine)": {
            "ref": "Reese NB, Bandy WD, 2020",
            "low_score": "Flessione spalla limitata: influisce sulla profondità in panca e stressa la spalla."
        },
        "External Rotation (90° abd)": {
            "ref": "Wilk KE et al., 2015",
            "low_score": "ER spalla a 90° ridotta: possibile rischio per instabilità anteriore in overhead."
        },
        "Pectoralis Minor Length": {
            "ref": "Borstad JD, 2006",
            "low_score": "PM corto: postura in protrazione scapolare, rischio impingement e setup panca compromesso."
        },
        "Thomas Test (modified)": {
            "ref": "Harvey D, 1998",
            "low_score": "Flessibilità anca ridotta: associata a compensi in estensione lombare in panca e squat."
        },
        # Deadlift
        "Active Knee Extension (AKE)": {
            "ref": "Sahrmann SA, 2002",
            "low_score": "Estensione attiva ginocchio ridotta: può riflettere tensione posteriore e rischio compensi."
        },
        "Straight Leg Raise (SLR)": {
            "ref": "Herrington L, 2011",
            "low_score": "SLR ridotto: tensione catena posteriore, rischio compensi lombari nello stacco."
        },
        "Modified Schober (lumbar)": {
            "ref": "Macrae IF et al., 1980",
            "low_score": "Mobilità lombare ridotta: può influenzare la flessione corretta nello stacco."
        },
        "Sorensen Endurance": {
            "ref": "Moreau CE et al., 2001",
            "low_score": "Scarsa endurance lombare: rischio di cedimento precoce durante lo stacco o esercizi statici."
        },
        # Neurodinamica
        "Popliteal Knee Bend (PKB)": {
            "ref": "Shacklock M, 2005",
            "low_score": "PKB ridotto: tensione del nervo sciatico, suggerisce alterata neurodinamica distale."
        },
        "ULNT1A (Median nerve)": {
            "ref": "Nee RJ et al., 2012",
            "low_score": "Test ULNT1A positivo: sensibilità aumentata del nervo mediano o deficit scorrimento brachiale."
        }
    }

    tests_with_problems = set()

    for _, r in df.iterrows():
        test = str(r["Test"]).strip()
        score = float(r["Score"])
        pain = bool(r["Dolore"])
        sym_score = r.get("SymScore", 10.0)

        issue_found = False

        # Score insufficiente
        if score < 4:
            msg = ebm_library.get(test, {}).get("low_score", f"Deficit rilevato nel test '{test}'.")
            notes.append(f"❗ {msg}")
            tests_with_problems.add(test)
            issue_found = True

        # Dolore
        if pain:
            notes.append(f"⚠️ Dolore presente nel test '{test}': considerare irritabilità tissutale e gestione del carico.")
            tests_with_problems.add(test)
            issue_found = True

        # Asimmetria
        try:
            sym = float(sym_score)
            if sym < 7:
                notes.append(f"↔️ Asimmetria significativa nel test '{test}' (SymScore: {sym:.1f}/10).")
                tests_with_problems.add(test)
                issue_found = True
        except:
            pass

        # ✅ Test superato senza problemi
        if not issue_found:
            notes.append(f"✅ Il test '{test}' soddisfa la sufficienza.")

    # Aggiunta riferimenti SOLO per i test con problemi
    for test in sorted(tests_with_problems):
        ref = ebm_library.get(test, {}).get("ref")
        if ref:
            notes.append(f"📚 Riferimento: {ref}")

    return notes



# 20. Commenti EBM
ebm_notes = ebm_from_df(df_show)
#generazione pdf
def pdf_report(logo_bytes, athlete, evaluator, date_str, section, df, body_buf, ebm_notes, radar_buf=None):
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.4 * cm, rightMargin=1.4 * cm,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm
    )
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
        ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 8))

    # Aggiunta referenze bibliografiche se esistono
    df["Riferimenti"] = df["Test"].map(lambda t: next(
        (x[6] for sec in TESTS.values() for x in sec if x[0] == t and len(x) > 6), ""
    ))

    disp = df[["Sezione","Test","Unità","Rif","Valore","Score","Dx","Sx","Delta","SymScore","Dolore","Riferimenti"]].copy()
    # Arrotonda e formatta numeri con massimo 2 decimali
    for col in ["Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore"]:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2).astype(str)
            disp[col] = disp[col].str.replace(".0$", "", regex=True)  # toglie ".0" inutili
    table = Table([disp.columns.tolist()] + disp.values.tolist(), repeatRows=1,
                  colWidths=[2.2*cm, 6.0*cm, 1.0*cm, 1.0*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.4*cm, 1.0*cm, 1.4*cm, 1.4*cm, 6.0*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 8))

    if radar_buf:
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

    from reportlab.lib.styles import ParagraphStyle

# Stili personalizzati
style_ebm_green = ParagraphStyle(
    'green_note',
    parent=normal,
    textColor=colors.green,
    spaceAfter=6,
)

style_ebm_red = ParagraphStyle(
    'red_note',
    parent=normal,
    textColor=colors.red,
    spaceAfter=6,
)

style_ebm_default = ParagraphStyle(
    'default_note',
    parent=normal,
    spaceAfter=6,
)

# Blocco EBM
story.append(Paragraph("<b>📋 Commento clinico (EBM)</b>", styles["Heading4"]))
story.append(Spacer(1, 6))

for note in ebm_notes:
    if note.startswith("✅"):
        story.append(Paragraph(note, style_ebm_green))
    elif note.startswith("❗") or note.startswith("⚠️") or note.startswith("↔️"):
        story.append(Paragraph(note, style_ebm_red))
    else:
        story.append(Paragraph(note, style_ebm_default))

    doc.build(story)
    buf.seek(0)
return buf

# 21. Esportazione PDF e CSV
colp1, colp2 = st.columns(2)

with colp1:
    if st.button("Genera PDF", use_container_width=True):
        try:
            pdf = pdf_report(
                LOGO,
                st.session_state["athlete"],
                st.session_state["evaluator"],
                st.session_state["date"],
                st.session_state["section"],
                df_show,
                bbuf,
                ebm_notes,
                radar_buf=radar_buf
            )
            st.download_button(
                "Scarica PDF",
                data=pdf.getvalue(),
                file_name=f"Fisiomove_{st.session_state['section']}_{st.session_state['date']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Errore durante generazione PDF: {e}")

with colp2:
    if st.button("Esporta CSV", use_container_width=True):
        csv_data = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Scarica CSV",
            data=csv_data,
            file_name=f"Fisiomove_{st.session_state['section']}_{st.session_state['date']}.csv",
            mime="text/csv",
            use_container_width=True
        )
