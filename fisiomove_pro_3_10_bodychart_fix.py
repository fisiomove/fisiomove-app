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
"""def bodychart_image_from_state(width=1200, height=800):
    base = BODYCHART_BASE.copy().resize((width, height))
    draw = ImageDraw.Draw(base)

    #Coordinate dei marker sulla body chart
    fx, bx = 0.255, 0.745
    points = {
        "shoulder_dx": (fx - 0.135, 0.225),
        "shoulder_sx": (fx + 0.058, 0.225),
        "hip_dx":      (fx - 0.112, 0.50),
        "hip_sx":      (fx + 0.038, 0.50),
        "knee_dx":     (fx - 0.106, 0.68),
        "knee_sx":     (fx + 0.030, 0.68),
        "ankle_dx":    (fx - 0.070, 0.94),
        "ankle_sx":    (fx + 0.025, 0.94),
        "thoracic":    (bx, 0.30),
        "lumbar":      (bx, 0.40),
    }

    f_all = build_df(st.session_state["section"])

    region_scores = {}
    region_pain = {}

    for region in points.keys():
        base_region = region.split("_")[0]
        side = None
        if "_dx" in region:
            side = "Dx"
        elif "_sx" in region:
            side = "Sx"

        score = 0.0
        pain = False

        matching = df_all[df_all["Regione"].notnull() & (df_all["Regione"].str.lower() == base_region)]
        if not matching.empty:
            try:
                if side:  # test laterale
                    vals = pd.to_numeric(matching[side], errors="coerce").dropna()
                    refs = pd.to_numeric(matching["Rif"], errors="coerce").dropna()
                    if len(vals) > 0 and len(refs) > 0:
                        avg = vals.mean()
                        ref = refs.mean()
                        score = ability_linear(avg, ref)

                    pain_col = f"Dolore{side}"
                    if pain_col in matching.columns:
                        pain_vals = matching[pain_col].astype(bool)
                        pain = pain_vals.any()

                else:  # test centrale
                    vals = pd.to_numeric(matching["Valore"], errors="coerce").dropna()
                    refs = pd.to_numeric(matching["Rif"], errors="coerce").dropna()
                    if len(vals) > 0 and len(refs) > 0:
                        avg = vals.mean()
                        ref = refs.mean()
                        score = ability_linear(avg, ref)

                    if "Dolore" in matching.columns:
                        pain = matching["Dolore"].astype(bool).any()
            except:
                score = 0.0
                pain = False

        region_scores[region] = float(np.clip(score, 0, 10))
        region_pain[region] = bool(pain)

    def score_color(score):
        if score > 7:
            return (22, 163, 74, 255)     # verde
        elif score >= 4:
            return (245, 158, 11, 255)   # giallo
        else:
            return (220, 38, 38, 255)    # rosso

    def draw_marker(xn, yn, score, pain):
        x = int(xn * width)
        y = int(yn * height)
        radius = int(10 + 6 * (1 - min(max(score, 0), 10) / 10))

        # Marker circolare colorato
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=score_color(score))

        # Checkmark bianco per punteggio alto
        if score > 7:
            draw.line((x - 4, y, x - 2, y + 6), fill=(255, 255, 255, 255), width=3)
            draw.line((x - 2, y + 6, x + 6, y - 4), fill=(255, 255, 255, 255), width=3)

        # Triangolo rosso se dolore (a destra del marker)
        if pain:
            dx = x + radius + 6
            tri = [
                (dx,     y - 6),  # punta
                (dx - 6, y + 6),  # base sinistra
                (dx + 6, y + 6),  # base destra
            ]
            draw.polygon(tri, fill=(255, 0, 0, 255))

    # disegna tutti i marker
    for region_label, coord in points.items():
        score = region_scores.get(region_label, 0.0)
        pain = region_pain.get(region_label, False)
        draw_marker(*coord, score, pain)

    # esporta immagine
    bio = io.BytesIO()
    base.save(bio, format="PNG")
    bio.seek(0)
    return bio"""


# 11. Radar plot (score)
def radar_plot(df, title="Punteggi (0–10)"):
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    labels = df["Test"].tolist()
    values = df["Score"].astype(float).tolist()

    if len(labels) < 3:
        raise ValueError("Servono almeno 3 test per il radar.")

    # Chiudere il cerchio aggiungendo il primo valore alla fine
    values += values[:1]
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Chiudere il cerchio

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#1E6CF4")
    ax.fill(angles, values, alpha=0.25, color="#1E6CF4")

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)

    # Etichette
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_title(title, y=1.1, fontsize=14)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf 

# 12. Asymmetry bar plot
def asymmetry_bar_plot(df, title="Asimmetria Dx–Sx"):
    import matplotlib.pyplot as plt
    import io

    df_bilat = df[df["Delta"].notnull()].copy()

    # ✅ Converti Delta in float, escludi righe con errori
    try:
        df_bilat["Delta"] = pd.to_numeric(df_bilat["Delta"], errors="coerce")
        df_bilat = df_bilat.dropna(subset=["Delta"])
    except Exception as e:
        print(f"Errore nella conversione di Delta: {e}")
        return None

    if df_bilat.empty:
        return None

    labels = df_bilat["Test"].tolist()
    deltas = df_bilat["Delta"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, deltas, color="#FF6B6B")

    ax.set_xlabel("Asimmetria (unità originali)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
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
            for item in TESTS[s]:
                items.append((s, *item))  # Aggiungi la sezione originale
    else:
        for item in TESTS.get(section, []):
            items.append((section, *item))  # Aggiungi la sezione originale

    for sec, name, unit, ref, bilat, region, desc in items:
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
                        key=f"{sec}_{name}_Dx"
                    )
                    pdx = st.checkbox(
                        "Dolore Dx",
                        value=bool(rec.get("DoloreDx", False)),
                        key=f"{sec}_{name}_pDx"
                    )
                with c2:
                    sx = st.slider(
                        f"Sx ({unit})",
                        0.0, ref * 1.5,
                        float(rec.get("Sx", 0.0)),
                        0.1,
                        key=f"{sec}_{name}_Sx"
                    )
                    psx = st.checkbox(
                        "Dolore Sx",
                        value=bool(rec.get("DoloreSx", False)),
                        key=f"{sec}_{name}_pSx"
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
                    key=f"{sec}_{name}_Val"
                )
                p = st.checkbox(
                    "Dolore",
                    value=bool(rec.get("Dolore", False)),
                    key=f"{sec}_{name}_p"
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
# ✅ Body chart disattivata in questa versione — se vuoi riattivarla, decommenta le righe qui sotto
# bbuf = bodychart_image_from_state()
# if bbuf:
#     body_img = Image.open(io.BytesIO(bbuf.getvalue()))
#     st.image(body_img, caption="Body Chart – Sintesi")
# else:
#     st.warning("⚠️ Body chart non disponibile.")

# 📉 Placeholder informativo
    st.info("📉 Body Chart disattivata in questa versione.")


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
def pdf_report_no_bodychart(
    logo_bytes,
    athlete,
    evaluator,
    date_str,
    section,
    df,
    ebm_notes,
    radar_buf=None,
    asym_buf=None
):
    import io
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
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

    # Tabella risultati
    disp = df[["Sezione", "Test", "Unità", "Rif", "Valore", "Score", "Dx", "Sx", "Delta", "SymScore", "Dolore"]].copy()
    disp["Delta"] = pd.to_numeric(disp["Delta"], errors="coerce").round(2)
    disp["SymScore"] = pd.to_numeric(disp["SymScore"], errors="coerce").round(2)

    table = Table([disp.columns.tolist()] + disp.values.tolist(), repeatRows=1,
                  colWidths=[2.2*cm, 6.5*cm, 1.2*cm, 1.2*cm, 1.6*cm, 1.6*cm, 1.4*cm, 1.4*cm, 1.2*cm, 1.6*cm, 1.6*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E6CF4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    # ▶️ Radar plot
    if radar_buf:
        story.append(Paragraph("<b>Radar – Punteggi (0–10)</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10*cm, height=10*cm))
        story.append(Spacer(1, 8))

    # ▶️ Asymmetry bar plot
    if asym_buf:
        story.append(Paragraph("<b>Grafico Asimmetrie Dx/Sx</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14*cm, height=6*cm))
        story.append(Spacer(1, 8))

    # ▶️ Regioni dolorose
    pain_regions = []

    for _, row in df.iterrows():
        regione = row.get("Regione", "").capitalize()
        if not regione:
            continue

        if row.get("DoloreDx", False):
            pain_regions.append(f"{regione} destra")
        if row.get("DoloreSx", False):
            pain_regions.append(f"{regione} sinistra")
        if row.get("Dolore", False) and not (row.get("DoloreDx") or row.get("DoloreSx")):
            pain_regions.append(f"{regione}")

    pain_regions = list(dict.fromkeys(pain_regions))

    story.append(Paragraph("<b>🩹 Regioni dolorose riscontrate durante il test:</b>", normal))
    if pain_regions:
        for reg in pain_regions:
            story.append(Paragraph(f"• {reg.capitalize()}", normal))
    else:
        story.append(Paragraph("Nessuna regione segnalata come dolorosa.", normal))

    story.append(Spacer(1, 12))

    # ▶️ Commento clinico EBM
    story.append(Paragraph("<b>🧠 Commento clinico (EBM)</b>", normal))
    story.append(Spacer(1, 4))

    if ebm_notes:
        for note in ebm_notes:
            if isinstance(note, str):
                story.append(Paragraph(f"• {note}", normal))
                story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("Nessun commento disponibile.", normal))

    doc.build(story)
    buf.seek(0)
    return buf

# 21. Esportazione PDF e CSV
colp1, colp2 = st.columns(2)

with colp1:
    if st.button("Genera PDF", use_container_width=True):
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
                asym_buf=asym_buf
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
