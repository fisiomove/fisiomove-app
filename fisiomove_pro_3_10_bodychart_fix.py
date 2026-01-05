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


st.set_page_config(page_title="Fisiomove MobilityPro v. 1.0", layout="centered")


# -----------------------------
# Constants & Assets
# -----------------------------
APP_TITLE = "Fisiomove MobilityPro v. 1.0"
PRIMARY = "#1E6CF4"
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
    "ULNT1A (Median nerve)": "Test neurodinamico mediano (ULNT1A)"
}



LOGO_PATHS = ["logo 2600x1000.jpg", "logo.png", "logo.jpg"]
BODYCHART_PATHS = ["8741B9DF-86A6-45B2-AB4C-20E2D2AA3EC7.png", "body_chart.png"]


def load_logo_bytes():
    for p in LOGO_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
    # Se il file non esiste, crea un'immagine di fallback
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

    # Fallback se l'immagine non esiste
    img = Image.new("RGBA", (1200, 800), (245, 245, 245, 255))
    d = ImageDraw.Draw(img)
    d.text(
        (20, 20),
        "Body chart non disponibile",
        fill=(10, 10, 10)
    )
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
        st.session_state["section"] = "Squat"

init_state()


# -----------------------------
# Funzioni di scoring
# -----------------------------
def ability_linear(val, ref):
    try:
        if ref <= 0:
            return 0.0
        if ref == 3.0:  # ⚠️ Caso specifico: test a punteggio soggettivo 0–3 (Wall Angel Test)
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
        return 10.0 * max(0.0, 1.0 - min(diff, scale)/scale)
    except:
        return 0.0

# -----------------------------
# Sezioni e TESTS aggiornati
# -----------------------------
ALL_SECTIONS = ["Squat", "Panca", "Deadlift", "Neurodinamica", "Valutazione Generale"]

TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10.0, True,  "ankle", "Dorsiflessione in carico."),
        ("Passive Hip Flexion",       "°",  120.0, True, "hip",   "Flessione d’anca passiva."),
        ("Hip Rotation (flexed 90°)", "°",   40.0, True, "hip",   "Rotazione anca in flessione."),
        ("Wall Angel Test",           "°",   10.0, False,"thoracic", "Valutazione wall angel."),
        ("Shoulder ER (adducted, low-bar)", "°", 70.0, True, "shoulder", "ER spalla per low-bar."),
    ],
    "Panca": [
        ("Shoulder Flexion (supine)", "°", 180.0, True, "shoulder", "Flessione spalla da supino."),
        ("External Rotation (90° abd)", "°", 90.0, True, "shoulder", "ER a 90° abduzione."),
        ("Wall Angel Test",           "°", 10.0, False, "thoracic", "Valutazione wall angel."),
        ("Pectoralis Minor Length", "cm", 10.0, True, "shoulder", "Lunghezza piccolo pettorale."),
        ("Thomas Test (modified)", "°", 10.0, True, "hip", "Flessibilità flessori anca."),
    ],
    "Deadlift": [
        ("Active Knee Extension (AKE)", "°", 90.0, True, "knee", "Estensione attiva ginocchio."),
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "Test SLR."),
        ("Weight Bearing Lunge Test", "cm", 10.0, True, "ankle", "Dorsiflessione in carico."),
        ("Sorensen Endurance", "sec", 180.0, False, "lumbar", "Endurance estensori lombari."),
    ],
    "Neurodinamica": [
        ("Straight Leg Raise (SLR)", "°", 90.0, True, "hip", "SLR neurodinamico."),
        ("ULNT1A (Median nerve)", "°", 90.0, True, "shoulder", "Upper Limb Neurodynamic Test 1A."),
    ]
}
# -----------------------------
# Seed valori di default
# -----------------------------
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

# -----------------------------
# Costruzione DataFrame
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
                sc = round(ability_linear((dx + sx) / 2.0, ref), 2)
                sym = round(symmetry_score(dx, sx, unit), 2)
                rows.append([
                    sec, name, unit, ref, f"{(dx+sx)/2:.1f}", sc,
                    round(dx, 2), round(sx, 2), round(abs(dx - sx), 2), sym,
                    bool(rec.get("DoloreDx", False) or rec.get("DoloreSx", False)),
                    region
                ])
            else:
                val = float(rec.get("Val", 0))
                sc = round(ability_linear(val, ref), 2)
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
# -----------------------------
# Radar plot (score per test)
# -----------------------------
def radar_plot(df, title="Punteggi (0–10)"):
    import matplotlib.pyplot as plt
    import numpy as np
    import io

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

    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#1E6CF4")
    ax.fill(angles, values, alpha=0.25, color="#1E6CF4")

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, y=1.1, fontsize=14)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

# -----------------------------
# Radar plot per sezione (media score)
# -----------------------------
def radar_plot_per_section(df, title="Media punteggi per sezione"):
    import matplotlib.pyplot as plt
    import numpy as np
    import io

    section_means = df.groupby("Sezione")["Score"].mean().dropna()
    labels = section_means.index.tolist()
    values = section_means.values.tolist()

    if len(labels) < 3:
        return None

    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, values, color="#10A37F", linewidth=2)
    ax.fill(angles, values, color="#10A37F", alpha=0.25)

    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_ylim(0, 10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=14, y=1.1)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# -----------------------------
# Asymmetry bar plot (SymScore)
# -----------------------------
def asymmetry_bar_plot(df, title="SymScore – Simmetria Dx/Sx"):
    import matplotlib.pyplot as plt
    import io

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

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, scores, color=colors_map)

    ax.set_xlabel("SymScore (0–10)")
    ax.set_title(title)
    ax.set_xlim(0, 10)
    ax.invert_yaxis()
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{width:.1f}", va='center')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf
# -----------------------------
# Titolo app
# -----------------------------
st.markdown(f"<h2 style='color:{PRIMARY};margin-bottom:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)

# -----------------------------
# Sidebar – dati atleta e sezione
# -----------------------------
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

# -----------------------------
# Rendering input dinamico
# -----------------------------
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

# Esegui rendering
render_inputs_for_section(st.session_state["section"])
# -----------------------------
# Visualizzazione risultati
# -----------------------------
df_show = build_df(st.session_state["section"])
st.markdown("#### Tabella risultati")
st.dataframe(df_show.round(2), use_container_width=True)

# -----------------------------
# Radar plot
# -----------------------------
try:
    if len(df_show) > 0:
        radar_buf = radar_plot(df_show, title=f"{st.session_state['section']} – Punteggi (0–10)")
        radar_img = Image.open(radar_buf)
        st.image(radar_img)
        st.caption("📊 Radar – Punteggi (0–10)")
    else:
        radar_buf = None
except Exception as e:
    radar_buf = None
    st.warning(f"■ Radar non disponibile ({e})")

# -----------------------------
# Body Chart – disattivata
# -----------------------------
st.info("📉 Body Chart disattivata in questa versione.")

# -----------------------------
# Asymmetry bar plot
# -----------------------------
try:
    if len(df_show) > 0 and "Delta" in df_show.columns:
        df_sym = df_show[df_show["Delta"].notnull()].copy()

        # Converti i campi
        df_sym["Delta"] = pd.to_numeric(df_sym["Delta"], errors="coerce").round(2)
        df_sym["SymScore"] = pd.to_numeric(df_sym["SymScore"], errors="coerce").round(2)

        # Mostra tabella
        if not df_sym.empty:
            st.markdown("#### Tabella Simmetria")
            st.dataframe(
                df_sym[["Test", "Delta", "SymScore"]].round(2),
                use_container_width=True
            )

            # Plot barre
            asym_buf = asymmetry_bar_plot(df_show, title=f"Asimmetrie – {st.session_state['section']}")
            if asym_buf:
                asym_img = Image.open(asym_buf)
                st.image(asym_img)
                st.caption("📉 Grafico delle asimmetrie tra Dx e Sx")

                # ▶️ Radar per sezione (solo se 'Valutazione Generale')
                try:
                    if st.session_state["section"] == "Valutazione Generale":
                        radar_sec_buf = radar_plot_per_section(df_show, title="📌 Media punteggi per sezione")
                        if radar_sec_buf:
                            radar_sec_img = Image.open(radar_sec_buf)
                            st.image(radar_sec_img, caption="📌 Radar – Media per sezione")
                except Exception as e:
                    st.warning(f"■ Radar sezione non disponibile ({e})")
        else:
            asym_buf = None
except Exception as e:
    st.warning(f"■ Tabella simmetria non disponibile ({e})")
    asym_buf = None
# -----------------------------
# Commenti EBM
# -----------------------------
def ebm_from_df(df, friendly=False):
    notes = []
    ebm_library = {
        "Weight Bearing Lunge Test": {
            "ref": "Bennell KL et al., 1998; Konor MM et al., 2012",
            "low_score": "Mobilità della caviglia ridotta: rischio di compensi, sollevamento del tallone e stress femoro-rotuleo.",
            "friendly": "Il test della caviglia è ridotto: potresti avere difficoltà nello squat profondo."
        },
        "Passive Hip Flexion": {
            "ref": "Reese NB, Bandy WD, 2020",
            "low_score": "Flessibilità anca ridotta: può limitare la profondità dello squat.",
            "friendly": "La flessione dell’anca è un po’ limitata."
        },
        "Shoulder ER (adducted, low-bar)": {
            "ref": "Wilk KE et al., 2015",
            "low_score": "Rotazione esterna spalla ridotta: può influenzare la posizione low-bar.",
            "friendly": "La spalla ruota un po’ meno del normale."
        },
        "Wall Angel Test": {
            "ref": "Kibler WB et al., 2013; Ludewig PM et al., 2009",
            "low_score": "Deficit nel controllo scapolare e nella mobilità toracica: possibili compensi nella postura o nei movimenti overhead."
        },

        "ULNT1A (Median nerve)": {
            "ref": "Nee RJ et al., 2012",
            "low_score": "Test positivo: possibile irritazione o tensione del nervo mediano.",
            "friendly": "Test sul nervo del braccio positivo: potresti sentire tensione o fastidio."
        }
    }

    problematic_tests = {}

    for _, r in df.iterrows():
        test = str(r["Test"])
        score = float(r["Score"])
        pain = bool(r["Dolore"])
        sym = float(r.get("SymScore", 10.0))

        comment_lines = []
        issue = False

        if score < 4:
            msg = ebm_library.get(test, {}).get("friendly" if friendly else "low_score", f"Deficit rilevato nel test '{test}'.")
            comment_lines.append(f"❗ {msg}")
            issue = True
        if sym < 7:
            comment_lines.append(f"↔ Asimmetria significativa nel test '{test}' (SymScore: {sym:.1f}/10).")
            issue = True
        if pain:
            comment_lines.append("⚠️ Dolore riportato durante il test.")

        if not issue:
            comment_lines.append(f"✅ Il test '{test}' soddisfa la sufficienza.")

        notes.extend(comment_lines)

        if issue:
            problematic_tests[test] = ebm_library.get(test, {}).get("ref")

    if not friendly and problematic_tests:
        notes.append("")
        for test, ref in problematic_tests.items():
            if ref:
                notes.append(f"📚 Riferimento: {ref}")

    return notes

# -----------------------------
# Esportazione PDF
# -----------------------------
def genera_pdf(story_title, df, friendly=False):
    try:
        ebm_notes = ebm_from_df(df, friendly=friendly)
        pdf = pdf_report_no_bodychart(
            logo_bytes=LOGO,
            athlete=st.session_state["athlete"],
            evaluator=st.session_state["evaluator"],
            date_str=st.session_state["date"],
            section=st.session_state["section"],
            df=df,
            ebm_notes=ebm_notes,
            radar_buf=radar_buf,
            asym_buf=asym_buf
        )
        st.download_button(
            f"📥 Scarica {story_title}",
            data=pdf.getvalue(),
            file_name=f"Fisiomove_{'client' if friendly else 'pro'}_{st.session_state['section']}_{st.session_state['date']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Errore durante generazione PDF: {e}")

# -----------------------------
# Bottoni PDF
# -----------------------------
# -----------------------------
# Esportazione PDF (clinico + friendly) e CSV
# -----------------------------
colpdf1, colpdf2 = st.columns(2)

with colpdf1:
    if st.button("📄 Esporta PDF Clinico", use_container_width=True):
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
                "⬇️ Scarica PDF Clinico",
                data=pdf.getvalue(),
                file_name=f"Fisiomove_Report_Clinico_{st.session_state['date']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Errore durante generazione PDF clinico: {e}")

with colpdf2:
    if st.button("🧾 Esporta PDF Client Friendly", use_container_width=True):
        try:
            pdf_client = pdf_report_client_friendly(
                logo_bytes=LOGO,
                athlete=st.session_state["athlete"],
                evaluator=st.session_state["evaluator"],
                date_str=st.session_state["date"],
                section=st.session_state["section"],
                df=df_show,
                radar_buf=radar_buf,
                asym_buf=asym_buf
            )
            st.download_button(
                "⬇️ Scarica PDF Client Friendly",
                data=pdf_client.getvalue(),
                file_name=f"Fisiomove_Report_Facile_{st.session_state['date']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Errore durante generazione PDF semplificato: {e}")

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

    # Rimozione test Schober
    disp = disp[~disp["Test"].str.lower().str.contains("schober")]

    # Arrotondamenti
    for col in ["Valore", "Score", "Dx", "Sx", "Delta", "SymScore"]:
        disp[col] = pd.to_numeric(disp[col], errors="coerce").round(2)

    # Tabella
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

    # Radar punteggi
    if radar_buf:
        story.append(Paragraph("<b>Radar – Punteggi (0–10)</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10*cm, height=10*cm))
        story.append(Spacer(1, 8))

    # Asimmetrie
    if asym_buf:
        story.append(Paragraph("<b>Grafico Asimmetrie Dx/Sx</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14*cm, height=6*cm))
        story.append(Spacer(1, 8))

    # Regioni dolorose
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

    # Commento EBM
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


def pdf_report_client_friendly(
    logo_bytes,
    athlete,
    evaluator,
    date_str,
    section,
    df,
    radar_buf=None,
    asym_buf=None
):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
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
    story.append(Paragraph(f"<b>Valutazione Funzionale – {section}</b>", title))
    story.append(Spacer(1, 6))

    story.append(Paragraph(f"👤 <b>Atleta:</b> {athlete}", normal))
    story.append(Paragraph(f"🧑‍⚕️ <b>Valutatore:</b> {evaluator}", normal))
    story.append(Paragraph(f"📅 <b>Data:</b> {date_str}", normal))
    story.append(Spacer(1, 12))

    # Radar
    if radar_buf:
        story.append(Paragraph("<b>Radar delle capacità funzionali</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(radar_buf.getvalue()), width=10*cm, height=10*cm))
        story.append(Spacer(1, 12))

    # Asimmetrie
    if asym_buf:
        story.append(Paragraph("<b>Simmetria tra lato destro e sinistro</b>", normal))
        story.append(Spacer(1, 4))
        story.append(RLImage(io.BytesIO(asym_buf.getvalue()), width=14*cm, height=6*cm))
        story.append(Spacer(1, 12))

    # Spiegazione punteggi
    story.append(Paragraph("📌 Ogni test è valutato su un punteggio da 0 a 10.", normal))
    story.append(Paragraph("🔴 0–3: da migliorare • 🟡 4–6: accettabile • 🟢 7–10: ottimale", normal))
    story.append(Spacer(1, 10))

    # Tabella semplificata
    simple_rows = []
    for _, r in df.iterrows():
        score = round(float(r["Score"]), 1)
        test_name = TEST_NAME_TRANSLATIONS.get(r["Test"], r["Test"])
        simple_rows.append([test_name, f"{score}/10"])

    t = Table([["Test", "Punteggio"]] + simple_rows, repeatRows=1, colWidths=[10*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E6CF4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    doc.build(story)
    buf.seek(0)
    return buf

TESTS = {
    "Squat": [
        ("Weight Bearing Lunge Test", "cm", 10.0, True,  "ankle",    "Test caviglia: dorsiflessione in carico."),
        ("Passive Hip Flexion",       "°",  120.0, True, "hip",      "Test anca: flessione passiva supina."),
        ("Hip Rotation (flexed 90°)", "°",   40.0, True, "hip",      "Test anca: rotazione a 90° flessione."),
        ("Wall Angel Test",           "score", 3.0, False, "thoracic", "Contatto scapolare/test posturale (0–3)."),
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
