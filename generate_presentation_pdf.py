"""
Script per generare PDF professionale della presentazione Fisiomove
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from PIL import Image, ImageDraw
import io

# Colori aziendali
PRIMARY = colors.HexColor("#1E6CF4")
SECONDARY = colors.HexColor("#4A90E2")
SUCCESS = colors.HexColor("#28a745")
WARNING = colors.HexColor("#ffc107")
DANGER = colors.HexColor("#dc3545")

def create_logo():
    """Crea logo se non esiste"""
    logo_paths = ["logo 2600x1000.jpg", "logo.png", "logo.jpg"]
    for p in logo_paths:
        if os.path.exists(p):
            return p
    
    # Crea logo placeholder
    img = Image.new("RGB", (1000, 260), (30, 108, 244))
    d = ImageDraw.Draw(img)
    try:
        d.text((30, 100), "Fisiomove", fill=(255, 255, 255))
    except:
        pass
    img.save("logo_temp.png")
    return "logo_temp.png"

def generate_presentation_pdf():
    """Genera PDF professionale della presentazione"""
    
    filename = "Fisiomove_Presentazione_Professionale.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=PRIMARY,
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=SECONDARY,
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        textColor=PRIMARY,
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=SECONDARY,
        spaceAfter=8,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        fontName='Helvetica'
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=11,
        leftIndent=20,
        spaceAfter=6,
        fontName='Helvetica'
    )
    
    # Logo e titolo
    logo_path = create_logo()
    try:
        logo = RLImage(logo_path, width=12*cm, height=3.12*cm)
        story.append(logo)
    except:
        pass
    
    story.append(Spacer(1, 1*cm))
    
    # Titolo principale
    story.append(Paragraph("FISIOMOVE MOBILITYPRO", title_style))
    story.append(Paragraph("Sistema Professionale di Valutazione Fisioterapica", subtitle_style))
    story.append(Spacer(1, 1*cm))
    
    # Introduzione
    story.append(Paragraph("Benvenuto", heading_style))
    intro_text = """
    Fisiomove MobilityPro è un sistema avanzato di valutazione fisioterapica che permette di 
    <b>analizzare con precisione la mobilità articolare</b> e creare <b>programmi di intervento 
    personalizzati</b> basati su obiettivi specifici del paziente.
    """
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 0.5*cm))
    
    # Cosa possiamo fare per te
    story.append(Paragraph("Cosa Possiamo Fare per Te", heading_style))
    
    story.append(Paragraph("✓ Valutazione Completa e Oggettiva", subheading_style))
    bullets = [
        "Test scientificamente validati per misurare la mobilità articolare",
        "Analisi bilaterale (destra vs sinistra) per identificare asimmetrie",
        "Monitoraggio del dolore durante ogni movimento",
        "Valutazione sport-specifica per atleti (Powerlifting, CrossFit, Running, Calcio, Tennis, ecc.)"
    ]
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", bullet_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("✓ Risultati Chiari e Comprensibili", subheading_style))
    bullets = [
        "Score visivi immediati (da 0 a 10) per ogni test",
        "Grafici radar che mostrano il profilo di mobilità",
        "Identificazione automatica delle aree critiche da migliorare",
        "Comparazione con valori di riferimento clinici"
    ]
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", bullet_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("✓ Programmi Personalizzati", subheading_style))
    bullets = [
        "Protocolli di esercizi specifici per le limitazioni individuali",
        "Esercizi di mobilità mirati alle articolazioni rigide",
        "Progressioni graduali adattate al livello del paziente",
        "Raccomandazioni sport-specifiche per atleti"
    ]
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", bullet_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("✓ Progressione nel Tempo", subheading_style))
    bullets = [
        "Storico completo di tutte le valutazioni",
        "Grafici di miglioramento che mostrano i progressi",
        "Confronto tra valutazioni diverse nel tempo",
        "Motivazione visibile dei risultati ottenuti"
    ]
    for bullet in bullets:
        story.append(Paragraph(f"• {bullet}", bullet_style))
    
    # Nuova pagina
    story.append(PageBreak())
    
    # Come funziona
    story.append(Paragraph("Come Funziona la Valutazione", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    phases = [
        ("1. Anamnesi Iniziale (10-15 minuti)", [
            "Storia clinica e infortuni precedenti",
            "Sport praticato e frequenza di allenamento",
            "Sintomi attuali e obiettivi",
            "Screening per eventuali red flags"
        ]),
        ("2. Test Oggettivi di Mobilità (20-30 minuti)", [
            "Weight Bearing Lunge Test (caviglia)",
            "Passive Hip Flexion (anca)",
            "Shoulder Flexion (spalla)",
            "Straight Leg Raise (catena posteriore)",
            "Thomas Test (flessori anca)",
            "E molti altri, selezionati in base agli obiettivi"
        ]),
        ("3. Analisi Risultati Immediata (10 minuti)", [
            "Punteggi per ogni test",
            "Aree di forza e debolezza",
            "Confronto destra/sinistra",
            "Interpretazione clinica dei risultati"
        ]),
        ("4. Piano di Intervento Personalizzato (10 minuti)", [
            "Esercizi specifici per le limitazioni",
            "Frequenza e dosaggio ottimali",
            "Progressione nel tempo",
            "Strategie di gestione del dolore"
        ]),
        ("5. Report Professionale in PDF", [
            "Tutti i dati e risultati della valutazione",
            "Grafici e visualizzazioni",
            "Protocolli di esercizi dettagliati",
            "Raccomandazioni cliniche prioritarie"
        ])
    ]
    
    for phase_title, phase_items in phases:
        story.append(Paragraph(phase_title, subheading_style))
        for item in phase_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 0.3*cm))
    
    # Nuova pagina
    story.append(PageBreak())
    
    # Perché scegliere Fisiomove
    story.append(Paragraph("Perché Scegliere Fisiomove MobilityPro", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    benefits = [
        ("✨ Precisione e Oggettività", 
         "Non ci basiamo su sensazioni: ogni movimento è misurato con precisione (gradi, centimetri, secondi)"),
        ("🎓 Basato su Evidenza Scientifica", 
         "Utilizziamo test validati dalla letteratura scientifica internazionale"),
        ("🎯 Personalizzazione Totale", 
         "Ogni programma è creato specificamente per il paziente e i suoi obiettivi"),
        ("📊 Trasparenza Completa", 
         "Il paziente vede esattamente cosa viene misurato e perché è importante"),
        ("📈 Motivazione Tangibile", 
         "Vedere i progressi nel tempo è la migliore motivazione per continuare"),
        ("💼 Professionalità", 
         "Report dettagliati condivisibili con altri professionisti sanitari")
    ]
    
    for icon_title, description in benefits:
        story.append(Paragraph(f"<b>{icon_title}</b>", subheading_style))
        story.append(Paragraph(description, body_style))
        story.append(Spacer(1, 0.2*cm))
    
    # Per chi è indicato
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Per Chi È Indicato", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    target_groups = [
        ("🏋️ Atleti Competitivi", [
            "Powerlifter, CrossFitter, Runner, Calciatori, Tennisti",
            "Ottimizzazione della performance",
            "Prevenzione infortuni",
            "Identificazione limitazioni che impattano la tecnica"
        ]),
        ("💪 Sportivi Amatoriali", [
            "Miglioramento nel proprio sport",
            "Dolori ricorrenti durante l'attività",
            "Prevenzione problemi futuri"
        ]),
        ("🩺 Persone con Dolore Cronico", [
            "Mal di schiena ricorrente",
            "Dolore a spalla, anca, ginocchio",
            "Limitazioni funzionali nella vita quotidiana"
        ]),
        ("🎯 Prevenzione", [
            "Mantenimento della salute articolare",
            "Lavoro sedentario o ripetitivo",
            "Storia familiare di problemi articolari"
        ])
    ]
    
    for group_title, group_items in target_groups:
        story.append(Paragraph(f"<b>{group_title}</b>", subheading_style))
        for item in group_items:
            story.append(Paragraph(f"• {item}", bullet_style))
        story.append(Spacer(1, 0.2*cm))
    
    # Nuova pagina
    story.append(PageBreak())
    
    # Cosa aspettarsi
    story.append(Paragraph("Cosa Aspettarsi dalla Prima Valutazione", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("<b>Prima dell'Appuntamento</b>", subheading_style))
    before_items = [
        "<b>Abbigliamento:</b> vestiti comodi che permettano movimento",
        "<b>Durata:</b> prevedi circa 60-75 minuti totali",
        "<b>Porta con te:</b> eventuali referti medici precedenti"
    ]
    for item in before_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("<b>Durante la Valutazione</b>", subheading_style))
    during_items = [
        "Ogni test viene spiegato prima di eseguirlo",
        "Nessun test deve causare dolore intenso",
        "Possibilità di fare domande in qualsiasi momento",
        "Tutto viene registrato in tempo reale nel sistema"
    ]
    for item in during_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    story.append(Spacer(1, 0.3*cm))
    
    story.append(Paragraph("<b>Dopo la Valutazione</b>", subheading_style))
    after_items = [
        "Report PDF immediato via email",
        "Accesso alla cronologia di tutte le valutazioni",
        "Possibilità di programmare rivalutazioni periodiche (ogni 4-6 settimane)",
        "Disponibilità per chiarimenti e domande"
    ]
    for item in after_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    
    # Domande frequenti
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Domande Frequenti", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    faqs = [
        ("È doloroso?", 
         "No, i test valutano la mobilità ai limiti del comfort. Se c'è dolore, viene registrato ma non si forza mai."),
        ("Quanto dura l'effetto?", 
         "Non è un trattamento singolo, ma un sistema di valutazione e monitoraggio continuo. I protocolli di esercizi vanno seguiti regolarmente."),
        ("Ogni quanto ripetere la valutazione?", 
         "Tipicamente ogni 4-6 settimane per monitorare i progressi, ma dipende dal caso specifico."),
        ("È adatto ai principianti?", 
         "Assolutamente sì! I test sono adattabili a qualsiasi livello di fitness."),
        ("Posso condividere il report?", 
         "Certamente! Il report PDF professionale è pensato per essere condiviso con medici, preparatori atletici, coach, ecc.")
    ]
    
    for question, answer in faqs:
        story.append(Paragraph(f"<b>Q: {question}</b>", subheading_style))
        story.append(Paragraph(f"A: {answer}", body_style))
        story.append(Spacer(1, 0.2*cm))
    
    # Nuova pagina
    story.append(PageBreak())
    
    # Esempio caso reale
    story.append(Paragraph("Esempio Caso Reale", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    case_data = [
        ["Paziente:", "Marco, 28 anni, powerlifter"],
        ["Problema:", "Squat profondo problematico, ginocchia che 'non scendono', fastidio anca sinistra"],
        ["", ""],
        ["Test Caviglia:", "DX 11cm ✓ | SX 10cm ✓ (entrambe nella norma)"],
        ["Test Flessione Anca:", "DX 118° ⚠ | SX 102° ✗ (sinistra limitata!)"],
        ["Test Rotazione Anca:", "DX 38° ⚠ | SX 28° ✗ (asimmetria significativa!)"],
        ["", ""],
        ["Diagnosi:", "Il problema NON era la caviglia (come pensava), ma l'anca sinistra molto rigida"],
        ["", ""],
        ["Intervento:", "Protocollo mobilità anca 3x/settimana con esercizi specifici"],
        ["", ""],
        ["Dopo 6 settimane:", "Flessione anca SX: 102° → 115° (+13°)"],
        ["", "Rotazione anca SX: 28° → 36° (+8°)"],
        ["", "Squat depth migliorato, zero fastidio"],
    ]
    
    case_table = Table(case_data, colWidths=[4*cm, 10*cm])
    case_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), PRIMARY),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    
    story.append(case_table)
    
    # Contatti
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Come Prenotare", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    contact_text = """
    <b>📧 Email:</b> [Inserire email]<br/>
    <b>📱 Telefono:</b> [Inserire telefono]<br/>
    <b>📍 Indirizzo:</b> [Inserire indirizzo studio]<br/>
    <b>🕐 Orari:</b> [Inserire orari apertura]<br/>
    <br/>
    <b>💰 Investimento:</b><br/>
    Prima Valutazione Completa: [€X] - Include valutazione 60-75 min, report PDF, programma personalizzato<br/>
    Rivalutazioni: [€Y] ogni 4-6 settimane
    """
    story.append(Paragraph(contact_text, body_style))
    
    # Chiamata all'azione
    story.append(Spacer(1, 0.8*cm))
    cta_style = ParagraphStyle(
        'CTA',
        parent=styles['Normal'],
        fontSize=14,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        spaceAfter=10
    )
    story.append(Paragraph("Il Primo Passo È Sempre il Più Importante", cta_style))
    story.append(Paragraph("Non Aspettare che un Piccolo Problema Diventi Cronico", cta_style))
    
    # Footer
    story.append(Spacer(1, 1*cm))
    footer_text = """
    <i>La vostra mobilità è la base della vostra salute e delle vostre performance</i>
    """
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    story.append(Paragraph(footer_text, footer_style))
    
    # Note legali
    story.append(PageBreak())
    story.append(Paragraph("Note Legali e Informazioni Professionali", heading_style))
    story.append(Spacer(1, 0.3*cm))
    
    legal_text = """
    Le valutazioni fornite da Fisiomove MobilityPro hanno scopo fisioterapico e non sostituiscono 
    la diagnosi medica. In caso di dolore acuto, trauma recente o patologia sospetta, consultare 
    prima un medico.<br/><br/>
    
    <b>Sistema sviluppato secondo:</b><br/>
    • Standard clinici internazionali<br/>
    • Linee guida evidence-based<br/>
    • Best practices in fisioterapia sportiva<br/>
    • Normative privacy e trattamento dati sanitari (GDPR compliant)<br/><br/>
    
    <b>Informazioni Professionista:</b><br/>
    Dott./Dott.ssa [Nome Fisioterapista]<br/>
    [Titolo professionale]<br/>
    [Numero iscrizione albo]<br/>
    [Specializzazioni]<br/><br/>
    
    © 2026 Fisiomove MobilityPro - Tutti i diritti riservati
    """
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    story.append(Paragraph(legal_text, disclaimer_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF generato con successo: {filename}")
    return filename

if __name__ == "__main__":
    try:
        pdf_file = generate_presentation_pdf()
        print(f"\n📄 Presentazione PDF creata: {pdf_file}")
        print("\n💡 Ricorda di personalizzare:")
        print("   - Contatti (email, telefono, indirizzo)")
        print("   - Tariffe")
        print("   - Nome fisioterapista e credenziali")
        print("   - Eventuali promozioni")
    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        import traceback
        traceback.print_exc()
