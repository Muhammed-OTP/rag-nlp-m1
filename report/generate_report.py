"""Phase 7: generate the project's PDF report (>= 10 pages, in French).

Usage: python -m report.generate_report
Reads evaluation/results.csv and the PNG charts in visualizations/ (and
report/architecture.png) to build report/Rapport_RAG_NLP.pdf.
"""

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, GROQ_MODEL, TOP_K

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIZ_DIR = os.path.join(ROOT, "visualizations")
RESULTS_FILE = os.path.join(ROOT, "evaluation", "results.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "Rapport_RAG_NLP.pdf")

AUTHOR = "Mohamed Salem Ebnou Echvagha Oubeid - C34613"
MODULE = "Module NLP - Projet RAG (Master 1, Semestre 2, 2025-2026)"

SOURCES = [
    "Attention (machine learning)", "BERT (language model)", "BLEU", "Bag-of-words model",
    "Cosine similarity", "FastText", "Fine-tuning (deep learning)", "GPT (language model)",
    "GloVe (machine learning)", "Hallucination (artificial intelligence)", "Information retrieval",
    "LangChain", "Lemmatization", "Long short-term memory", "N-gram", "Named-entity recognition",
    "Perplexity (natural language processing)", "Question answering (computing)", "ROUGE (metric)",
    "Recurrent neural network", "Retrieval-augmented generation", "Sentence embedding",
    "Sentiment analysis", "Seq2seq", "Stemming", "Stop word", "TF-IDF", "Text classification",
    "Text segmentation", "Tokenization (lexical analysis)", "Transfer learning",
    "Transformer (machine learning)", "Vector database", "Word2vec",
]


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=26, spaceAfter=12))
    styles.add(ParagraphStyle("SubTitle", parent=styles["Normal"], fontSize=14, alignment=1, spaceAfter=6))
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=18, spaceAfter=10))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle("Body", parent=styles["BodyText"], alignment=4, spaceAfter=8, leading=15))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=9, alignment=1,
                               textColor=colors.grey, spaceAfter=14))
    styles.add(ParagraphStyle("BulletItem", parent=styles["Body"], leftIndent=14, bulletIndent=2))
    return styles


def title_page(styles):
    flow = [
        Spacer(1, 5 * cm),
        Paragraph("Conception d'un systeme RAG", styles["TitleBig"]),
        Paragraph("Assistant question-reponse sur les concepts de NLP/Machine Learning", styles["SubTitle"]),
        Spacer(1, 2 * cm),
        Paragraph(f"<b>Auteur :</b> {AUTHOR}", styles["SubTitle"]),
        Paragraph(MODULE, styles["SubTitle"]),
        Paragraph("Rapport de projet - Retrieval-Augmented Generation", styles["SubTitle"]),
        Spacer(1, 1 * cm),
        Paragraph("Domaine choisi : concepts fondamentaux du traitement automatique du langage "
                  "naturel (NLP) et du machine learning, a partir d'un corpus d'articles Wikipedia.",
                  styles["SubTitle"]),
        PageBreak(),
    ]
    return flow


def table_of_contents(styles):
    sections = [
        "1. Introduction",
        "2. Architecture du systeme",
        "3. Donnees et preparation du corpus",
        "4. Implementation",
        "5. Methodologie d'evaluation",
        "6. Resultats et visualisations",
        "7. Discussion",
        "8. Conclusion et perspectives",
        "9. References",
    ]
    flow = [Paragraph("Table des matieres", styles["H1"])]
    for s in sections:
        flow.append(Paragraph(s, styles["Body"]))
    flow.append(PageBreak())
    return flow


def section_introduction(styles):
    flow = [Paragraph("1. Introduction", styles["H1"])]
    flow.append(Paragraph(
        "Ce projet a ete realise dans le cadre du module NLP du Master 1. Il consiste a concevoir, "
        "implementer et evaluer un systeme complet de generation augmentee par recuperation "
        "(Retrieval-Augmented Generation, RAG), capable de repondre a des questions en langage "
        "naturel en s'appuyant sur un corpus de documents et un grand modele de langage (LLM).",
        styles["Body"]))
    flow.append(Paragraph(
        "Le domaine choisi est celui des <b>concepts fondamentaux du NLP et du machine learning</b> "
        "(transformers, embeddings, recherche d'information, metriques d'evaluation, etc.). Ce choix "
        "permet de constituer un corpus coherent et de qualite a partir d'articles Wikipedia, tout en "
        "restant directement utile pour reviser les notions vues dans le module lui-meme : le systeme "
        "peut etre utilise comme un assistant de revision repondant a des questions sur ces concepts.",
        styles["Body"]))
    flow.append(Paragraph(
        "L'objectif general du projet est de couvrir l'ensemble de la chaine d'un systeme RAG : "
        "collecte et nettoyage des donnees, decoupage en chunks, indexation vectorielle, pipeline de "
        "recuperation et de generation, interface utilisateur, evaluation quantitative et visualisation "
        "des resultats. Chaque etape est implementee dans un script Python dedie et documentee dans "
        "ce rapport.",
        styles["Body"]))
    flow.append(Paragraph("Le rapport est organise comme suit :", styles["Body"]))
    bullets = [
        "L'<b>architecture</b> generale du systeme (section 2) ;",
        "Les <b>donnees</b> utilisees et leur preparation (section 3) ;",
        "Les choix d'<b>implementation</b> (modele d'embedding, base vectorielle, LLM, "
        "prompt, interface) (section 4) ;",
        "La <b>methodologie d'evaluation</b> (section 5) ;",
        "Les <b>resultats</b> obtenus, accompagnes de graphiques (section 6) ;",
        "Une <b>discussion</b> sur les forces et limites du systeme (section 7) ;",
        "La <b>conclusion</b> et les pistes d'amelioration (section 8).",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(PageBreak())
    return flow


def section_architecture(styles):
    flow = [Paragraph("2. Architecture du systeme", styles["H1"])]
    flow.append(Paragraph(
        "Le systeme suit l'architecture classique d'un pipeline RAG, divisee en deux phases : une "
        "phase d'<b>indexation hors ligne</b> (executee une seule fois, ou a chaque mise a jour du "
        "corpus) et une phase de <b>reponse en ligne</b> (executee a chaque question posee par "
        "l'utilisateur). La figure 1 illustre ce flux.",
        styles["Body"]))
    flow.append(Image(os.path.join(os.path.dirname(__file__), "architecture.png"), width=14.5 * cm, height=8.96 * cm))
    flow.append(Paragraph("Figure 1 - Architecture generale du pipeline RAG.", styles["Caption"]))

    flow.append(Paragraph("2.1 Phase d'indexation (hors ligne)", styles["H2"]))
    bullets1 = [
        f"<b>Corpus</b> : 36 articles Wikipedia telecharges automatiquement par "
        f"<font face='Courier'>collect_corpus.py</font> ;",
        "<b>Nettoyage et chunking</b> : suppression des pages d'homonymie et du contenu "
        "non pertinent (sections \"See also\", \"References\", etc.), puis decoupage en chunks "
        f"de {CHUNK_SIZE} caracteres avec un chevauchement de {CHUNK_OVERLAP} caracteres "
        "(<font face='Courier'>src/prepare_data.py</font>) ;",
        f"<b>Embeddings</b> : chaque chunk est vectorise avec le modele "
        f"<font face='Courier'>{EMBEDDING_MODEL}</font> (<font face='Courier'>sentence-transformers</font>) ;",
        "<b>Index vectoriel</b> : les vecteurs, textes et metadonnees (document source) sont "
        "stockes dans une collection ChromaDB persistante (<font face='Courier'>src/build_index.py</font>).",
    ]
    for b in bullets1:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))

    flow.append(Paragraph("2.2 Phase de reponse (en ligne)", styles["H2"]))
    bullets2 = [
        "La question de l'utilisateur est encodee avec le meme modele d'embedding ;",
        f"ChromaDB renvoie les <font face='Courier'>top-{TOP_K}</font> chunks les plus proches "
        "(similarite cosinus) ;",
        "Ces chunks sont inseres dans un template de prompt avec la question ;",
        f"Le prompt est envoye au LLM via l'API Groq (modele "
        f"<font face='Courier'>{GROQ_MODEL}</font>), qui genere une reponse ancree dans le contexte ;",
        "La reponse, les sources utilisees et des metriques (temps de reponse, fidelite) sont "
        "affichees dans l'interface Streamlit.",
    ]
    for b in bullets2:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(PageBreak())
    return flow


def section_data(styles):
    flow = [Paragraph("3. Donnees et preparation du corpus", styles["H1"])]
    flow.append(Paragraph(
        "Le corpus est constitue de 36 articles Wikipedia en anglais, couvrant les principaux concepts "
        "du NLP et du machine learning abordes dans le module (modeles de langage, embeddings, "
        "architectures de reseaux de neurones, metriques d'evaluation, etc.). Les articles sont "
        "telecharges automatiquement avec la bibliotheque <font face='Courier'>wikipedia</font> et "
        "sauvegardes en texte brut dans <font face='Courier'>data/raw/</font>.",
        styles["Body"]))

    flow.append(Paragraph("3.1 Nettoyage", styles["H2"]))
    flow.append(Paragraph(
        "Deux articles se sont reveles etre des pages d'homonymie (\"BM25\" et \"Tokenization\") et "
        "ne contenaient pas de contenu exploitable ; ils ont ete ecartes, le sujet de la tokenisation "
        "etant deja couvert par l'article <i>Tokenization (lexical analysis)</i>. Le corpus final "
        "compte donc <b>34 documents</b>, ce qui respecte la contrainte minimale de 30 documents. "
        "Pour chaque document conserve, les sections de fin generiques et non informatives "
        "(\"See also\", \"References\", \"Notes\", \"External links\") sont supprimees afin de ne "
        "garder que le contenu encyclopedique pertinent.",
        styles["Body"]))

    flow.append(Paragraph("3.2 Decoupage en chunks", styles["H2"]))
    flow.append(Paragraph(
        f"Chaque document nettoye est decoupe en chunks de <b>{CHUNK_SIZE} caracteres</b> avec un "
        f"chevauchement de <b>{CHUNK_OVERLAP} caracteres</b> entre chunks consecutifs. Ce "
        "chevauchement permet d'eviter qu'une information importante soit coupee entre deux chunks "
        "et perdue lors de la recherche. Ce decoupage produit <b>1443 chunks</b>, stockes avec leur "
        "identifiant, leur document source et leur texte dans "
        "<font face='Courier'>data/processed/chunks.jsonl</font> (format JSON Lines).",
        styles["Body"]))

    flow.append(Paragraph("3.3 Documents du corpus", styles["H2"]))
    flow.append(Paragraph(
        "Le tableau suivant liste les 34 articles Wikipedia utilises (titres originaux en anglais) :",
        styles["Body"]))

    half = (len(SOURCES) + 1) // 2
    rows = []
    for i in range(half):
        left = SOURCES[i]
        right = SOURCES[i + half] if i + half < len(SOURCES) else ""
        rows.append([left, right])
    table = Table(rows, colWidths=[8.5 * cm, 8.5 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ]))
    flow.append(table)
    flow.append(PageBreak())
    return flow


def section_implementation(styles):
    flow = [Paragraph("4. Implementation", styles["H1"])]

    flow.append(Paragraph("4.1 Choix techniques", styles["H2"]))
    flow.append(Paragraph(
        "Les choix techniques ont ete centralises dans <font face='Courier'>config.py</font> afin "
        "d'eviter toute duplication de constantes (taille de chunk, chevauchement, modele "
        "d'embedding, top-k, modele LLM).",
        styles["Body"]))
    rows = [
        ["Composant", "Choix", "Justification"],
        ["Langage", "Python 3.13", "Ecosysteme NLP/ML mature (sentence-transformers, ChromaDB, Streamlit)."],
        ["Embeddings", EMBEDDING_MODEL,
         "Modele compact (~80 Mo), rapide en CPU, bonnes performances pour la recherche "
         "semantique de phrases courtes."],
        ["Base vectorielle", "ChromaDB (persistant)",
         "Simple a integrer, ne necessite pas de serveur externe, persistance locale sur disque."],
        ["LLM", f"Groq - {GROQ_MODEL}",
         "API gratuite et tres rapide (inference LPU), modele open-weight performant."],
        ["Interface", "Streamlit",
         "Permet de creer rapidement une interface web interactive en Python pur, avec onglets "
         "Chat / Evaluation / Gestion du corpus."],
        ["Decoupage", f"{CHUNK_SIZE} caracteres / chevauchement {CHUNK_OVERLAP}",
         "Compromis entre granularite (chunks suffisamment specifiques) et contexte "
         "(chunks suffisamment longs pour etre comprehensibles seuls)."],
        [f"Top-k", str(TOP_K),
         "Nombre de chunks fournis au LLM ; suffisant pour couvrir la reponse sans diluer le "
         "contexte avec des passages non pertinents."],
    ]
    table = Table(rows, colWidths=[3.2 * cm, 4 * cm, 9.8 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3a6ea5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 0.3 * cm))

    flow.append(Paragraph("4.2 Pipeline de recuperation et de generation", styles["H2"]))
    flow.append(Paragraph(
        "Le module <font face='Courier'>src/rag_pipeline.py</font> implemente les fonctions "
        "centrales du systeme : <font face='Courier'>retrieve()</font> (encode la question et "
        "interroge ChromaDB), <font face='Courier'>build_prompt()</font> (assemble le contexte "
        "recupere et la question dans un template) et "
        "<font face='Courier'>generate_answer()</font> (appelle le LLM Groq et retourne la reponse "
        "ainsi que les chunks sources). Le template de prompt demande explicitement au modele de "
        "repondre uniquement a partir du contexte fourni, et d'indiquer s'il ne connait pas la "
        "reponse - ce qui limite les hallucinations.",
        styles["Body"]))
    flow.append(Paragraph(
        "Une version interactive en ligne de commande (<font face='Courier'>python -m "
        "src.rag_pipeline</font>) permet de poser des questions et d'afficher la reponse ainsi que "
        "les documents sources utilises.",
        styles["Body"]))

    flow.append(Paragraph("4.3 Interface Streamlit", styles["H2"]))
    flow.append(Paragraph(
        "L'interface (<font face='Courier'>app.py</font>) est organisee en trois onglets :",
        styles["Body"]))
    bullets = [
        "<b>Chat</b> : interface conversationnelle pour poser des questions, avec affichage du "
        "temps de reponse, du nombre de chunks utilises, d'un score de fidelite, et des extraits "
        "sources avec leur score de similarite ;",
        "<b>Evaluation</b> : permet de charger un jeu de questions (JSON/CSV), de lancer "
        "l'evaluation automatique, d'afficher les metriques agregees, des graphiques interactifs "
        "(Plotly) et d'exporter les resultats en CSV ou en PDF ;",
        "<b>Gestion du corpus</b> : affiche des statistiques sur le corpus (nombre de documents, "
        "de chunks, taille moyenne), permet d'ajouter de nouveaux documents (PDF/TXT), de "
        "selectionner les sources a indexer et de reconstruire l'index vectoriel.",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(Paragraph(
        "La barre laterale permet de choisir le modele LLM Groq, d'ajuster le nombre de chunks "
        "recuperes (k) et le seuil de similarite, et d'afficher l'etat de l'index vectoriel et de "
        "la cle API.",
        styles["Body"]))
    flow.append(PageBreak())
    return flow


def section_evaluation_methodology(styles):
    flow = [Paragraph("5. Methodologie d'evaluation", styles["H1"])]
    flow.append(Paragraph(
        "Un jeu de <b>34 questions</b> a ete constitue manuellement "
        "(<font face='Courier'>evaluation/eval_questions.json</font>), couvrant l'ensemble des 34 "
        "documents du corpus - une question par concept, et quelques questions associees a deux "
        "documents lorsque les notions sont etroitement liees (par exemple lemmatisation/"
        "stemmatisation, ou LSTM/RNN). Cela depasse le minimum de 20 questions demande par le sujet.",
        styles["Body"]))
    flow.append(Paragraph(
        "Pour chaque question, le script <font face='Courier'>evaluation/evaluate.py</font> execute "
        "le pipeline complet (recuperation + generation) et calcule quatre metriques :",
        styles["Body"]))
    bullets = [
        f"<b>Precision@{TOP_K}</b> : proportion des {TOP_K} chunks recuperes qui appartiennent "
        "effectivement au(x) document(s) attendu(s) pour la question ;",
        f"<b>Recall@{TOP_K}</b> : proportion des documents attendus qui figurent parmi les chunks "
        "recuperes (indique si l'information necessaire a ete retrouvee) ;",
        "<b>Fidelite (faithfulness)</b> : proportion des mots de la reponse generee qui apparaissent "
        "egalement dans le contexte recupere - une mesure simple, sans dependance externe, de "
        "l'ancrage de la reponse dans les sources (absence d'hallucination) ;",
        "<b>Temps de reponse</b> : duree totale (en secondes) entre l'envoi de la question et la "
        "reception de la reponse du LLM, incluant l'encodage de la question, la recherche "
        "vectorielle et l'appel a l'API Groq.",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(Paragraph(
        "Les resultats detailles (une ligne par question) sont sauvegardes dans "
        "<font face='Courier'>evaluation/results.csv</font>, puis transformes en graphiques par "
        "<font face='Courier'>visualizations/plot_results.py</font> (section suivante). La meme "
        "logique d'evaluation est disponible de maniere interactive dans l'onglet \"Evaluation\" de "
        "l'interface Streamlit, qui permet de charger n'importe quel autre jeu de questions.",
        styles["Body"]))
    flow.append(PageBreak())
    return flow


def section_results(styles):
    flow = [Paragraph("6. Resultats et visualisations", styles["H1"])]

    df = pd.read_csv(RESULTS_FILE)
    means = df[[f"precision@{TOP_K}", f"recall@{TOP_K}", "faithfulness", "response_time_s"]].mean()

    flow.append(Paragraph(
        f"L'evaluation a ete executee sur les {len(df)} questions du jeu de test. Le tableau "
        "ci-dessous resume les moyennes obtenues :",
        styles["Body"]))

    rows = [
        ["Metrique", "Valeur moyenne"],
        [f"Precision@{TOP_K}", f"{means[f'precision@{TOP_K}']:.3f}"],
        [f"Recall@{TOP_K}", f"{means[f'recall@{TOP_K}']:.3f}"],
        ["Fidelite (faithfulness)", f"{means['faithfulness']:.3f}"],
        ["Temps de reponse (s)", f"{means['response_time_s']:.3f}"],
    ]
    table = Table(rows, colWidths=[8 * cm, 4 * cm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3a6ea5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 0.4 * cm))

    flow.append(Image(os.path.join(VIZ_DIR, "average_metrics.png"), width=11 * cm, height=9.2 * cm))
    flow.append(Paragraph("Figure 2 - Moyennes des metriques d'evaluation sur les 34 questions.", styles["Caption"]))
    flow.append(PageBreak())

    flow.append(Image(os.path.join(VIZ_DIR, "precision_recall_per_question.png"), width=17 * cm, height=7.1 * cm))
    flow.append(Paragraph(
        f"Figure 3 - Precision@{TOP_K} et Recall@{TOP_K} pour chacune des 34 questions. "
        "La grande majorite des questions atteignent une precision et un recall de 1.0 ; quelques "
        "questions (par exemple celles couvrant un sujet aussi traite par un document voisin, comme "
        "lemmatisation/stemmatisation) presentent une precision ou un recall plus bas car les chunks "
        "recuperes proviennent en partie d'un document lie mais different de celui attendu.",
        styles["Caption"]))

    flow.append(Image(os.path.join(VIZ_DIR, "faithfulness_distribution.png"), width=11.5 * cm, height=7.2 * cm))
    flow.append(Paragraph(
        "Figure 4 - Distribution du score de fidelite (recouvrement lexical entre la reponse "
        "generee et le contexte recupere). La plupart des reponses ont un score superieur a 0.8, "
        "ce qui indique qu'elles restent bien ancrees dans les passages recuperes.",
        styles["Caption"]))
    flow.append(PageBreak())

    flow.append(Image(os.path.join(VIZ_DIR, "response_time_distribution.png"), width=11.5 * cm, height=7.2 * cm))
    flow.append(Paragraph(
        "Figure 5 - Distribution des temps de reponse. La plupart des reponses sont generees en "
        "moins d'une seconde grace a la rapidite de l'API Groq ; quelques pics au-dela de 2 secondes "
        "correspondent aux questions necessitant des reponses plus longues.",
        styles["Caption"]))

    flow.append(Image(os.path.join(VIZ_DIR, "time_vs_faithfulness.png"), width=11.5 * cm, height=7.2 * cm))
    flow.append(Paragraph(
        "Figure 6 - Relation entre le temps de reponse, la fidelite et la precision. Aucune "
        "correlation forte n'est observee : les reponses plus longues a generer ne sont ni "
        "systematiquement plus fideles, ni moins precises.",
        styles["Caption"]))
    flow.append(PageBreak())
    return flow


def section_discussion(styles):
    flow = [Paragraph("7. Discussion", styles["H1"])]

    flow.append(Paragraph("7.1 Points forts", styles["H2"]))
    bullets = [
        "Le pipeline atteint une <b>precision@3 moyenne de 0.92</b> et un <b>recall@3 moyen de "
        "0.99</b> sur le jeu de 34 questions, ce qui montre que l'index vectoriel base sur "
        "<font face='Courier'>all-MiniLM-L6-v2</font> retrouve quasi systematiquement le bon "
        "document source, meme pour des formulations differentes de celles du texte original.",
        "La <b>fidelite moyenne de 0.89</b> indique que les reponses generees par le LLM restent "
        "tres largement ancrees dans le contexte recupere, grace au prompt qui impose de repondre "
        "uniquement a partir du contexte fourni.",
        "Le <b>temps de reponse moyen de 0.81 seconde</b> rend l'assistant utilisable de maniere "
        "interactive, l'API Groq etant tres rapide meme pour un modele de 70 milliards de "
        "parametres.",
        "L'interface Streamlit centralise les trois besoins du projet (chat, evaluation, gestion "
        "du corpus), ce qui facilite a la fois la demonstration et la maintenance du systeme.",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))

    flow.append(Paragraph("7.2 Limites observees", styles["H2"]))
    bullets = [
        "Quelques questions obtiennent une precision plus faible (0.33 a 0.67) lorsque le sujet "
        "recoupe un document voisin (ex. \"FastText\" vs \"Word2vec\", \"sentence embedding\" vs "
        "\"sentence-transformers\"/\"BERT\") : les chunks les plus proches semantiquement "
        "n'appartiennent pas toujours au document considere comme \"attendu\", bien que le contenu "
        "reste pertinent.",
        "La metrique de fidelite utilisee (recouvrement lexical) est une approximation simple : elle "
        "ne detecte pas les reformulations correctes (synonymes) ni les hallucinations subtiles qui "
        "reutiliseraient le vocabulaire du contexte de maniere incorrecte. Une evaluation par un "
        "second LLM (LLM-as-judge) serait plus precise mais plus couteuse et plus lente.",
        "Le corpus etant issu de Wikipedia (anglais), le systeme repond en anglais et est limite "
        "aux connaissances disponibles au moment du telechargement des articles ; il ne reflete pas "
        "les developpements les plus recents du domaine.",
        "La recherche vectorielle ne tient pas compte du recoupement entre articles tres proches "
        "(par exemple les familles de modeles de langage), ce qui peut amener le retriever a "
        "melanger des chunks issus de documents differents mais traitant du meme sous-theme.",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(PageBreak())
    return flow


def section_conclusion(styles):
    flow = [Paragraph("8. Conclusion et perspectives", styles["H1"])]
    flow.append(Paragraph(
        "Ce projet a permis de mettre en oeuvre l'ensemble de la chaine d'un systeme RAG, depuis la "
        "collecte et le nettoyage d'un corpus jusqu'a une interface utilisateur complete et une "
        "evaluation quantitative outillee. Les resultats obtenus (precision@3 = 0.92, recall@3 = "
        "0.99, fidelite = 0.89, temps de reponse moyen = 0.81 s) montrent que le systeme repond de "
        "maniere pertinente, rapide et bien ancree dans les sources sur l'ensemble du corpus de "
        "concepts NLP/ML choisi.",
        styles["Body"]))
    flow.append(Paragraph(
        "Plusieurs pistes d'amelioration ont ete identifiees pour des travaux futurs :",
        styles["Body"]))
    bullets = [
        "Remplacer la metrique de fidelite par une approche \"LLM-as-judge\" pour detecter des "
        "hallucinations plus subtiles ;",
        "Etendre le corpus avec des sources plus recentes ou multilingues ;",
        "Ajouter un re-ranking des chunks recuperes (par exemple avec un modele cross-encoder) "
        "pour ameliorer la precision sur les sujets qui se recoupent ;",
        "Permettre la citation des sources directement dans le texte de la reponse generee.",
    ]
    for b in bullets:
        flow.append(Paragraph(f"- {b}", styles["BulletItem"]))
    flow.append(PageBreak())
    return flow


def section_references(styles):
    flow = [Paragraph("9. References", styles["H1"])]
    refs = [
        "Wikipedia, articles divers sur les concepts de NLP et de machine learning "
        "(en.wikipedia.org), consultes en juin 2026.",
        "Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese "
        "BERT-Networks. EMNLP. (modele all-MiniLM-L6-v2)",
        "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.",
        "Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP "
        "Tasks. NeurIPS.",
        "Documentation ChromaDB - https://docs.trychroma.com/",
        "Documentation Groq API - https://console.groq.com/docs/",
        "Documentation Streamlit - https://docs.streamlit.io/",
        "Documentation sentence-transformers - https://www.sbert.net/",
    ]
    for r in refs:
        flow.append(Paragraph(f"- {r}", styles["BulletItem"]))
    return flow


def main():
    styles = build_styles()
    doc = SimpleDocTemplate(
        OUT_FILE, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    flow = []
    flow += title_page(styles)
    flow += table_of_contents(styles)
    flow += section_introduction(styles)
    flow += section_architecture(styles)
    flow += section_data(styles)
    flow += section_implementation(styles)
    flow += section_evaluation_methodology(styles)
    flow += section_results(styles)
    flow += section_discussion(styles)
    flow += section_conclusion(styles)
    flow += section_references(styles)
    doc.build(flow)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
