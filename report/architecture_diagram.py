"""Generates report/architecture.png: a block diagram of the RAG pipeline used in the report."""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_FILE = os.path.join(os.path.dirname(__file__), "architecture.png")

BOX_STYLE = dict(boxstyle="round,pad=0.05", facecolor="#dbe7f6", edgecolor="#3a6ea5", linewidth=1.5)
LLM_STYLE = dict(boxstyle="round,pad=0.05", facecolor="#fde2c8", edgecolor="#c97a1a", linewidth=1.5)

W, H, GAP = 1.8, 1.0, 0.4


def box(ax, x, y, text, style=BOX_STYLE):
    ax.add_patch(FancyBboxPatch((x, y), W, H, **style))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=10)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, color="#333333"))


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    cols = [0.3, 2.5, 4.7, 6.9]  # x position of each column
    row1, row2, row3 = 5.0, 2.7, 0.4  # y position of each row

    # Offline indexing pipeline (top row)
    ax.text(5, row1 + H + 0.35, "Indexation (hors ligne)", ha="center", fontsize=12, fontweight="bold")
    box(ax, cols[0], row1, "Corpus\nWikipedia\n(36 articles)")
    box(ax, cols[1], row1, "Nettoyage\n+ chunking\n(512 / 50)")
    box(ax, cols[2], row1, "Embeddings\nall-MiniLM-L6-v2")
    box(ax, cols[3], row1, "Index ChromaDB\n(persistant)")
    for x in cols[:3]:
        arrow(ax, x + W, row1 + H / 2, x + GAP + W, row1 + H / 2)

    # Online query pipeline (middle row)
    ax.text(5, row2 + H + 0.35, "Reponse a une question (en ligne)", ha="center", fontsize=12, fontweight="bold")
    box(ax, cols[0], row2, "Question\nutilisateur")
    box(ax, cols[1], row2, "Embedding\nde la question")
    box(ax, cols[2], row2, "Recherche top-k\ndans ChromaDB")
    box(ax, cols[3], row2, "Prompt\n(contexte + question)")
    for x in cols[:3]:
        arrow(ax, x + W, row2 + H / 2, x + GAP + W, row2 + H / 2)

    # Index feeds the retrieval step, and the prompt is sent to the LLM
    arrow(ax, cols[3] + W / 2, row1, cols[3] + W / 2, row2 + H)
    arrow(ax, cols[3] + W / 2, row2, cols[3] + W / 2, row3 + H)

    # Bottom row: LLM call and final answer
    box(ax, cols[3], row3, "LLM Groq\n(Llama 3.3 70B)", style=LLM_STYLE)
    box(ax, cols[1], row3, "Reponse +\nsources citees")
    box(ax, cols[0], row3, "Interface\nStreamlit")
    arrow(ax, cols[3], row3 + H / 2, cols[1] + W, row3 + H / 2)
    arrow(ax, cols[1], row3 + H / 2, cols[0] + W, row3 + H / 2)
    arrow(ax, cols[0] + W / 2, row3 + H, cols[0] + W / 2, row2)

    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
