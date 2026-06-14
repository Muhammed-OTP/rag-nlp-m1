"""Phase 6: turn evaluation/results.csv into PNG charts saved under visualizations/.

Usage: python -m visualizations.plot_results
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import TOP_K

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results.csv")
OUT_DIR = os.path.dirname(__file__)

sns.set_theme(style="whitegrid")

PRECISION_COL = f"precision@{TOP_K}"
RECALL_COL = f"recall@{TOP_K}"


def plot_precision_recall(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    df[[PRECISION_COL, RECALL_COL]].plot(kind="bar", ax=ax)
    ax.set_xlabel("Question index")
    ax.set_ylabel("Score")
    ax.set_title(f"Precision@{TOP_K} and Recall@{TOP_K} per question")
    ax.set_ylim(0, 1.05)
    ax.legend([f"Precision@{TOP_K}", f"Recall@{TOP_K}"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "precision_recall_per_question.png"), dpi=150)
    plt.close(fig)


def plot_response_time(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["response_time_s"], bins=10, ax=ax, kde=True)
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Number of questions")
    ax.set_title("Response time distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "response_time_distribution.png"), dpi=150)
    plt.close(fig)


def plot_faithfulness(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["faithfulness"], bins=10, ax=ax, kde=True, color="seagreen")
    ax.set_xlabel("Faithfulness (lexical overlap with retrieved context)")
    ax.set_ylabel("Number of questions")
    ax.set_title("Faithfulness distribution")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "faithfulness_distribution.png"), dpi=150)
    plt.close(fig)


def plot_summary(df: pd.DataFrame) -> None:
    means = df[[PRECISION_COL, RECALL_COL, "faithfulness"]].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=means.index, y=means.values, hue=means.index, ax=ax, palette="viridis", legend=False)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Average score")
    ax.set_title("Average evaluation metrics")
    for i, v in enumerate(means.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "average_metrics.png"), dpi=150)
    plt.close(fig)


def plot_time_vs_faithfulness(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="response_time_s", y="faithfulness", hue=PRECISION_COL, palette="coolwarm", ax=ax)
    ax.set_xlabel("Response time (s)")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Faithfulness vs. response time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "time_vs_faithfulness.png"), dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS_FILE)
    plot_precision_recall(df)
    plot_response_time(df)
    plot_faithfulness(df)
    plot_summary(df)
    plot_time_vs_faithfulness(df)
    print(f"Saved 5 charts to {OUT_DIR}")


if __name__ == "__main__":
    main()
