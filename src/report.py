"""Build all tables and figures from the 'Results to Report' section.

Reads results/{en,hn}_*.json and writes:
  results/tables/*.csv
  results/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

CATS = C.CATEGORIES
TBL = C.RES_TABLES
FIG = C.RES_FIGURES


def _load(name):
    p = C.path(C.EXP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def loo_table():
    for prefix, pool in [("en", "Reddit (POOL-EN)"), ("hn", "Hacker News")]:
        d = _load(f"{prefix}_loo.json")
        if not d:
            continue
        df = pd.DataFrame(d["rows"])
        df.to_csv(C.path(TBL, f"{prefix}_leave_one_out.csv"), index=False)
        print(f"\n[LOO {pool}] n={d['n']} model={d['model']} "
              f"recall_full@90={d['recall_full@90']:.3f}")
        print(df.to_string(index=False))


def sis_side_by_side():
    en = _load("en_loo.json")
    hn = _load("hn_loo.json")
    rows = []
    for code in CATS + ["STYLE"]:
        r = {"category": code}
        for prefix, d in [("Reddit", en), ("HN", hn)]:
            if d:
                m = {x["condition"]: x for x in d["rows"]}.get(f"NO_{code}", {})
                r[f"SIS_{prefix}"] = m.get("SIS_rank1")          # robust SIS for the bar chart
                r[f"SIS90_{prefix}"] = m.get("SIS_recall90")     # recall@90 SIS
                r[f"rank1_{prefix}"] = m.get("rank1_acc")
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(C.path(TBL, "sis_reddit_vs_hn.csv"), index=False)
    print("\n[SIS Reddit vs HN]")
    print(df.to_string(index=False))
    return df


def sis_barchart(df):
    labels = df["category"].tolist()
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "SIS_Reddit" in df:
        ax.bar(x - w/2, df["SIS_Reddit"].fillna(0), w, label="Reddit", color="#3b6fb6")
    if "SIS_HN" in df:
        ax.bar(x + w/2, df["SIS_HN"].fillna(0), w, label="Hacker News", color="#d1782f")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Signal Importance Score (rank-1, %)")
    ax.set_title("SIS by signal category: Reddit vs Hacker News")
    ax.axhline(0, color="k", lw=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(C.path(FIG, "sis_reddit_vs_hn.png"), dpi=150)
    print(f"  -> {FIG}/sis_reddit_vs_hn.png")


def additive_curve():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for prefix, pool, col in [("en", "Reddit", "#3b6fb6"), ("hn", "HN", "#d1782f")]:
        d = _load(f"{prefix}_additive.json")
        if not d:
            continue
        df = pd.DataFrame(d["rows"])
        df.to_csv(C.path(TBL, f"{prefix}_additive.csv"), index=False)
        ax.plot(df["step"], df["recall@90"], "-o", color=col, label=f"{pool} (MSSS: {d['MSSS']})")
        ax.set_xticks(df["step"])
        ax.set_xticklabels(df["profile"], rotation=30, ha="right", fontsize=7)
        print(f"\n[Additive {pool}] MSSS={d['MSSS']}")
        print(df.to_string(index=False))
    ax.set_ylabel("Recall@90% precision")
    ax.set_title("Additive ablation: accumulated recall")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.path(FIG, "additive_recall_curve.png"), dpi=150)
    print(f"  -> {FIG}/additive_recall_curve.png")


def interaction_table():
    for prefix, pool in [("en", "Reddit"), ("hn", "HN")]:
        d = _load(f"{prefix}_interaction.json")
        if not d:
            continue
        df = pd.DataFrame(d["rows"])
        df.to_csv(C.path(TBL, f"{prefix}_interaction.csv"), index=False)
        print(f"\n[Interaction {pool}] top3={d['top3']}")
        print(df.to_string(index=False))


def style_table():
    for prefix, pool in [("en", "Reddit"), ("hn", "HN")]:
        d = _load(f"{prefix}_style_ttr.json")
        if not d:
            continue
        df = pd.DataFrame(d["rows"])
        df.to_csv(C.path(TBL, f"{prefix}_style_by_ttr.csv"), index=False)
        print(f"\n[STYLE-only by TTR {pool}]")
        print(df.to_string(index=False))


def main():
    os.makedirs(C.path(TBL), exist_ok=True)
    os.makedirs(C.path(FIG), exist_ok=True)
    loo_table()
    df = sis_side_by_side()
    sis_barchart(df)
    additive_curve()
    interaction_table()
    style_table()
    print("\nAll tables -> results/tables/, figures -> results/figures/")


if __name__ == "__main__":
    main()
