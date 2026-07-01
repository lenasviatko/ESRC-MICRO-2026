# ESRC Ablation Pipeline — Signal Importance for LLM Deanonymization

A full ESRC pipeline (Extract → Search → Reason → Calibrate) plus a
Leave-One-Out / Additive / Interaction ablation and a Hacker-News replication,
measuring the **Signal Importance Score (SIS)** of each personal-signal category.

## Layout

```
src/
  common.py            data conventions + centralized paths, OpenAI client, loaders
  classify_tags.py     Step 4 — classify every g2 tag into GEO/PROF/DEMO/INT/VAL
  style_features.py    Step 5 — 5 stylometric features, computed per side (query/candidate)
  esrc.py              Search (mpnet + exact cosine, top-15), Reason (LLM), Calibrate
                       (PR curve, Recall@90/99), metrics (SIS, Cohen's d, t-test, bootstrap)
  experiments.py       Steps 6-11 driver (rich filter, LOO, additive, interaction, style)
  qualitative.py       success/failure case dump
  report.py            tables + figures
data/
  raw/                 POOL-EN, POOL-HN, POOL-XLING-* raw *_query/candidate.jsonl
  summaries/           query/candidate_profiles_g2_812.json  (Extract output)
  tag_classifications/ tag_categories_full.json  (tag -> category)
  style_features/      style_{query,candidate}_POOL-*.json
  splits/              valid_users_812.json, rich_users_*.json
experiments/           {en,hn}_{loo,additive,interaction,style_ttr,qualitative}.json
results/
  tables/  *.csv       figures/  *.png       cache/  reason_*.json (cached decisions)
notebooks/             analysis.ipynb (loads results, renders tables/figures)
```

All paths are centralized in `src/common.py` (RAW, SUMMARIES, TAGCLS, STYLEDIR,
SPLITS, EXP, RES_TABLES, RES_FIGURES, CACHE_DIR), so the layout can be changed in
one place. Inputs under `data/` and the cached decisions under `results/cache/`
are already present; never delete `results/cache/` (it holds paid LLM calls).

## Run

The whole study (Reddit + Hacker News, all experiments, tables and figures):

```bash
./run_study.sh                 # development run on gpt-4o-mini
MODEL=gpt-4o ./run_study.sh    # final reported run for Exp 1/2 + HN
```

Individual stages (run from inside `src/`):

```bash
python classify_tags.py        # tag -> category
python style_features.py       # writing-style features (local, free)
python experiments.py rich     # rich-profile filter report
python experiments.py all EN   # Leave-One-Out + Additive + Interaction + style (Reddit)
python experiments.py loo HN   # Hacker News replication
python report.py               # tables + figures
```

Knobs: `MODEL=gpt-4o` for the final reported runs; `MAX_USERS=N` for a quick
subset; `MIN_CATS=5` (default) keeps users with all five content categories.
Reason answers are cached in `results/cache/`, so re-runs are free and an
interrupted run resumes from where it stopped.
