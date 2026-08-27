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

## Study 2 — style vs. content (S1–S3)

A self-contained follow-up study: classical authorship attribution vs. the
LLM pipeline, on the same users and the same candidate pools (158 rich EN,
99 rich HN). The earlier ablation (BSP4) enters as prior work.

```
src/
  s1_style_attack.py   S1 — style-only attack, no LLM: 5 feature groups
                       (function words, char 3-4-grams, punctuation rates,
                       sentence/word length stats, POS distribution), each
                       L2-normalised, cosine top-15 on the same pools.
                       Profiles truncated to first 100 comments per period;
                       hard test fails on any query/candidate vector leak.
  s2_check_cache.py    verifies every per-user LLM decision is
                       reconstructible from results/cache (no API calls).
  s2_confidence.py     S2 — score separability of the two attacks:
                       Recall@90%Precision (tie-aware) for LLM confidence vs
                       style sim vs top1-top2 gap; LLM reliability table;
                       accuracy per decile bin; McNemar style vs LLM with
                       Bonferroni correction. No API calls.
  s3_masked_ablation.py S3 — masked ablation on 5-section summaries
                       (Extract, cached): a category is ablated by replacing
                       its section with [masked], so every condition has the
                       same shape; LOO + additive with monotonicity check.
                       Pilot-first token measurement, sliding-window TPM
                       throttle, token ledger with a hard stop in code.
experiments/           s1_style_{en,hn}.json (summary + per-user records:
                       pred, rank of true user, sim top1/top2, gap)
                       s2_confidence_{en,hn}.json, s3_masked_en.json
results/tables/        s1_style_only.csv, s2_recall_at_p90.csv,
                       s2_reliability_llm.csv, s2_accuracy_by_bin.csv,
                       s2_mcnemar.csv, s3_loo.csv, s3_additive.csv
```

```bash
cd src && python s1_style_attack.py all       # zero cost, no API key needed
cd src && python s2_check_cache.py            # cache-coverage check for S2
cd src && MODEL=gpt-4o python s2_confidence.py  # against the final gpt-4o run
cd src && python s3_masked_ablation.py        # pilot, extract, LOO + additive
```

The reason cache holds two complete generations per pool (gpt-4o-mini dev
run and the final gpt-4o run); `MODEL=` selects which one S2 reconstructs.
The reported tables use gpt-4o (matches `experiments/{en,hn}_loo.json`).
