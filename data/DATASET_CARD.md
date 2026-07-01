# ESRC-Micro-2026 — Methodology

**Author:** Tatiana Petrova, University of Luxembourg
**Version:** 1.0
**Anchor paper:** Lermen, Paleka, Swanson, Aerni, Carlini, Tramèr (2026). *Large-scale online deanonymization with LLMs.* arXiv:2602.16800.

This document specifies, in a form intended to be reproduced verbatim in
papers and dataset cards, the construction of the **ESRC-Micro-2026**
dataset — a compact reproducibility-oriented subset for evaluating
LLM-based deanonymization attacks at BSc-thesis scale.


## 1. Motivation and design goal

The ESRC framework (Lermen et al. 2026, §3) requires per-user collections
of unstructured online text on the order of hundreds to thousands of
comments per user, and pools of thousands to millions of candidate users.
Reproducing such experiments from raw Pushshift dumps is computationally
expensive (≥ 100 GB of compressed input, ≥ 100 EUR of frontier-LLM API
spend per condition), placing reproduction outside the reach of typical
BSc-thesis projects.

ESRC-Micro-2026 is a deliberately small subset — sized for ≈ ±5 percentage
point Wilson 95% confidence intervals on Recall@90%Precision under
expected effect sizes — designed so that the full ESRC pipeline can be
run end-to-end in under one hour at < 30 EUR per student, while
preserving the filter spec, temporal-split logic, and pool composition
of the original work.


## 2. Source data

| Source | Provenance | License |
|---|---|---|
| **Reddit** | Pushshift monthly dumps, distributed via Academic Torrents `30dee5f0406da7a353aff6a8caa2d54fd01f2ca1` ("Reddit comments and submissions 2005-06 to 2025-06") | CC BY 4.0 |
| **Hacker News** | BigQuery public dataset `bigquery-public-data.hacker_news.full` | Public domain (Y Combinator) |
| **Stack Exchange** | Internet Archive Stack Exchange Data Dump (snapshot 2024-04) | CC BY-SA 4.0 |

Eleven monthly Reddit dumps were used: `RC_2015-06`, `RC_2016-01`,
`RC_2016-06`, `RC_2017-01`, `RC_2017-06`, `RC_2018-01`, `RC_2018-06`,
`RC_2018-12`, `RC_2019-06`, `RC_2020-01`, `RC_2020-06`. Months were chosen
to cover the union of query/candidate windows required by four BSc topics
(deanonymization defence, minimum-text, multi-language, signal-ablation)
and to span at least four years to permit a one-year gap between query
and candidate halves of any user's history.


## 3. User-level filter specification

Following Lermen et al. (2026, App. G.1), a Reddit user is retained iff
all of the following hold:

| Criterion | Threshold | Rationale |
|---|---|---|
| Lifetime comments | ≥ 1 000 | At least 500 on each side after the temporal split below |
| Activity span (days) | ≥ 1 460 (≈ 4 years) | Permits a valid 1-year gap between query and candidate windows |
| Mean comments per active day | ≤ 24 | Excludes bots and crawler accounts |
| Unique subreddits | ≥ 5 | Excludes mono-topic accounts; ensures profile diversity |
| Username pattern | does **not** match `/(bot\|gpt\|mod)$/i` | Excludes self-identifying automated accounts |
| Account state | not `[deleted]`, not `AutoModerator` | Excludes deleted and platform accounts |
| Language | ≥ 80 % comments classified as the target language by `fasttext lid.176` | Ensures monolingual pools for the language-specific experiments |

Identical activity, anti-bot, and minimum-history criteria are applied
to Hacker News authors via BigQuery, omitting the subreddit-diversity
criterion (Hacker News has no subreddits) and the language filter (HN
is overwhelmingly English).


## 4. Temporal split (closed-world ground truth)

Following Lermen et al. (2026, §6), each retained user's chronologically
ordered comment history is split as follows:

  T*       := timestamp at the median index of the user's comments
  query    := { c : t(c) < T* − 182 days }
  candidate := { c : t(c) > T* + 182 days }
  discarded := { c : T* − 182 days ≤ t(c) ≤ T* + 182 days }

The 365-day discard window prevents trivial matching via short-lived
context (current events, ongoing personal discussions). A user is
dropped if either side has fewer than 500 comments after the split.

Each side is then capped at 500 comments: when there are more, a
deterministic pseudo-random sample (`numpy.random.default_rng(seed=42)`,
nested per-user with key `f"{SEED}:{author_hash}"`) is drawn, then the
chosen comments are re-sorted chronologically. The cap matches the
largest truncation level (T7 = 500) used in the BSc text-volume
experiment (Petrov, Topic 3).


## 5. Pool sampling

From the qualifying-user pool of each language, the following counts
are sampled deterministically (`numpy.random.default_rng(seed=42)`,
without replacement):

| Pool             | matched users | distractor users | Source         | Language |
|------------------|--------------:|-----------------:|----------------|----------|
| POOL-EN          | 500           | 500              | Reddit         | en       |
| POOL-FR          | 200           | 200              | Reddit         | fr       |
| POOL-DE          | 200           | 200              | Reddit         | de       |
| POOL-HN          | 300           | 300              | Hacker News    | en       |
| POOL-XLING-ES    | 200           | 0                | Stack Exchange | en, es   |
| POOL-XLING-PT    | 200           | 0                | Stack Exchange | en, pt   |

**Matched** users contribute both `_query.jsonl` and `_candidate.jsonl`.
**Distractor** users contribute only `_candidate.jsonl` and have no
matching query — they exist to enforce a precision penalty on attacks
that always answer rather than abstain. The 1:1 distractor ratio is
chosen to be small enough to keep the candidate-side LLM cost
manageable while large enough to make Recall@90 % Precision a
meaningful metric (a naïve always-guess attack achieves ≤ 50 %
precision on a 1:1 pool).


## 6. Cross-lingual ground truth (Stack Exchange)

For POOL-XLING-{ES, PT}, the ground truth is constructed by joining
on the **AccountId** field of the Stack Exchange `Users.xml` table, which
is global across all SE sites: a user with AccountId = N on
`stackoverflow.com` and AccountId = N on `es.stackoverflow.com` is
verifiably the same person. The query side is composed of posts on
the foreign-language SE site; the candidate side is the posts of the
same AccountId on `stackoverflow.com` (English). Users with fewer than
10 posts on either side are excluded.

### Methodology amendment (vs. Kohut Topic 2 PDF)

The Kohut methodology specifies `fr.stackoverflow.com` as the
French-language source for cross-lingual ground truth (Dataset C). At
the time of dataset construction (May 2026), this file does not exist
in the Internet Archive Stack Exchange Data Dump: Stack Overflow has
no French-language regional clone, only Spanish (`es.stackoverflow.com`),
Portuguese (`pt.stackoverflow.com`), Russian, and Japanese. The cross-
lingual experiment is therefore amended to use **EN ↔ ES** and **EN ↔ PT**
as the two language pairs. The methodology design (cross-lingual
matched-account joins via `AccountId`) is preserved verbatim — only the
foreign-language site is amended.


## 7. Pseudonymization

Original platform usernames are replaced with `user_<8 hex>` derived
from SHA-256(`<bucket-blake2-hash> ":" <salt>`)`[:8]`, where the salt
is held privately by the dataset curator. The bucket-blake2-hash is
internal to the build pipeline (used to consistently shard authors
across the 11 monthly dumps without storing the original username) and
is never released. The salt is generated at build time
(`secrets.token_hex(16)`) and stored in `data/work/SALT_DO_NOT_SHARE.txt`
on the build machine, never transferred to the released dataset. As a
result, the released `user_<...>` IDs are not reversible to original
usernames without access to the salt, even by an adversary in
possession of the original Pushshift dumps.


## 8. Sample size justification

Pool sizes were chosen to provide Wilson 95% confidence intervals of
width ≤ 5 percentage points on Recall@90% Precision at expected
effect sizes:

| Pool size n | CI half-width at p = 0.30 | CI half-width at p = 0.50 |
|---:|---:|---:|
| 500 | ± 4.0 pp | ± 4.4 pp |
| 300 | ± 5.2 pp | ± 5.7 pp |
| 200 | ± 6.4 pp | ± 7.0 pp |

These intervals are sufficient to detect the within-condition effects
expected in the four BSc projects (anonymization-defence reductions of
20–40 pp; truncation-level changes of 10–25 pp; language gap of 15–30
pp; per-signal ablations of 5–15 pp). Absolute recall numbers must
not be compared directly to Lermen et al. (2026)'s 10 000–100 000-user
experiments, where the relevant denominator is the full pool size.


## 9. Determinism and reproducibility

The build is end-to-end deterministic:

- All sampling uses `numpy.random.default_rng(seed=42)`, with nested
  per-user RNGs keyed by `f"{SEED}:{author_hash}"` so that ordering
  changes during sharding do not affect which comments are sampled
  during cap-and-sort.
- Filter thresholds, temporal-split parameters, and pool sizes are
  centralized in `scripts/config.py`.
- The build pipeline is invoked as
  `01_index_reddit.py` → `02_select_users.py` → `03_temporal_split.py` →
  `04_sample_pools.py` → `05_hackernews.py` → `06_stackexchange.py` →
  `07_package.py`. Each step writes its outputs to `data/work/` (intermediate)
  or `data/output/` (final), with `.DONE` markers enabling resume after
  an interrupted run.
- The build host writes a `preprocessing_log.json` containing the
  resolved values of all parameters, the host platform, and the build
  completion timestamp.

Given the same source data (the eleven Pushshift monthly dumps, the
April 2024 Stack Exchange snapshot, and an HN BigQuery snapshot of
matching date), and the same `config.py` and salt, the build is
bit-identical.


## 10. Ethical statement

[TO BE FILLED IN] This study was approved by the University of
Luxembourg Ethics Review Panel under reference [TBC]. The dataset is
released under research-use terms compatible with the source-data
licences (CC BY 4.0 for Reddit, public domain for Hacker News,
CC BY-SA 4.0 for Stack Exchange). Original usernames are pseudonymized
as described in §7; the salt-to-username mapping is held by the
curator and not released. Users may request removal of any record
referencing them by contacting tatiana.petrova@uni.lu; verification
is performed via the curator-held salt without disclosing it.


## 11. Intended use and out-of-scope use

**Intended:** comparative evaluation of ESRC-style deanonymization
pipelines under controlled ablation conditions
(anonymization defences, text volume, language, signal-category
ablation). Designed for instructional and reproducibility settings
where running the full Lermen et al. (2026) protocol is computationally
infeasible.

**Out-of-scope:** real-world re-identification of any individual;
training of attack tools for deployment; any analysis whose goal
includes deanonymizing the included users. The dataset is research
infrastructure, not a target.


## 12. Known limitations

1. Sample sizes are deliberately small; absolute recall numbers should
   not be compared directly to large-scale evaluations.
2. Reddit temporal coverage 2015–2020 reflects Pushshift dump
   availability, not contemporary user behaviour.
3. Cross-lingual coverage is limited to the language pairs present in
   the Stack Exchange dumps (no `fr.stackoverflow.com`).
4. Language identification uses `fasttext lid.176`, which can
   misclassify very short comments (< 20 words).
5. The build pipeline assumes the unmodified directory layout of the
   Pushshift Academic Torrents distribution
   (`reddit/comments/RC_YYYY-MM.zst`).


## References

[1] Lermen, S., Paleka, D., Swanson, J., Aerni, M., Carlini, N.,
    Tramèr, F. (2026). *Large-scale online deanonymization with LLMs.*
    arXiv:2602.16800.

[2] De Montjoye, Y.-A., Hidalgo, C. A., Verleysen, M., Blondel, V. D.
    (2013). *Unique in the crowd: The privacy bounds of human mobility.*
    Scientific Reports, 3, 1376.
