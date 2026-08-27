"""S1: style-only linkage attack — classical authorship attribution, no LLM.

Five feature groups per user per period (function words, char 3-4-grams,
punctuation rates, sentence/word length stats, POS distribution), each
L2-normalised to unit length before concatenation, then the same cosine
top-15 search on the same rich users and candidate pools as the LLM run.
Profiles are truncated to the first 100 comments per period; a test fails
if a user's two periods ever share a feature vector.

Usage:  python s1_style_attack.py [EN|HN|all]
Writes: experiments/s1_style_{en,hn}.json  (summary + per-user records)
        results/tables/s1_style_only.csv
"""
import json
import re
import string
import sys

import numpy as np

import common as C
import esrc
import experiments as E

MAX_COMMENTS = 100
TOP_K = esrc.TOP_K
POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

_nlp = None


# Data loading
def nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm",
                          disable=["parser", "ner", "lemmatizer", "attribute_ruler"])
        _nlp.max_length = 3_000_000
    return _nlp


def comments_of(pool, uid, side):
    out = []
    with open(C.path(C.raw_pool(pool), f"{uid}_{side}.jsonl")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                b = json.loads(line).get("b", "")
            except Exception:
                continue
            if b and b.strip():
                out.append(b)
            if len(out) >= MAX_COMMENTS:
                break
    return out


# Feature groups
def l2(x):
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def function_word_freqs(words, vocab):
    n = max(1, len(words))
    counts = {}
    for w in words:
        w = w.lower()
        if w in vocab:
            counts[w] = counts.get(w, 0) + 1
    return np.array([counts.get(w, 0) / n for w in vocab], dtype="float32")


def punctuation_rates(text):
    n = max(1, len(text))
    return np.array([text.count(p) / n * 1000 for p in string.punctuation],
                    dtype="float32")


def length_stats(text, words):
    sents = [s for s in SENT_RE.findall(text) if WORD_RE.search(s)]
    slens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    wlens = [len(w) for w in words] or [0]
    return np.array([np.mean(slens), np.std(slens), np.mean(wlens), np.std(wlens)],
                    dtype="float32")


def pos_distribution(docs):
    counts = dict.fromkeys(POS_TAGS, 0)
    total = 0
    for d in docs:
        for tok in d:
            if tok.pos_ in counts:
                counts[tok.pos_] += 1
                total += 1
    total = max(1, total)
    return np.array([counts[t] / total for t in POS_TAGS], dtype="float32")


# Vectors and search
def build_vectors(pool, uids):
    """uid -> unit-norm concatenated style vector, per period."""
    from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                                 TfidfVectorizer)
    fw_vocab = sorted(ENGLISH_STOP_WORDS)

    texts = {}   # (side, uid) -> full text of the truncated profile
    for side in ("query", "candidate"):
        for uid in uids:
            texts[(side, uid)] = " ".join(comments_of(pool, uid, side))

    # n-gram vocabulary comes from the candidate period only
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                            max_features=5000)
    tfidf.fit([texts[("candidate", u)] for u in uids])

    print(f"  [{pool}] POS-tagging {2 * len(uids)} profiles ...", flush=True)
    pos_docs = {}
    keys = list(texts.keys())
    for key, doc in zip(keys, nlp().pipe((texts[k] for k in keys), batch_size=8)):
        pos_docs[key] = doc

    vecs = {"query": {}, "candidate": {}}
    for side in ("query", "candidate"):
        ngram = tfidf.transform([texts[(side, u)] for u in uids]).toarray()
        for i, uid in enumerate(uids):
            text = texts[(side, uid)]
            words = WORD_RE.findall(text)
            groups = [
                l2(function_word_freqs(words, fw_vocab)),
                l2(ngram[i]),
                l2(punctuation_rates(text)),
                l2(length_stats(text, words)),
                l2(pos_distribution([pos_docs[(side, uid)]])),
            ]
            vecs[side][uid] = l2(np.concatenate(groups))
    return vecs


def assert_no_period_leak(vecs, uids):
    """Fail hard if any user's query and candidate vectors coincide."""
    for uid in uids:
        if np.allclose(vecs["query"][uid], vecs["candidate"][uid], atol=1e-6):
            raise AssertionError(
                f"period leak: query and candidate share a feature vector for {uid}")
    print(f"  period-leak test passed for {len(uids)} users", flush=True)


def run_pool(pool, prefix):
    """Run the style attack on one pool; writes the per-user records."""
    q, c, cat = E.load_all()
    uids, _, total = E.select_users(pool, q, cat)
    print(f"\n=== S1 style-only attack [{pool}] n={len(uids)} ===", flush=True)

    vecs = build_vectors(pool, uids)
    assert_no_period_leak(vecs, uids)

    X = np.vstack([vecs["candidate"][u] for u in uids])
    Q = np.vstack([vecs["query"][u] for u in uids])
    sims = Q @ X.T
    order = np.argsort(-sims, axis=1)

    results = []
    for i, uid in enumerate(uids):
        ranked = [uids[j] for j in order[i]]
        rank_true = ranked.index(uid) + 1
        s = sims[i][order[i]]
        results.append({
            "uid": uid,
            "pred": ranked[0],
            "correct": ranked[0] == uid,
            "confidence": float(s[0]),
            "sim": float(s[0]),
            "sim_top2": float(s[1]),
            "gap_top1_top2": float(s[0] - s[1]),
            "rank_true": rank_true,
            "retrieved": rank_true <= TOP_K,
            "top15": ranked[:TOP_K],
        })

    acc = esrc.rank1_accuracy(results)
    a_lo, a_hi = esrc.bootstrap_ci(results, esrc.rank1_accuracy, n_boot=10_000)
    rec15 = float(np.mean([r["retrieved"] for r in results]))
    r_lo, r_hi = esrc.bootstrap_ci(
        results, lambda r: float(np.mean([x["retrieved"] for x in r])), n_boot=10_000)

    summary = {
        "pool": pool, "n": len(uids), "model": "none (classical stylometry)",
        "max_comments": MAX_COMMENTS, "top_k": TOP_K,
        "feature_dims": int(X.shape[1]),
        "rank1_acc": round(acc, 4), "rank1_CI": [round(a_lo, 3), round(a_hi, 3)],
        "recall@15": round(rec15, 4), "recall15_CI": [round(r_lo, 3), round(r_hi, 3)],
        "period_leak_test": "passed",
    }
    C.save_json({**summary, "results": results}, f"{C.EXP}/s1_style_{prefix}.json")
    print(f"  rank1 = {acc:.4f} [{a_lo:.3f}, {a_hi:.3f}]   "
          f"recall@15 = {rec15:.4f} [{r_lo:.3f}, {r_hi:.3f}]   dims={X.shape[1]}",
          flush=True)
    return summary


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = [("POOL-EN", "en"), ("POOL-HN", "hn")] if arg == "all" else \
        [({"EN": "POOL-EN", "HN": "POOL-HN"}[arg], arg.lower())]
    rows = [run_pool(pool, prefix) for pool, prefix in todo]

    import csv
    out = C.path(C.RES_TABLES, "s1_style_only.csv")
    cols = ["pool", "n", "rank1_acc", "rank1_CI", "recall@15", "recall15_CI",
            "feature_dims", "max_comments"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
