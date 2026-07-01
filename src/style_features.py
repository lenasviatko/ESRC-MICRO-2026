"""Compute the five stylometric features from raw comment text.

Features are computed separately for each user's query period and candidate
period, so cross-period stylometric consistency is what the matcher sees.

Outputs: style_query_<POOL>.json and style_candidate_<POOL>.json (uid -> 5 features).
"""
import json
import os
import re
import sys

import common as C

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+")
PUNCT = ".,!?"
FIRST_PERSON = {"i", "my", "me", "myself"}


def features_from_text(text):
    words = WORD_RE.findall(text)
    n_words = len(words)
    if n_words == 0:
        return None
    n_sent = max(1, len(SENT_RE.findall(text)))
    n_chars = sum(len(w) for w in words)
    uniq = len(set(w.lower() for w in words))
    punct = sum(text.count(p) for p in PUNCT)
    fp = sum(1 for w in words if w.lower() in FIRST_PERSON)
    return {
        "avg_sentence_length": n_words / n_sent,
        "avg_word_length": n_chars / n_words,
        "ttr": uniq / n_words,
        "punctuation_frequency": punct / n_words * 100,
        "first_person_frequency": fp / n_words * 100,
    }


def text_of(pool, fname):
    parts = []
    with open(C.path(C.raw_pool(pool), fname)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts.append(json.loads(line).get("b", ""))
            except Exception:
                pass
    return " ".join(parts)


def build(pool):
    fs = sorted(os.listdir(C.raw_pool(pool)))
    for side, suffix in [("query", "_query.jsonl"), ("candidate", "_candidate.jsonl")]:
        out = {}
        for fn in fs:
            if not fn.endswith(suffix):
                continue
            uid = fn.replace(suffix, "")
            feats = features_from_text(text_of(pool, fn))
            if feats:
                out[uid] = feats
        C.save_json(out, f"{C.STYLEDIR}/style_{side}_{pool}.json")
        print(f"{pool} {side}: {len(out)} users -> style_{side}_{pool}.json")


if __name__ == "__main__":
    pools = sys.argv[1:] or ["POOL-EN", "POOL-HN"]
    for p in pools:
        build(p)
