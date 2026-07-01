"""ESRC pipeline: Search, Reason, Calibrate, and the ablation metrics."""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common as C

TOP_K = 15
EMBED_MODEL = "all-mpnet-base-v2"
_st_model = None
_embed_cache = {}


def style_to_text(feat):
    if not feat:
        return ""
    return (
        "writing style: "
        f"avg sentence length {feat['avg_sentence_length']:.1f} words, "
        f"avg word length {feat['avg_word_length']:.2f} chars, "
        f"type-token ratio {feat['ttr']:.3f}, "
        f"punctuation {feat['punctuation_frequency']:.1f} per 100 words, "
        f"first-person pronouns {feat['first_person_frequency']:.1f} per 100 words"
    )


def build_repr(uid, tags_by_uid, cat_map, style_by_uid,
               exclude=frozenset(), use_style=True, only_style=False):
    if only_style:
        return style_to_text(style_by_uid.get(uid, {}))
    parts = []
    for t in C.split_tags(tags_by_uid.get(uid, "")):
        if cat_map.get(t, "INT") in exclude:
            continue
        parts.append(t)
    text = ", ".join(parts)
    if use_style:
        s = style_to_text(style_by_uid.get(uid, {}))
        if s:
            text = (text + ". " + s) if text else s
    return text


def _model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


def embed(texts):
    texts = [t if t and t.strip() else "[empty profile]" for t in texts]
    todo = [t for t in set(texts) if t not in _embed_cache]
    if todo:
        vecs = _model().encode(todo, normalize_embeddings=True,
                               show_progress_bar=len(todo) > 200, batch_size=64)
        for t, v in zip(todo, vecs):
            _embed_cache[t] = np.asarray(v, dtype="float32")
    return np.vstack([_embed_cache[t] for t in texts])


class FlatIndex:
    """Exact cosine search over L2-normalised vectors."""

    def __init__(self, matrix):
        self.X = matrix

    def search(self, Q, top_k):
        sims = Q @ self.X.T
        k = min(top_k, sims.shape[1])
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        order = np.argsort(-sims[rows, part], axis=1)
        idx = part[rows, order]
        return idx, sims[rows, idx]


def build_index(candidate_texts):
    X = embed(candidate_texts)
    return FlatIndex(X), X


def search(query_texts, index, top_k=TOP_K):
    return index.search(embed(query_texts), top_k)


REASON_PROMPT = """You are de-anonymizing an online user. Below is a profile built
from a user's EARLIER comments (the query), followed by {k} candidate profiles
built from LATER comments of different users. Exactly one candidate is the same
person as the query. Decide which candidate is the same person.

QUERY PROFILE:
{query}

CANDIDATES:
{candidates}

Respond with ONLY a JSON object:
{{"best": <candidate number 1-{k}, or 0 if none is a plausible match>,
  "confidence": <number 0..1, how certain you are>}}
"""

_reason_lock = threading.Lock()


class DailyLimitReached(Exception):
    pass


class APIUnavailable(Exception):
    pass


def _reason_cache_path(tag):
    return C.path("results", "cache", f"reason_{tag}.json")


def _load_reason_cache(tag):
    p = _reason_cache_path(tag)
    return json.load(open(p)) if os.path.exists(p) else {}


def _key(uid, query_text, cand_texts, model):
    blob = query_text + "||" + "||".join(cand_texts)
    return f"{model}:{uid}:{hashlib.md5(blob.encode()).hexdigest()}"


def reason_one(client, model, query_text, cand_texts):
    body = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(cand_texts))
    msg = REASON_PROMPT.format(k=len(cand_texts), query=query_text, candidates=body)
    for attempt in range(8):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": msg}], timeout=40)
            d = json.loads(r.choices[0].message.content)
            return int(d.get("best", 0)), max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        except Exception as e:
            txt = str(e)
            if "rate_limit" in txt or "429" in txt:
                if "per day" in txt or "RPD" in txt:
                    raise DailyLimitReached(txt[:160])
                m = re.search(r"try again in ([\d.]+)s", txt)
                time.sleep(min(float(m.group(1)) + 0.5, 25) if m else 5)
                continue
            if "timed out" in txt.lower() or "connection" in txt.lower():
                if attempt >= 3:
                    raise APIUnavailable(txt[:160])
                time.sleep(2 * (attempt + 1))
                continue
            if attempt == 7:
                return 0, 0.0
            time.sleep(1.5 * (attempt + 1))
    return 0, 0.0


def run_condition(query_reprs, candidate_reprs, *, model="gpt-4o-mini",
                  cache_tag="default", index=None, cand_uid_order=None,
                  max_workers=8, verbose=True):
    """Run one ESRC condition; returns one result dict per query."""
    cand_uids = cand_uid_order or list(candidate_reprs.keys())
    if index is None:
        index, _ = build_index([candidate_reprs[u] for u in cand_uids])

    q_uids = list(query_reprs.keys())
    q_texts = [query_reprs[u] for u in q_uids]
    topk, topsim = search(q_texts, index, TOP_K)

    cache = _load_reason_cache(cache_tag)
    client = C.get_client()
    results = [None] * len(q_uids)
    done, new_calls = [0], [0]
    abort = threading.Event()

    def work(i):
        if abort.is_set():
            return
        uid = q_uids[i]
        cand_for_q = [cand_uids[j] for j in topk[i]]
        cand_texts_q = [candidate_reprs[c] for c in cand_for_q]
        k = _key(uid, q_texts[i], cand_texts_q, model)
        if k in cache:
            best, conf = cache[k]
        else:
            try:
                best, conf = reason_one(client, model, q_texts[i], cand_texts_q)
            except (DailyLimitReached, APIUnavailable):
                abort.set()
                return
            with _reason_lock:
                cache[k] = [best, conf]
                new_calls[0] += 1
                if new_calls[0] % 25 == 0:
                    C.save_json(cache, _reason_cache_path(cache_tag))
        with _reason_lock:
            done[0] += 1
            if verbose and done[0] % 40 == 0:
                print(f"    [{cache_tag}] {done[0]}/{len(q_uids)}", flush=True)
        pred = cand_for_q[best - 1] if 1 <= best <= len(cand_for_q) else None
        results[i] = {
            "uid": uid,
            "pred": pred,
            "correct": pred == uid,
            "confidence": conf if pred is not None else 0.0,
            "sim": float(topsim[i][best - 1]) if pred is not None else 0.0,
            "retrieved": uid in cand_for_q,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(work, range(len(q_uids))))

    C.save_json(cache, _reason_cache_path(cache_tag))
    if abort.is_set():
        raise DailyLimitReached(
            f"stopped during '{cache_tag}' ({done[0]}/{len(q_uids)}); re-run to resume")
    if verbose:
        acc = np.mean([r["correct"] for r in results])
        ret = np.mean([r["retrieved"] for r in results])
        print(f"  [{cache_tag}] n={len(results)} retrieved@15={ret:.3f} rank1-acc={acc:.3f}",
              flush=True)
    return results


def _score(r):
    return (r["confidence"], r.get("sim", 0.0))


def pr_curve(results):
    rows = sorted(results, key=_score, reverse=True)
    n = len(rows)
    prec, rec, tp, fp = [], [], 0, 0
    for r in rows:
        if r["correct"]:
            tp += 1
        else:
            fp += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / n)
    return np.array(prec), np.array(rec)


def recall_at_precision(results, target):
    prec, rec = pr_curve(results)
    if prec.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(prec[::-1])[::-1]
    ok = rec[envelope >= target]
    return float(ok.max()) if ok.size else 0.0


def bootstrap_ci(results, fn, n_boot=1000, seed=C.SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(results, dtype=object)
    vals = [fn(list(arr[rng.integers(0, len(arr), len(arr))])) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def rank1_accuracy(results):
    return float(np.mean([r["correct"] for r in results]))


def sis(recall_full, recall_without):
    if recall_full == 0:
        return 0.0
    return (recall_full - recall_without) / recall_full * 100.0


def cohens_d_paired(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def paired_test(baseline_results, cond_results):
    """Paired t-test, Cohen's d and McNemar on per-user correctness."""
    from scipy import stats
    by_uid = {r["uid"]: r["correct"] for r in cond_results}
    a, b = [], []
    for r in baseline_results:
        if r["uid"] in by_uid:
            a.append(1.0 if r["correct"] else 0.0)
            b.append(1.0 if by_uid[r["uid"]] else 0.0)
    a, b = np.array(a), np.array(b)
    t_p = float(stats.ttest_rel(a, b).pvalue) if len(a) and a.std() + b.std() > 0 else 1.0
    b01 = int(np.sum((a == 1) & (b == 0)))
    b10 = int(np.sum((a == 0) & (b == 1)))
    mc_p = float(stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0
    return {"t_pvalue": t_p, "cohens_d": cohens_d_paired(a, b), "mcnemar_pvalue": mc_p,
            "n": int(len(a))}
