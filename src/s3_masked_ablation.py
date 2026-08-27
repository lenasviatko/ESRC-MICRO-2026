"""S3: leave-one-out and additive ablation on masked summaries.

Every profile is one summary with the same five sections (LOCATION,
PROFESSION, DEMOGRAPHICS, INTERESTS, VALUES), built once per user and
period from the first 100 comments; a category is ablated by replacing
its section text with "[masked]", so every condition has the same shape.
The writing-style sentence is appended identically in all conditions.
LOO runs FULL plus NO_<cat> for each category; the additive curve keeps
the top-i categories by the resulting SIS ranking and is checked for
monotonicity. Extract calls are cached by (model, prompt hash, input
hash), Reason calls use the standard esrc cache; a cumulative token
ledger aborts at MAX_TOKENS_TOTAL and a sliding 60s window keeps the
call rate under the TPM limit. A pilot batch runs first to measure real
token usage and extrapolate the cost.

Usage:  python s3_masked_ablation.py
Env:    MODEL=gpt-4o-mini  PILOT=20  PILOT_ONLY=1
        MAX_TOKENS_TOTAL=20000000  TPM_BUDGET=170000
Writes: experiments/s3_masked_en.json
        results/tables/s3_{loo,additive}.csv
"""
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common as C
import esrc
import experiments as E
from s1_style_attack import comments_of

POOL, PREFIX = "POOL-EN", "en"
MODEL = os.environ.get("MODEL", "gpt-4o-mini")
PILOT = int(os.environ.get("PILOT", "20"))
PILOT_ONLY = os.environ.get("PILOT_ONLY", "") == "1"
MAX_TOKENS_TOTAL = int(os.environ.get("MAX_TOKENS_TOTAL", "20000000"))
EUR_IN, EUR_OUT = 0.15, 0.60          # gpt-4o-mini, per 1M tokens
SECTIONS = [("GEO", "LOCATION"), ("PROF", "PROFESSION"), ("DEMO", "DEMOGRAPHICS"),
            ("INT", "INTERESTS"), ("VAL", "VALUES")]

EXTRACT_PROMPT = """You are building a profile of an online user from their \
comments, for a privacy study. Summarize what the comments let you infer \
about the user in EXACTLY five JSON fields:
 "GEO":  places they live or have lived, region, country, city
 "PROF": profession, education, work life
 "DEMO": age, gender, family, relationships
 "INT":  interests and hobbies
 "VAL":  values, politics, religion, worldview
For each field write 1-3 short sentences with the concrete inferences the \
comments support. If the comments support nothing for a field, write exactly \
"no signal". Respond with ONLY a JSON object with those five keys.

COMMENTS:
{comments}"""

LEDGER_PATH = C.path(C.CACHE_DIR, "s3_token_ledger.json")
_lock = threading.Lock()


class BudgetExceeded(Exception):
    pass


def _load_ledger():
    if os.path.exists(LEDGER_PATH):
        return json.load(open(LEDGER_PATH))
    return {"prompt_tokens": 0, "completion_tokens": 0}


LEDGER = _load_ledger()


def _spend(usage):
    with _lock:
        LEDGER["prompt_tokens"] += usage[0]
        LEDGER["completion_tokens"] += usage[1]
        C.save_json(LEDGER, LEDGER_PATH)


def _check_budget():
    total = LEDGER["prompt_tokens"] + LEDGER["completion_tokens"]
    if total > MAX_TOKENS_TOTAL:
        raise BudgetExceeded(
            f"hard stop: {total:,} tokens spent > ceiling {MAX_TOKENS_TOTAL:,}")


def eur(pt, ct):
    return pt / 1e6 * EUR_IN + ct / 1e6 * EUR_OUT


# Extract: one summary per user per period, cached
def _extract_cache_path():
    return C.path(C.CACHE_DIR, f"s3_extract_{POOL}.json")


# TPM throttle (sliding 60s window)
TPM_BUDGET = int(os.environ.get("TPM_BUDGET", "170000"))
_tpm_window = []          # (timestamp, est_tokens)
_tpm_lock = threading.Lock()


def _throttle(est_tokens):
    while True:
        now = time.time()
        with _tpm_lock:
            while _tpm_window and _tpm_window[0][0] < now - 60:
                _tpm_window.pop(0)
            used = sum(t for _, t in _tpm_window)
            if used + est_tokens <= TPM_BUDGET or not _tpm_window:
                _tpm_window.append((now, est_tokens))
                return
            wait = _tpm_window[0][0] + 60 - now
        time.sleep(max(wait, 0.5))


def _call_json(client, prompt):
    """One chat call returning (parsed json, usage); retries like esrc."""
    import re
    last = None
    for attempt in range(12):
        _check_budget()
        _throttle(len(prompt) // 4 + 400)
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}], timeout=60)
            d = json.loads(r.choices[0].message.content)
            return d, (r.usage.prompt_tokens, r.usage.completion_tokens)
        except Exception as e:
            last = e
            txt = str(e)
            if "rate_limit" in txt or "429" in txt:
                if "per day" in txt or "RPD" in txt:
                    raise esrc.DailyLimitReached(txt[:160])
                m = re.search(r"try again in ([\d.]+)m?s", txt)
                time.sleep(min(float(m.group(1)) + 1, 30) if m else 8)
                continue
            time.sleep(2 * (attempt + 1))
    raise esrc.APIUnavailable(f"retries exhausted: {str(last)[:160]}")


def extract_summaries(uids, client, verbose_tag=""):
    cache = json.load(open(_extract_cache_path())) if os.path.exists(_extract_cache_path()) else {}
    prompt_hash = hashlib.md5(EXTRACT_PROMPT.encode()).hexdigest()[:8]
    todo = []
    for uid in uids:
        for side in ("query", "candidate"):
            text = "\n---\n".join(comments_of(POOL, uid, side))
            k = f"{MODEL}:{prompt_hash}:{uid}:{side}:{hashlib.md5(text.encode()).hexdigest()}"
            if k not in cache:
                todo.append((k, uid, side, text))
    done = [0]

    def work(item):
        k, uid, side, text = item
        d, usage = _call_json(client, EXTRACT_PROMPT.format(comments=text))
        summary = {c: str(d.get(c, "no signal")).strip() or "no signal"
                   for c, _ in SECTIONS}
        with _lock:
            cache[k] = {"uid": uid, "side": side, "summary": summary,
                        "usage": list(usage)}
            done[0] += 1
            if done[0] % 20 == 0:
                C.save_json(cache, _extract_cache_path())
                print(f"    [extract{verbose_tag}] {done[0]}/{len(todo)}", flush=True)
        _spend(usage)

    if todo:
        print(f"  [extract{verbose_tag}] {len(todo)} new calls "
              f"({len(uids) * 2 - len(todo)} cached)", flush=True)
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(work, todo))
        C.save_json(cache, _extract_cache_path())

    out = {"query": {}, "candidate": {}}
    for v in cache.values():
        if v["uid"] in set(uids):
            out[v["side"]][v["uid"]] = v["summary"]
    return out


def render(summary, style_sentence, mask=frozenset()):
    lines = []
    for code, label in SECTIONS:
        body = "[masked]" if code in mask else summary.get(code, "no signal")
        lines.append(f"{label}: {body}")
    if style_sentence:
        lines.append(style_sentence)
    return "\n".join(lines)


# Reason: esrc prompt and cache keys, plus usage tracking
def reason_condition(qr, cr, index, cand_order, cache_tag, client):
    cache = esrc._load_reason_cache(cache_tag)
    q_uids = list(qr.keys())
    topk, topsim = esrc.search([qr[u] for u in q_uids], index, esrc.TOP_K)
    results = [None] * len(q_uids)
    new = [0]

    def work(i):
        uid = q_uids[i]
        cand_for_q = [cand_order[j] for j in topk[i]]
        cand_texts = [cr[u] for u in cand_for_q]
        k = esrc._key(uid, qr[uid], cand_texts, MODEL)
        if k in cache:
            best, conf = cache[k]
        else:
            body = "\n".join(f"[{j + 1}] {t}" for j, t in enumerate(cand_texts))
            msg = esrc.REASON_PROMPT.format(k=len(cand_texts), query=qr[uid],
                                            candidates=body)
            d, usage = _call_json(client, msg)
            best = int(d.get("best", 0))
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
            _spend(usage)
            with _lock:
                cache[k] = [best, conf]
                new[0] += 1
                if new[0] % 25 == 0:
                    C.save_json(cache, esrc._reason_cache_path(cache_tag))
        pred = cand_for_q[best - 1] if 1 <= best <= len(cand_for_q) else None
        results[i] = {"uid": uid, "pred": pred, "correct": pred == uid,
                      "confidence": conf if pred is not None else 0.0,
                      "retrieved": uid in cand_for_q}

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(len(q_uids))))
    C.save_json(cache, esrc._reason_cache_path(cache_tag))
    acc = float(np.mean([r["correct"] for r in results]))
    ret = float(np.mean([r["retrieved"] for r in results]))
    print(f"  [{cache_tag}] n={len(results)} retrieved@15={ret:.3f} "
          f"rank1={acc:.3f}  ({new[0]} new calls)", flush=True)
    return results


def spent_str():
    pt, ct = LEDGER["prompt_tokens"], LEDGER["completion_tokens"]
    return f"{pt + ct:,} tokens (~EUR {eur(pt, ct):.2f} at mini prices)"


def main():
    q, c, cat = E.load_all()
    uids, _, _ = E.select_users(POOL, q, cat)
    style_q, style_c = E.load_style(POOL)
    client = C.get_client()
    print(f"S3 masked ablation [{POOL}] n={len(uids)} model={MODEL} "
          f"ceiling={MAX_TOKENS_TOTAL:,} tokens; spent so far: {spent_str()}",
          flush=True)

    # pilot: measure real usage, extrapolate
    pilot_uids = uids[:PILOT]
    before = LEDGER["prompt_tokens"] + LEDGER["completion_tokens"]
    extract_summaries(pilot_uids, client, verbose_tag=":pilot")
    pilot_spend = LEDGER["prompt_tokens"] + LEDGER["completion_tokens"] - before
    if pilot_spend:
        per_user = pilot_spend / len(pilot_uids)
        proj_extract = per_user * len(uids)
        print(f"  pilot: {pilot_spend:,} tokens for {len(pilot_uids)} users "
              f"-> projected extract total {proj_extract:,.0f} tokens", flush=True)
    else:
        print("  pilot: all cached, no new spend", flush=True)

    # full extract, then project the Reason cost from real summary sizes
    summaries = extract_summaries(uids, client)
    missing = [u for u in uids if u not in summaries["query"]
               or u not in summaries["candidate"]]
    if missing:
        raise RuntimeError(f"extract incomplete for {len(missing)} users")

    sq_style = {u: esrc.style_to_text(style_q.get(u, {})) for u in uids}
    sc_style = {u: esrc.style_to_text(style_c.get(u, {})) for u in uids}
    cr = {u: render(summaries["candidate"][u], sc_style[u]) for u in uids}

    avg_chars = np.mean([len(t) for t in cr.values()])
    n_conditions = 10                       # FULL + 5 LOO + 4 additive
    proj_reason = n_conditions * len(uids) * (16 * avg_chars / 4 + 260)
    print(f"  avg profile {avg_chars:.0f} chars -> projected Reason "
          f"~{proj_reason:,.0f} tokens (~EUR {proj_reason / 1e6 * EUR_IN:.2f}); "
          f"ceiling {MAX_TOKENS_TOTAL:,}", flush=True)
    if PILOT_ONLY:
        print("PILOT_ONLY=1 — stopping before the Reason step.", flush=True)
        return

    cand_order = list(cr.keys())
    index, _ = esrc.build_index([cr[u] for u in cand_order])

    def query_reprs(mask):
        return {u: render(summaries["query"][u], sq_style[u], mask=mask)
                for u in uids}

    # Leave-one-out
    out = {}
    conds = [("FULL", frozenset())] + [(f"NO_{x}", frozenset({x}))
                                       for x in C.CATEGORIES]
    for name, mask in conds:
        out[name] = reason_condition(query_reprs(mask), cr, index, cand_order,
                                     f"s3_{PREFIX}_loo_{name}", client)

    base = out["FULL"]
    acc_full = esrc.rank1_accuracy(base)
    loo_rows = []
    for name, res in out.items():
        acc = esrc.rank1_accuracy(res)
        lo, hi = esrc.bootstrap_ci(res, esrc.rank1_accuracy, n_boot=10_000)
        row = {"condition": name, "rank1_acc": round(acc, 4),
               "rank1_CI": [round(lo, 3), round(hi, 3)],
               "recall@90": round(esrc.recall_at_precision(res, 0.90), 4),
               "retrieved@15": round(float(np.mean([x["retrieved"] for x in res])), 3)}
        if name != "FULL":
            row["SIS_rank1"] = round(esrc.sis(acc_full, acc), 1)
            st = esrc.paired_test(base, res)
            row.update({"t_pvalue": round(st["t_pvalue"], 4),
                        "cohens_d": round(st["cohens_d"], 3),
                        "mcnemar_p": round(st["mcnemar_pvalue"], 4)})
        loo_rows.append(row)

    order = [r["condition"][3:] for r in sorted(
        (r for r in loo_rows if r["condition"].startswith("NO_")),
        key=lambda r: r["SIS_rank1"], reverse=True)]
    print(f"  SIS ranking (masked): {order}", flush=True)

    # Additive (new SIS order; step 5 == FULL)
    add_rows = []
    prev_res = None
    for i in range(1, len(order) + 1):
        mask = frozenset(set(C.CATEGORIES) - set(order[:i]))
        res = out["FULL"] if not mask else reason_condition(
            query_reprs(mask), cr, index, cand_order,
            f"s3_{PREFIX}_add_{i}", client)
        acc = esrc.rank1_accuracy(res)
        lo, hi = esrc.bootstrap_ci(res, esrc.rank1_accuracy, n_boot=10_000)
        row = {"step": i, "profile": "+".join(order[:i]),
               "rank1_acc": round(acc, 4), "rank1_CI": [round(lo, 3), round(hi, 3)],
               "frac_of_full": round(acc / acc_full, 3) if acc_full else 0.0}
        if prev_res is not None:
            row["drop_vs_prev"] = round(add_rows[-1]["rank1_acc"] - acc, 4)
            row["mcnemar_p_vs_prev"] = round(
                esrc.paired_test(prev_res, res)["mcnemar_pvalue"], 4)
        add_rows.append(row)
        prev_res = res

    drops = [r for r in add_rows if r.get("drop_vs_prev", 0) > 0]
    sig_drops = [r for r in drops if r.get("mcnemar_p_vs_prev", 1) < 0.05]
    monotone = "monotone" if not drops else (
        "non-monotone within noise" if not sig_drops else "NON-MONOTONE (significant)")

    C.save_json({"pool": POOL, "n": len(uids), "model": MODEL,
                 "representation": "5-section summary, ablation by [masked]",
                 "style_sentence": "identical in all conditions",
                 "truncation": "first 100 comments per period",
                 "rank1_full": round(acc_full, 4),
                 "loo": loo_rows, "sis_order": order, "additive": add_rows,
                 "monotonicity": monotone,
                 "tokens_spent": dict(LEDGER)},
                f"{C.EXP}/s3_masked_{PREFIX}.json")

    import csv
    with open(C.path(C.RES_TABLES, "s3_loo.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "rank1_acc", "rank1_CI",
                                          "recall@90", "retrieved@15", "SIS_rank1",
                                          "t_pvalue", "cohens_d", "mcnemar_p"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(loo_rows)
    with open(C.path(C.RES_TABLES, "s3_additive.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "profile", "rank1_acc", "rank1_CI",
                                          "frac_of_full", "drop_vs_prev",
                                          "mcnemar_p_vs_prev"], extrasaction="ignore")
        w.writeheader()
        w.writerows(add_rows)

    print(f"\nAdditive curve: {monotone}", flush=True)
    print(f"Total spend: {spent_str()}", flush=True)
    print("Wrote experiments/s3_masked_en.json, results/tables/s3_{loo,additive}.csv",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as e:
        print(f"\n*** {e}", flush=True)
        raise SystemExit(3)
    except esrc.DailyLimitReached as e:
        print(f"\n*** STOPPED: {e}\nProgress cached; re-run to resume.", flush=True)
        raise SystemExit(2)
