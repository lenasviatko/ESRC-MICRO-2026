"""S2: compare how well each attack's score separates right from wrong.

LLM attack: the stated confidence, reconstructed per user from the reason
cache (no API calls). Style attack: the top-1 cosine similarity from S1,
plus the top1-top2 gap as an alternative signal. Reports
Recall@90%Precision for all three scores, a reliability table for the
stated confidence, accuracy per decile bin of the style scores, and a
per-user McNemar between the two attacks, Bonferroni-corrected. Ties are
cut only at the end of a block of equal scores.

Usage:  MODEL=gpt-4o python s2_confidence.py
Writes: experiments/s2_confidence_{en,hn}.json
        results/tables/s2_{recall_at_p90,reliability_llm,accuracy_by_bin,mcnemar}.csv
"""
import csv

import numpy as np
from scipy import stats

import common as C
import esrc
import experiments as E

MODEL = E.MODEL
TARGET = 0.90
N_BOOT = 10_000


# LLM decisions, reconstructed from cache
def llm_results(pool, prefix):
    q, c, cat = E.load_all()
    uids, _, _ = E.select_users(pool, q, cat)
    style_q, style_c = E.load_style(pool)
    qr, cr = E.make_reprs(uids, q, c, cat, style_q, style_c)
    cand_order = list(cr.keys())
    index, _ = esrc.build_index([cr[u] for u in cand_order])
    topk, _ = esrc.search([qr[u] for u in uids], index, esrc.TOP_K)

    cache = esrc._load_reason_cache(f"{prefix}_loo_FULL")
    out = []
    for i, uid in enumerate(uids):
        cand_for_q = [cand_order[j] for j in topk[i]]
        k = esrc._key(uid, qr[uid], [cr[u] for u in cand_for_q], MODEL)
        if k not in cache:
            raise RuntimeError(f"cache miss for {uid} — run s2_check_cache.py")
        best, conf = cache[k]
        pred = cand_for_q[best - 1] if 1 <= best <= len(cand_for_q) else None
        out.append({"uid": uid, "pred": pred, "correct": pred == uid,
                    "confidence": conf if pred is not None else 0.0})
    return out


# Recall at 90% precision (ties cut only at block ends)
def recall_at_p(scores, correct, target=TARGET):
    scores = np.asarray(scores, float)
    correct = np.asarray(correct, bool)
    n_true = len(correct)
    order = np.argsort(-scores, kind="mergesort")
    s, c = scores[order], correct[order]
    hits = np.cumsum(c)
    prec = hits / np.arange(1, len(s) + 1)
    rec = hits / n_true
    block_end = np.r_[s[1:] != s[:-1], True]
    ok = (prec >= target) & block_end
    return float(rec[ok].max()) if ok.any() else 0.0


def boot_ci(fn, n, seed=C.SEED):
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


def r90_with_ci(scores, correct):
    scores = np.asarray(scores, float)
    correct = np.asarray(correct, bool)
    val = recall_at_p(scores, correct)
    lo, hi = boot_ci(lambda idx: recall_at_p(scores[idx], correct[idx]), len(correct))
    return round(val, 4), [lo, hi]


# Reliability of the stated confidence
def reliability_table(res):
    conf = np.array([round(r["confidence"], 2) for r in res])
    corr = np.array([r["correct"] for r in res], dtype=bool)
    rows = []
    for v in sorted(set(conf)):
        m = conf == v
        acc = float(corr[m].mean())
        lo, hi = boot_ci(lambda idx, c=corr[m]: float(c[idx].mean()), int(m.sum()))
        rows.append({"stated_confidence": v, "n": int(m.sum()),
                     "actual_accuracy": round(acc, 3), "acc_CI": [lo, hi],
                     "gap": round(v - acc, 3)})
    return rows


# Accuracy by decile bin
def accuracy_by_bin(scores, correct, n_bins=10):
    scores = np.asarray(scores, float)
    correct = np.asarray(correct, bool)
    order = np.argsort(scores, kind="mergesort")
    rows = []
    for b, idx in enumerate(np.array_split(order, n_bins), 1):
        acc = float(correct[idx].mean())
        lo, hi = boot_ci(lambda i, c=correct[idx]: float(c[i].mean()), len(idx))
        rows.append({"bin": b, "n": len(idx),
                     "score_range": [round(float(scores[idx].min()), 4),
                                     round(float(scores[idx].max()), 4)],
                     "accuracy": round(acc, 3), "acc_CI": [lo, hi]})
    return rows


# McNemar (continuity-corrected)
def mcnemar(a_correct, b_correct):
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    n_b = int(np.sum(a & ~b))     # style right, LLM wrong
    n_c = int(np.sum(~a & b))     # style wrong, LLM right
    if n_b + n_c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_raw": 1.0}
    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
    return {"b": n_b, "c": n_c, "chi2": round(chi2, 3),
            "p_raw": float(stats.chi2.sf(chi2, 1))}


def run_pool(pool, prefix):
    print(f"\n=== S2 [{pool}] ===", flush=True)
    llm = llm_results(pool, prefix)
    style = C.load_json(f"{C.EXP}/s1_style_{prefix}.json")["results"]
    s_by_uid = {r["uid"]: r for r in style}
    uids = [r["uid"] for r in llm]
    assert set(uids) == set(s_by_uid), "S1 and LLM user sets differ"

    llm_corr = np.array([r["correct"] for r in llm], dtype=bool)
    llm_conf = np.array([r["confidence"] for r in llm], float)
    st_corr = np.array([s_by_uid[u]["correct"] for u in uids], dtype=bool)
    st_sim = np.array([s_by_uid[u]["sim"] for u in uids], float)
    st_gap = np.array([s_by_uid[u]["gap_top1_top2"] for u in uids], float)

    r90 = {
        "llm_confidence": r90_with_ci(llm_conf, llm_corr),
        "style_sim_top1": r90_with_ci(st_sim, st_corr),
        "style_gap_top1_top2": r90_with_ci(st_gap, st_corr),
    }
    for name, (v, ci) in r90.items():
        print(f"  Recall@90%P  {name:<22} {v:.4f} {ci}", flush=True)

    rel = reliability_table(llm)
    bins_sim = accuracy_by_bin(st_sim, st_corr)
    bins_gap = accuracy_by_bin(st_gap, st_corr)
    mc = mcnemar(st_corr, llm_corr)
    print(f"  rank1: style={st_corr.mean():.3f} llm={llm_corr.mean():.3f}   "
          f"McNemar b={mc['b']} c={mc['c']} chi2={mc.get('chi2')} p={mc['p_raw']:.2e}",
          flush=True)

    out = {"pool": pool, "n": len(uids), "model_llm": MODEL, "target_precision": TARGET,
           "tie_handling": "cuts only at ends of equal-score blocks",
           "recall_at_p90": {k: {"value": v, "CI": ci} for k, (v, ci) in r90.items()},
           "rank1": {"style": round(float(st_corr.mean()), 4),
                     "llm": round(float(llm_corr.mean()), 4)},
           "reliability_llm": rel,
           "accuracy_by_bin_sim": bins_sim,
           "accuracy_by_bin_gap": bins_gap,
           "mcnemar_style_vs_llm": mc}
    C.save_json(out, f"{C.EXP}/s2_confidence_{prefix}.json")
    return out


def write_csvs(res):
    T = C.RES_TABLES
    with open(C.path(T, "s2_recall_at_p90.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "score", "recall@90%P", "CI_lo", "CI_hi"])
        for r in res:
            for name, d in r["recall_at_p90"].items():
                w.writerow([r["pool"], name, d["value"], d["CI"][0], d["CI"][1]])
    with open(C.path(T, "s2_reliability_llm.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "stated_confidence", "n", "actual_accuracy",
                    "acc_CI_lo", "acc_CI_hi", "gap"])
        for r in res:
            for row in r["reliability_llm"]:
                w.writerow([r["pool"], row["stated_confidence"], row["n"],
                            row["actual_accuracy"], *row["acc_CI"], row["gap"]])
    with open(C.path(T, "s2_accuracy_by_bin.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "score", "bin", "n", "score_lo", "score_hi",
                    "accuracy", "acc_CI_lo", "acc_CI_hi"])
        for r in res:
            for name, key in [("sim_top1", "accuracy_by_bin_sim"),
                              ("gap_top1_top2", "accuracy_by_bin_gap")]:
                for row in r[key]:
                    w.writerow([r["pool"], name, row["bin"], row["n"],
                                *row["score_range"], row["accuracy"], *row["acc_CI"]])
    with open(C.path(T, "s2_mcnemar.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "n", "style_rank1", "llm_rank1",
                    "b_style_only_right", "c_llm_only_right", "chi2",
                    "p_raw", "p_bonferroni_k2"])
        for r in res:
            mc = r["mcnemar_style_vs_llm"]
            w.writerow([r["pool"], r["n"], r["rank1"]["style"], r["rank1"]["llm"],
                        mc["b"], mc["c"], mc.get("chi2"), f"{mc['p_raw']:.3e}",
                        f"{mc['p_corrected']:.3e}"])


def main():
    res = [run_pool("POOL-EN", "en"), run_pool("POOL-HN", "hn")]
    # Bonferroni over the two comparisons
    k = len(res)
    for r in res:
        mc = r["mcnemar_style_vs_llm"]
        mc["p_corrected"] = min(1.0, mc["p_raw"] * k)
        C.save_json(r, f"{C.EXP}/s2_confidence_{r['pool'].split('-')[1].lower()}.json")
    write_csvs(res)
    print("\nWrote results/tables/s2_*.csv")


if __name__ == "__main__":
    main()
