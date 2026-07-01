"""Driver for the ablation experiments.

Usage:
    python experiments.py rich           # rich-profile filter report
    python experiments.py loo  EN        # Leave-One-Out + SIS
    python experiments.py additive EN    # Additive (needs loo first)
    python experiments.py interaction EN # Interaction
    python experiments.py hn             # Hacker News replication (loo + additive)
    python experiments.py style EN       # STYLE-only by TTR group
    python experiments.py all            # everything

Env knobs:
    MODEL=gpt-4o-mini   (Reason model; use gpt-4o for the final runs)
    MAX_USERS=0         (0 = all rich users; set small for a quick run)
"""
import os
import sys
from collections import Counter

import numpy as np

import common as C
import esrc

MODEL = os.environ.get("MODEL", "gpt-4o-mini")
MAX_USERS = int(os.environ.get("MAX_USERS", "0"))
MIN_CATS = int(os.environ.get("MIN_CATS", "5"))  # rich = >= this many of 5 categories
CAT_FILE = C.CAT_FILE


# Data assembly
def load_all():
    q, c = C.load_g2_profiles()
    cat = {C.norm_tag(k): v for k, v in C.load_json(CAT_FILE).items()}
    return q, c, cat


def load_style(pool):
    """Stylometric features for the query and candidate periods."""
    sq = C.load_json(f"{C.STYLEDIR}/style_query_{pool}.json")
    sc = C.load_json(f"{C.STYLEDIR}/style_candidate_{pool}.json")
    return sq, sc


def categories_present(uid, tags_by_uid, cat):
    return {cat.get(t, "INT") for t in C.split_tags(tags_by_uid.get(uid, ""))} & set(C.CATEGORIES)


def rich_users(pool, q, cat, min_cats=MIN_CATS):
    pool_uids = C.pool_users(pool)
    users, absent = [], Counter()
    for uid in sorted(pool_uids):
        present = categories_present(uid, q, cat)
        for code in C.CATEGORIES:
            if code not in present:
                absent[code] += 1
        if len(present) >= min_cats:
            users.append(uid)
    return users, absent, len(pool_uids)


def make_reprs(uids, q, c, cat, style_q, style_c, *, exclude=frozenset(),
               use_style=True, only_style=False):
    """Build query (ablated) and candidate (full) profile representations."""
    qr = {u: esrc.build_repr(u, q, cat, style_q, exclude=exclude,
                             use_style=use_style, only_style=only_style)
          for u in uids}
    cr = {u: esrc.build_repr(u, c, cat, style_c,
                             exclude=frozenset(),
                             use_style=not only_style,
                             only_style=only_style)
          for u in uids}
    return qr, cr


# Experiment 1: Leave-One-Out
def exp_loo(pool, uids, q, c, cat, style_q, style_c, prefix):
    print(f"\n=== Exp1 Leave-One-Out [{pool}] n={len(uids)} model={MODEL} ===")
    _, cr_full = make_reprs(uids, q, c, cat, style_q, style_c)
    cand_order = list(cr_full.keys())
    index, _ = esrc.build_index([cr_full[u] for u in cand_order])

    conditions = [("FULL", frozenset(), True)]
    for code in C.CATEGORIES:
        conditions.append((f"NO_{code}", frozenset({code}), True))
    conditions.append(("NO_STYLE", frozenset(), False))

    out = {}
    for name, excl, use_style in conditions:
        qr, _ = make_reprs(uids, q, c, cat, style_q, style_c, exclude=excl, use_style=use_style)
        res = esrc.run_condition(qr, cr_full, model=MODEL,
                                 cache_tag=f"{prefix}_loo_{name}",
                                 index=index, cand_uid_order=cand_order)
        out[name] = res

    rows, base = [], out["FULL"]
    r_full_90 = esrc.recall_at_precision(base, 0.90)
    r_full_99 = esrc.recall_at_precision(base, 0.99)
    acc_full = esrc.rank1_accuracy(base)
    r90_lo, r90_hi = esrc.bootstrap_ci(base, lambda r: esrc.recall_at_precision(r, 0.90))
    for name, res in out.items():
        r90 = esrc.recall_at_precision(res, 0.90)
        acc = esrc.rank1_accuracy(res)
        ci_lo, ci_hi = esrc.bootstrap_ci(res, lambda r: esrc.recall_at_precision(r, 0.90))
        row = {"condition": name,
               "recall@90": round(r90, 4), "r90_CI": [round(ci_lo, 3), round(ci_hi, 3)],
               "recall@99": round(esrc.recall_at_precision(res, 0.99), 4),
               "rank1_acc": round(acc, 4),
               "retrieved@15": round(float(np.mean([x["retrieved"] for x in res])), 3)}
        if name.startswith("NO_"):
            row["SIS_recall90"] = round(esrc.sis(r_full_90, r90), 1)
            row["SIS_rank1"] = round(esrc.sis(acc_full, acc), 1)
            st = esrc.paired_test(base, res)
            row["t_pvalue"] = round(st["t_pvalue"], 4)
            row["cohens_d"] = round(st["cohens_d"], 3)
            row["mcnemar_p"] = round(st["mcnemar_pvalue"], 4)
        rows.append(row)

    C.save_json({"pool": pool, "n": len(uids), "model": MODEL,
                 "recall_full@90": r_full_90, "recall_full@99": r_full_99,
                 "recall_full@90_CI": [round(r90_lo, 3), round(r90_hi, 3)],
                 "rank1_full": round(acc_full, 4), "rows": rows},
                f"{C.EXP}/{prefix}_loo.json")
    _print_table(rows)
    return rows, r_full_90


# Experiment 2: Additive (uses SIS ranking from LOO)
def sis_ranking(loo_rows):
    """Order the five content categories by rank-1 SIS (most important first)."""
    ranked = [(r["condition"][3:], r.get("SIS_rank1", 0.0))
              for r in loo_rows if r["condition"].startswith("NO_") and r["condition"] != "NO_STYLE"]
    ranked = [x for x in ranked if x[0] in C.CATEGORIES]
    ranked.sort(key=lambda x: x[1], reverse=True)
    order = [code for code, _ in ranked]
    return order


def exp_additive(pool, uids, q, c, cat, style_q, style_c, prefix, order, r_full_90):
    print(f"\n=== Exp2 Additive [{pool}] order={order}+STYLE ===")
    _, cr_full = make_reprs(uids, q, c, cat, style_q, style_c)
    cand_order = list(cr_full.keys())
    index, _ = esrc.build_index([cr_full[u] for u in cand_order])

    acc_full = esrc.rank1_accuracy(
        esrc.run_condition({u: esrc.build_repr(u, q, cat, style_q) for u in uids},
                           cr_full, model=MODEL, cache_tag=f"{prefix}_loo_FULL",
                           index=index, cand_uid_order=cand_order, verbose=False))

    def add_row(step, profile, res):
        r90 = esrc.recall_at_precision(res, 0.90)
        acc = esrc.rank1_accuracy(res)
        return {"step": step, "profile": profile,
                "recall@90": round(r90, 4), "rank1_acc": round(acc, 4),
                "frac_of_full": round(acc / acc_full, 3) if acc_full else 0.0}

    rows = []
    for i in range(1, len(order) + 1):
        keep = set(order[:i])
        exclude = frozenset(set(C.CATEGORIES) - keep)
        qr, _ = make_reprs(uids, q, c, cat, style_q, style_c, exclude=exclude, use_style=False)
        res = esrc.run_condition(qr, cr_full, model=MODEL, cache_tag=f"{prefix}_add_{i}",
                                 index=index, cand_uid_order=cand_order)
        rows.append(add_row(i, "+".join(order[:i]), res))
    qr, _ = make_reprs(uids, q, c, cat, style_q, style_c, exclude=frozenset(), use_style=True)
    res = esrc.run_condition(qr, cr_full, model=MODEL, cache_tag=f"{prefix}_add_full",
                             index=index, cand_uid_order=cand_order)
    rows.append(add_row(len(order) + 1, "+".join(order) + "+STYLE", res))

    msss = next((r["profile"] for r in rows if r["frac_of_full"] >= 0.80), rows[-1]["profile"])
    C.save_json({"pool": pool, "order": order, "rows": rows, "MSSS": msss},
                f"{C.EXP}/{prefix}_additive.json")
    for r in rows:
        print(f"  step{r['step']:>2} {r['profile']:<40} recall@90={r['recall@90']:.3f} "
              f"rank1={r['rank1_acc']:.3f} ({r['frac_of_full']*100:.0f}% of full)")
    print(f"  MSSS (>=80% of full rank-1): {msss}")
    return rows


# Experiment 3: Interaction (top-3 by SIS)
def exp_interaction(pool, uids, q, c, cat, style_q, style_c, prefix, order, loo_rows):
    top3 = order[:3]
    print(f"\n=== Exp3 Interaction [{pool}] top3={top3} ===")
    _, cr_full = make_reprs(uids, q, c, cat, style_q, style_c)
    cand_order = list(cr_full.keys())
    index, _ = esrc.build_index([cr_full[u] for u in cand_order])

    base = C.load_json(f"{C.EXP}/{prefix}_loo.json")
    r_full = base["rank1_full"]
    single = {r["condition"][3:]: r["rank1_acc"] for r in loo_rows if r["condition"].startswith("NO_")}

    rows = []
    pairs = [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2])]
    for ci, cj in pairs:
        qr, _ = make_reprs(uids, q, c, cat, style_q, style_c, exclude=frozenset({ci, cj}), use_style=True)
        res = esrc.run_condition(qr, cr_full, model=MODEL,
                                 cache_tag=f"{prefix}_int_{ci}_{cj}",
                                 index=index, cand_uid_order=cand_order)
        r_ij = esrc.rank1_accuracy(res)
        d_i = r_full - single[ci]
        d_j = r_full - single[cj]
        d_ij = r_full - r_ij
        if abs(d_ij - (d_i + d_j)) < 0.02:
            kind = "independent"
        elif d_ij < d_i + d_j:
            kind = "complementary"
        else:
            kind = "substitutive"
        rows.append({"pair": f"{ci}+{cj}", "recall@90_without_both": round(r_ij, 4),
                     "dI": round(d_i, 4), "dJ": round(d_j, 4),
                     "dIJ": round(d_ij, 4), "dI+dJ": round(d_i + d_j, 4),
                     "interaction": kind})
    C.save_json({"pool": pool, "top3": top3, "rows": rows}, f"{C.EXP}/{prefix}_interaction.json")
    for r in rows:
        print(f"  {r['pair']:<10} dIJ={r['dIJ']:.3f} vs dI+dJ={r['dI+dJ']:.3f} -> {r['interaction']}")
    return rows


# STYLE-only by TTR group
def exp_style(pool, uids, q, c, cat, style_q, style_c, prefix):
    print(f"\n=== Step11 STYLE-only by TTR group [{pool}] ===")
    have = [u for u in uids if u in style_q and style_q[u].get("ttr") is not None]
    ttrs = sorted(have, key=lambda u: style_q[u]["ttr"])
    n = len(ttrs)
    groups = {"low_TTR": ttrs[:n // 3], "mid_TTR": ttrs[n // 3:2 * n // 3], "high_TTR": ttrs[2 * n // 3:]}
    rows = []
    for gname, gusers in groups.items():
        if not gusers:
            continue
        qr, cr = make_reprs(gusers, q, c, cat, style_q, style_c, only_style=True)
        cand_order = list(cr.keys())
        index, _ = esrc.build_index([cr[u] for u in cand_order])
        res = esrc.run_condition(qr, cr, model=MODEL, cache_tag=f"{prefix}_style_{gname}",
                                 index=index, cand_uid_order=cand_order)
        rows.append({"group": gname, "n": len(gusers),
                     "ttr_range": [round(style_q[gusers[0]]["ttr"], 3), round(style_q[gusers[-1]]["ttr"], 3)],
                     "recall@90": round(esrc.recall_at_precision(res, 0.90), 4),
                     "rank1_acc": round(esrc.rank1_accuracy(res), 4)})
    C.save_json({"pool": pool, "rows": rows}, f"{C.EXP}/{prefix}_style_ttr.json")
    for r in rows:
        print(f"  {r['group']:<9} n={r['n']:<4} ttr{r['ttr_range']} recall@90={r['recall@90']:.3f}")
    return rows


def _print_table(rows):
    hdr = ["condition", "recall@90", "rank1_acc", "retrieved@15",
           "SIS_recall90", "SIS_rank1", "t_pvalue", "cohens_d", "mcnemar_p"]
    print("  " + " ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print("  " + " ".join(f"{str(r.get(h,'')):>10}" for h in hdr))


def select_users(pool, q, cat):
    users, absent, total = rich_users(pool, q, cat)
    if MAX_USERS:
        users = users[:MAX_USERS]
    return users, absent, total


def cmd_rich(q, c, cat):
    print(f"=== rich-profile filter (min {MIN_CATS}/5 categories) ===")
    for pool in ["POOL-EN", "POOL-HN"]:
        users, absent, total = rich_users(pool, q, cat)
        print(f"\n{pool}: total={total} rich={len(users)} ({len(users)/total*100:.0f}%)")
        print("  most-absent categories:", dict(absent.most_common()))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    arg = sys.argv[2] if len(sys.argv) > 2 else "EN"
    q, c, cat = load_all()
    pool = {"EN": "POOL-EN", "HN": "POOL-HN"}.get(arg, arg)
    prefix = arg.lower()

    if cmd == "rich":
        cmd_rich(q, c, cat)
        return

    users, absent, total = select_users(pool, q, cat)
    style_q, style_c = load_style(pool)
    print(f"{pool}: using {len(users)} rich users (of {total})")

    if cmd in ("loo", "all"):
        loo_rows, r_full = exp_loo(pool, users, q, c, cat, style_q, style_c, prefix)
    else:
        data = C.load_json(f"{C.EXP}/{prefix}_loo.json")
        loo_rows, r_full = data["rows"], data["recall_full@90"]
    order = sis_ranking(loo_rows)

    if cmd in ("additive", "all"):
        exp_additive(pool, users, q, c, cat, style_q, style_c, prefix, order, r_full)
    if cmd in ("interaction", "all"):
        exp_interaction(pool, users, q, c, cat, style_q, style_c, prefix, order, loo_rows)
    if cmd in ("style", "all"):
        exp_style(pool, users, q, c, cat, style_q, style_c, prefix)


if __name__ == "__main__":
    try:
        main()
    except esrc.DailyLimitReached as e:
        print(f"\n*** STOPPED: {e}", flush=True)
        print("Progress is cached in results/cache/. Re-run the same command after "
              "the OpenAI daily quota resets (00:00 UTC) to resume from cache.", flush=True)
        sys.exit(2)
