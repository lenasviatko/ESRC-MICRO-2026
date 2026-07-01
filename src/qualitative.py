"""Qualitative analysis: dump success and failure cases for manual inspection
(Results to Report -> 'Manual analysis of 100 success/failure cases').

Reuses the cached FULL-condition Reason decisions (no new API calls).
"""
import sys

import numpy as np

import common as C
import esrc
import experiments as E


def run(prefix="en", n_each=50):
    pool = {"en": "POOL-EN", "hn": "POOL-HN"}[prefix]
    q, c, cat = E.load_all()
    style_q, style_c = E.load_style(pool)
    users, _, _ = E.rich_users(pool, q, cat)
    if E.MAX_USERS:
        users = users[:E.MAX_USERS]

    qr, cr = E.make_reprs(users, q, c, cat, style_q, style_c)
    cand_order = list(cr.keys())
    index, _ = esrc.build_index([cr[u] for u in cand_order])
    res = esrc.run_condition(qr, cr, model=E.MODEL, cache_tag=f"{prefix}_loo_FULL",
                             index=index, cand_uid_order=cand_order, verbose=False)

    def case(r):
        return {
            "uid": r["uid"], "predicted": r["pred"], "correct": r["correct"],
            "confidence": round(r["confidence"], 3), "retrieved_true": r["retrieved"],
            "query_profile": qr.get(r["uid"], ""),
            "true_candidate_profile": cr.get(r["uid"], ""),
            "predicted_candidate_profile": cr.get(r["pred"], "") if r["pred"] else "",
        }

    succ = [case(r) for r in res if r["correct"]]
    fail = [case(r) for r in res if not r["correct"]]
    # failure breakdown
    miss_retrieval = sum(1 for r in res if not r["retrieved"])
    wrong_pick = sum(1 for r in res if r["retrieved"] and not r["correct"])
    out = {
        "pool": pool, "n": len(res),
        "n_success": len(succ), "n_failure": len(fail),
        "failure_due_to_retrieval_miss": miss_retrieval,
        "failure_due_to_wrong_reason_pick": wrong_pick,
        "success_cases": succ[:n_each],
        "failure_cases": fail[:n_each],
    }
    C.save_json(out, f"{C.EXP}/{prefix}_qualitative.json")
    print(f"[{pool}] success={len(succ)} failure={len(fail)} "
          f"(retrieval-miss={miss_retrieval}, wrong-pick={wrong_pick})")
    print(f"  -> {C.EXP}/{prefix}_qualitative.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "en")
