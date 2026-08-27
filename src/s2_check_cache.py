"""Check that the per-user Reason decisions are recoverable from cache.

Rebuilds the FULL-condition representations, redoes the deterministic
embedding + top-15 search, recomputes every cache key and looks it up
in results/cache/reason_*_loo_FULL.json. No API calls.

Usage: MODEL=gpt-4o python s2_check_cache.py
"""
import esrc
import experiments as E

MODEL = E.MODEL


def check(pool, prefix):
    q, c, cat = E.load_all()
    uids, _, _ = E.select_users(pool, q, cat)
    style_q, style_c = E.load_style(pool)
    qr, cr = E.make_reprs(uids, q, c, cat, style_q, style_c)
    cand_order = list(cr.keys())
    index, _ = esrc.build_index([cr[u] for u in cand_order])
    topk, _ = esrc.search([qr[u] for u in uids], index, esrc.TOP_K)

    cache = esrc._load_reason_cache(f"{prefix}_loo_FULL")
    hits = miss = 0
    for i, uid in enumerate(uids):
        cand_texts = [cr[cand_order[j]] for j in topk[i]]
        k = esrc._key(uid, qr[uid], cand_texts, MODEL)
        if k in cache:
            hits += 1
        else:
            miss += 1
    print(f"{pool}: {hits}/{hits + miss} per-user records recoverable "
          f"from cache ({'OK' if miss == 0 else 'MISSING — re-run needed'})")
    return miss == 0


if __name__ == "__main__":
    ok = all([check("POOL-EN", "en"), check("POOL-HN", "hn")])
    raise SystemExit(0 if ok else 1)
