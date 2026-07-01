"""Classify every unique tag into one of GEO/PROF/DEMO/INT/VAL.

Uses an LLM in batches, resumes from a cache, and writes tag_categories_full.json.
"""
import json
import sys
import time

import common as C

BATCH = 40
MODEL = "gpt-4o-mini"
CACHE = C.CAT_FILE

PROMPT = """Classify each user-attribute tag into exactly one category.

Categories:
- GEO: geographic location, origin, residence, nationality
- PROF: profession, occupation, academic status, employment
- DEMO: demographics — age, gender, family/marital status, housing, having children/pets
- INT: interests, hobbies, fan communities, sports teams, entertainment
- VAL: values, beliefs, political views, religious views, ideology, dietary ethics

Return ONLY a JSON object mapping each input tag (verbatim) to its category code.
Use exactly one of: GEO, PROF, DEMO, INT, VAL.

Tags:
{tags}
"""


def collect_unique_tags():
    q, c = C.load_g2_profiles()
    tags = set()
    for prof in list(q.values()) + list(c.values()):
        tags.update(C.split_tags(prof))
    return sorted(tags)


def classify_batch(client, batch):
    msg = PROMPT.format(tags="\n".join(batch))
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": msg}],
            )
            data = json.loads(r.choices[0].message.content)
            out = {}
            for t in batch:
                v = data.get(t) or data.get(t.strip())
                if isinstance(v, str) and v.strip().upper() in C.CATEGORIES:
                    out[t] = v.strip().upper()
            return out
        except Exception as e:
            print(f"  retry {attempt+1}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return {}


def main():
    all_tags = collect_unique_tags()
    print(f"unique tags: {len(all_tags)}")

    # seed from any existing classification
    cache = {}
    try:
        prev = C.load_json(C.TAGCLS + "/tag_categories_llm.json")
        for k, v in prev.items():
            if v in C.CATEGORIES:
                cache[C.norm_tag(k)] = v
    except Exception:
        pass
    try:
        cache.update(C.load_json(CACHE))
    except Exception:
        pass

    todo = [t for t in all_tags if t not in cache]
    print(f"already classified: {len(cache)}  todo: {len(todo)}")

    client = C.get_client()
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        res = classify_batch(client, batch)
        # fallback: anything the model skipped defaults to INT (rare)
        for t in batch:
            cache[t] = res.get(t, "INT")
        C.save_json(cache, CACHE)
        print(f"  {min(i+BATCH, len(todo))}/{len(todo)} classified", flush=True)

    # final coverage report
    covered = sum(1 for t in all_tags if t in cache)
    from collections import Counter
    dist = Counter(cache[t] for t in all_tags)
    print(f"\ncoverage: {covered}/{len(all_tags)}")
    print("distribution:", dict(dist))


if __name__ == "__main__":
    main()
