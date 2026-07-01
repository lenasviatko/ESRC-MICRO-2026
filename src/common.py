"""Shared paths, data loaders and the OpenAI client."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATEGORIES = ["GEO", "PROF", "DEMO", "INT", "VAL"]
SEED = 42

RAW = "data/raw"
SUMMARIES = "data/summaries"
TAGCLS = "data/tag_classifications"
STYLEDIR = "data/style_features"
SPLITS = "data/splits"
EXP = "experiments"
RES_TABLES = "results/tables"
RES_FIGURES = "results/figures"
CACHE_DIR = "results/cache"
CAT_FILE = TAGCLS + "/tag_categories_full.json"


def path(*p):
    return os.path.join(ROOT, *p)


def raw_pool(name):
    return path(RAW, name)


def load_env():
    env = path(".env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_client():
    from openai import OpenAI
    import httpx
    load_env()
    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=12, max_keepalive_connections=0),
    )
    return OpenAI(max_retries=0, http_client=http_client)


def uid_of(key):
    return key.replace("_query.jsonl", "").replace("_candidate.jsonl", "")


def norm_tag(t):
    return t.strip().lower()


def split_tags(profile_str):
    out, seen = [], set()
    for t in profile_str.split(","):
        t = norm_tag(t)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_json(p):
    with open(path(p) if not os.path.isabs(p) else p) as f:
        return json.load(f)


def save_json(obj, p):
    full = path(p) if not os.path.isabs(p) else p
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_g2_profiles():
    q = {uid_of(k): v for k, v in load_json(SUMMARIES + "/query_profiles_g2_812.json").items()}
    c = {uid_of(k): v for k, v in load_json(SUMMARIES + "/candidate_profiles_g2_812.json").items()}
    return q, c


def pool_users(pool):
    fs = set(os.listdir(raw_pool(pool)))
    uids = set()
    for fn in fs:
        if fn.endswith("_query.jsonl"):
            uid = fn.replace("_query.jsonl", "")
            if (uid + "_candidate.jsonl") in fs:
                uids.add(uid)
    return uids
