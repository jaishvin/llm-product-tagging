# %% [markdown]
# # LLM Product Attribute Tagging: Gemini vs a Classic ML Baseline
# **Data:** 44k Myntra fashion products (Kaggle) · **Stack:** Gemini API (structured JSON output), scikit-learn, pandas
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jaishvin/llm-product-tagging/blob/main/notebooks/llm_product_tagging.ipynb)
#
# **Business question:** catalogue teams tag every new product with attributes (gender, article type, colour,
# usage) that drive search filters, category reporting and demand planning. Can an LLM tag products from the
# title alone, and when is it better than a model trained on labelled data?
#
# **Approach**
# 1. Take product titles with known catalogue tags as ground truth.
# 2. **Baseline:** TF-IDF + logistic regression per attribute, trained on labelled titles.
# 3. **LLM:** Gemini with zero-shot and few-shot prompts, batched (25 products per call), forced to return
#    JSON restricted to the allowed values.
# 4. Compare accuracy and macro-F1 per attribute, check how much labelled data the baseline needs to match the
#    LLM, and analyse the errors.

# %% [markdown]
# ## 1. Setup

# %%
import os, sys, json, time, glob, zipfile, subprocess, warnings
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai", "kaggle"], check=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.figsize": (10, 4), "axes.grid": True, "grid.alpha": 0.3})

# %% [markdown]
# ## 2. Configuration
# `DEMO_MODE = True` uses a small synthetic catalogue and a fake LLM, so you can test the notebook end to end
# without any API keys.

# %%
DEMO_MODE = False                 # True = synthetic data + fake LLM (quick test, no keys needed)
DATA_DIR = "data"
RESULTS_DIR = "results"
N_TYPES = 20                      # keep the 20 most common article types
N_TEST_PER_TYPE = 15              # balanced test set: 20 types x 15 = 300 products
BATCH_SIZE = 25                   # products per LLM call (fewer calls = stays inside free-tier limits)
SECONDS_BETWEEN_CALLS = 7         # free tier allows ~10 requests/minute
N_FEW_SHOT = 12                   # labelled examples shown to the LLM in the few-shot prompt
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]
ATTRS = ["gender", "articleType", "baseColour", "usage"]
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## 3. Get the data
# Uses the `KAGGLE_API_TOKEN` secret you already added for the M5 project. The Gemini key goes in a
# `GOOGLE_API_KEY` secret (Colab's key panel → **Gemini API keys** → create one, it's free).

# %%
def download_styles(data_dir):
    target = os.path.join(data_dir, "styles.csv")
    if os.path.exists(target):
        return target
    if IN_COLAB:
        from google.colab import userdata
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    ds = "paramaggarwal/fashion-product-images-small"
    for f in ["styles.csv", "myntradataset/styles.csv"]:     # try to download just the small metadata file
        r = subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-f", f, "-p", data_dir],
                           capture_output=True, text=True)
        if r.returncode == 0:
            break
    else:                                                     # fallback: whole dataset (~600 MB), keep styles.csv only
        print("Downloading the full dataset (a few minutes)...")
        r = subprocess.run(["kaggle", "datasets", "download", "-d", ds, "-p", data_dir], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Kaggle download failed:\n" + r.stdout + r.stderr)
    for z in glob.glob(os.path.join(data_dir, "*.zip")):
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if member.endswith("styles.csv"):
                    zf.extract(member, data_dir)
        os.remove(z)
    found = glob.glob(os.path.join(data_dir, "**", "styles.csv"), recursive=True)
    if found and found[0] != target:
        os.replace(found[0], target)
    return target


def make_synthetic_catalogue(n=3000, seed=SEED):
    """Small catalogue in the same format as styles.csv (for testing only)."""
    r = np.random.default_rng(seed)
    types = {"Tshirts": "Casual", "Shirts": "Formal", "Kurtas": "Ethnic", "Jeans": "Casual", "Sports Shoes": "Sports",
             "Heels": "Party", "Watches": "Casual", "Handbags": "Casual", "Track Pants": "Sports", "Sarees": "Ethnic"}
    colours = ["Black", "White", "Blue", "Navy Blue", "Red", "Grey", "Green", "Pink", "Brown", "Beige"]
    genders = ["Men", "Women", "Boys", "Girls", "Unisex"]
    brands = ["Roadster", "Puma", "Nike", "Fabindia", "W", "Wrogn", "Biba", "Fossil", "Baggit", "HRX"]
    rows = []
    for i in range(n):
        t = r.choice(list(types)); c = r.choice(colours); g = r.choice(genders, p=[.45, .4, .05, .05, .05])
        usage = types[t] if r.random() > 0.15 else r.choice(["Casual", "Formal", "Sports", "Ethnic", "Party"])
        extra = r.choice(["", "Printed", "Slim Fit", "Solid", "Striped", "Classic"])
        name = " ".join(x for x in [r.choice(brands), g if r.random() > 0.2 else "", c, extra, t] if x)
        rows.append({"id": i, "gender": g, "masterCategory": "Apparel", "subCategory": "x", "articleType": t,
                     "baseColour": c, "season": "Summer", "year": 2016, "usage": usage, "productDisplayName": name})
    return pd.DataFrame(rows)


if DEMO_MODE:
    raw = make_synthetic_catalogue()
else:
    raw = pd.read_csv(download_styles(DATA_DIR), on_bad_lines="skip")   # a few rows have stray commas
print(f"{len(raw):,} products loaded")
raw.head()

# %% [markdown]
# ## 4. Prepare labels and a balanced test set

# %%
df = raw.dropna(subset=ATTRS + ["productDisplayName"]).copy()
df["title"] = df["productDisplayName"].str.strip()
df = df[df["articleType"].isin(df["articleType"].value_counts().head(N_TYPES).index)]
top_colours = df["baseColour"].value_counts().head(15).index
df["baseColour"] = df["baseColour"].where(df["baseColour"].isin(top_colours), "Other")
usage_counts = df["usage"].value_counts()
df = df[df["usage"].isin(usage_counts[usage_counts >= 30].index)]
df = df.drop_duplicates("title").reset_index(drop=True)

ALLOWED = {a: sorted(str(v) for v in df[a].unique()) for a in ATTRS}
for a in ATTRS:
    print(f"{a:12s} {len(ALLOWED[a]):2d} classes: {ALLOWED[a][:8]}{' ...' if len(ALLOWED[a]) > 8 else ''}")

test = df.groupby("articleType").sample(n=N_TEST_PER_TYPE, random_state=SEED).reset_index(drop=True)
train = df[~df["id"].isin(test["id"])].reset_index(drop=True)
print(f"\nTrain: {len(train):,} labelled titles | Test: {len(test)} titles")
test[["title"] + ATTRS].head(8)

# %% [markdown]
# ## 5. Baseline: TF-IDF + logistic regression
# One classifier per attribute, trained on labelled titles. Also trained on smaller samples (50 to all rows) to
# see how much labelled data a classic model needs.

# %%
def train_baseline(train_df, n=None):
    sample = train_df if n is None or n >= len(train_df) else train_df.sample(n, random_state=SEED)
    models = {}
    for a in ATTRS:
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=2000, C=5))
        y = sample[a]
        if y.nunique() < 2:            # tiny samples can contain a single class
            models[a] = ("constant", y.iloc[0])
        else:
            models[a] = m.fit(sample["title"], y)
    return models

def predict_baseline(models, titles):
    out = {}
    for a, m in models.items():
        out[a] = [m[1]] * len(titles) if isinstance(m, tuple) else m.predict(titles)
    return pd.DataFrame(out)

t0 = time.time()
baseline_full = train_baseline(train)
pred_baseline = predict_baseline(baseline_full, test["title"])
print(f"Baseline trained on {len(train):,} titles in {time.time() - t0:.1f}s")

# %% [markdown]
# ## 6. LLM tagging with Gemini
# Each call sends 25 titles and asks for a JSON array. A **response schema** restricts every field to the allowed
# values, so the output can be parsed without cleanup. Raw responses are cached to `results/`, so re-running the
# notebook doesn't spend API quota again.

# %%
def build_prompt(items, examples=None):
    lines = [
        "You are a catalogue assistant for an Indian fashion e-commerce site.",
        "For each product title, assign gender, articleType, baseColour and usage.",
        "Use ONLY the allowed values. If the title doesn't say, choose the most likely value.",
        "",
        "Allowed values:",
        *[f"- {a}: {', '.join(ALLOWED[a])}" for a in ATTRS],
    ]
    if examples is not None and len(examples):
        lines += ["", "Examples of correctly tagged products:"]
        for r in examples.itertuples():
            tags = {a: getattr(r, a) for a in ATTRS}
            lines.append(f'- "{r.title}" -> {json.dumps(tags)}')
    lines += ["", "Products:", json.dumps([{"id": int(i), "title": t} for i, t in items], ensure_ascii=False),
              "", "Return a JSON array with one object per product: id, gender, articleType, baseColour, usage."]
    return "\n".join(lines)


RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {"type": "OBJECT",
              "properties": {"id": {"type": "INTEGER"},
                             **{a: {"type": "STRING", "enum": ALLOWED[a]} for a in ATTRS}},
              "required": ["id"] + ATTRS},
}


class GeminiTagger:
    def __init__(self):
        from google import genai
        from google.genai import types
        key = None
        if IN_COLAB:
            from google.colab import userdata
            for name in ["GOOGLE_API_KEY", "GEMINI_API_KEY"]:
                try:
                    key = userdata.get(name)
                    break
                except Exception:
                    continue
        key = key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Add your Gemini key as a Colab secret named GOOGLE_API_KEY (notebook access on).")
        self.client, self.types = genai.Client(api_key=key), types
        self.model = self._pick_model()

    def _pick_model(self):
        for name in MODELS_TO_TRY:
            try:
                self.client.models.generate_content(model=name, contents="Reply with OK")
                print("Using model:", name)
                return name
            except Exception as e:
                print(f"{name} not available ({str(e)[:80]}), trying next...")
        raise RuntimeError("None of MODELS_TO_TRY worked. Check https://ai.google.dev/gemini-api/docs/models")

    def __call__(self, prompt):
        cfg = self.types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                               response_schema=RESPONSE_SCHEMA)
        for attempt in range(6):
            try:
                t0 = time.time()
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                u = resp.usage_metadata
                return resp.text, {"latency_s": time.time() - t0,
                                   "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
                                   "output_tokens": getattr(u, "candidates_token_count", 0) or 0}
            except Exception as e:
                wait = 30 * (attempt + 1) if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else 5 * (attempt + 1)
                print(f"  API error ({str(e)[:70]}), retrying in {wait}s")
                time.sleep(wait)
        raise RuntimeError("Gemini call failed after retries")


class FakeTagger:
    """DEMO_MODE stand-in: keyword rules with some noise, returning the same JSON shape as Gemini."""
    def __call__(self, prompt):
        items = json.loads(prompt.split("Products:\n", 1)[1].split("\n", 1)[0])
        r = np.random.default_rng(len(prompt))
        out = []
        for it in items:
            t = it["title"]
            rec = {"id": it["id"]}
            for a in ATTRS:
                hits = [v for v in ALLOWED[a] if v.lower() in t.lower()]
                rec[a] = max(hits, key=len) if hits else r.choice(ALLOWED[a])
                if r.random() < 0.08:
                    rec[a] = r.choice(ALLOWED[a])
            out.append(rec)
        return json.dumps(out), {"latency_s": 0.01, "input_tokens": len(prompt) // 4, "output_tokens": 30 * len(items)}


def tag_with_llm(llm, frame, mode, examples=None, use_cache=True):
    cache = os.path.join(RESULTS_DIR, f"llm_raw_{mode}_{len(frame)}{'_demo' if DEMO_MODE else ''}.jsonl")
    done = {}
    if not use_cache and os.path.exists(cache):
        os.remove(cache)
    if os.path.exists(cache):
        for line in open(cache):
            rec = json.loads(line)
            done[rec["batch"]] = rec
    items = list(zip(range(len(frame)), frame["title"]))
    batches = [items[i:i + BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    with open(cache, "a") as fh:
        for b, batch in enumerate(batches):
            if b in done:
                continue
            text, usage = llm(build_prompt(batch, examples))
            rec = {"batch": b, "text": text, **usage}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            done[b] = rec
            print(f"  {mode}: batch {b + 1}/{len(batches)} done")
            if not DEMO_MODE and b < len(batches) - 1:
                time.sleep(SECONDS_BETWEEN_CALLS)

    preds, invalid = {}, 0
    for rec in done.values():
        try:
            for p in json.loads(rec["text"]):
                preds[int(p["id"])] = {a: (p.get(a) if p.get(a) in ALLOWED[a] else None) for a in ATTRS}
                invalid += sum(p.get(a) not in ALLOWED[a] for a in ATTRS)
        except (json.JSONDecodeError, TypeError, KeyError):
            invalid += BATCH_SIZE * len(ATTRS)
    pred = pd.DataFrame([preds.get(i, {a: None for a in ATTRS}) for i in range(len(frame))])
    stats = pd.DataFrame(list(done.values()))[["latency_s", "input_tokens", "output_tokens"]].sum()
    stats["calls"], stats["missing_or_invalid"] = len(done), invalid + sum(i not in preds for i in range(len(frame))) * len(ATTRS)
    return pred, stats


llm = FakeTagger() if DEMO_MODE else GeminiTagger()
few_shot_examples = (train.groupby("articleType").sample(n=1, random_state=SEED)    # one per type, varied
                          .sample(frac=1, random_state=SEED).head(N_FEW_SHOT))

pred_zero, stats_zero = tag_with_llm(llm, test, "zero_shot")
pred_few, stats_few = tag_with_llm(llm, test, "few_shot", examples=few_shot_examples)
pd.DataFrame({"Zero-shot": stats_zero, "Few-shot": stats_few}).round(1)

# %% [markdown]
# ## 7. Results

# %%
METHODS = {"TF-IDF + LogReg (all labels)": pred_baseline, "Gemini zero-shot": pred_zero, "Gemini few-shot": pred_few}

def score(pred):
    row = {}
    for a in ATTRS:
        y, p = test[a].values, pred[a].fillna("__missing__").values
        row[(a, "Accuracy")] = accuracy_score(y, p)
        row[(a, "Macro-F1")] = f1_score(y, p, average="macro", labels=ALLOWED[a], zero_division=0)
    row[("All 4 correct", "Accuracy")] = np.mean([(pred.loc[i, ATTRS].values == test.loc[i, ATTRS].values).all()
                                                  for i in range(len(test))])
    return row

results = pd.DataFrame({m: score(p) for m, p in METHODS.items()}).T
results.columns = pd.MultiIndex.from_tuples(results.columns)
results.to_csv(f"{RESULTS_DIR}/results_by_method.csv")
results.style.format("{:.1%}")

# %%
acc = results.xs("Accuracy", axis=1, level=1)
ax = acc.T.plot.bar(figsize=(11, 4.5), color=["#9aa5b1", "#7fb3f5", "#2a6fdb"], width=0.8, rot=0)
ax.set_ylim(0, 1.08); ax.set_ylabel(f"Accuracy on {len(test)} test products")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
for c in ax.containers:
    ax.bar_label(c, labels=[f"{v:.0%}" for v in c.datavalues], fontsize=8, padding=2)
ax.set_title("Attribute accuracy by method")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/accuracy_by_method.png", dpi=120); plt.show()

# %% [markdown]
# ## 8. How much labelled data does the baseline need?
# The LLM needs zero (or 12) labelled examples. The baseline is retrained on increasing amounts of labelled data
# to find where it catches up.

# %%
sizes = [s for s in [25, 50, 100, 200, 500, 1000, 2000, 5000] if s < len(train)] + [len(train)]
curve = []
for n in sizes:
    p = predict_baseline(train_baseline(train, n), test["title"])
    curve.append({"labelled_titles": n, **{a: accuracy_score(test[a], p[a]) for a in ATTRS}})
curve = pd.DataFrame(curve)
curve.to_csv(f"{RESULTS_DIR}/learning_curve.csv", index=False)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for ax, a in zip(axes, ATTRS):
    ax.plot(curve["labelled_titles"], curve[a], marker="o", color="#5b6b7f", label="TF-IDF + LogReg")
    ax.axhline(acc.loc["Gemini zero-shot", a], color="#7fb3f5", ls="--", label="Gemini zero-shot")
    ax.axhline(acc.loc["Gemini few-shot", a], color="#2a6fdb", ls="-", label="Gemini few-shot")
    ax.set_xscale("log"); ax.set_title(a); ax.set_xlabel("labelled training titles (log)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
axes[0].set_ylabel("Accuracy"); axes[0].legend(fontsize=8, loc="lower right")
plt.suptitle("Labelled data the baseline needs to match the LLM", y=1.03)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/learning_curve.png", dpi=120, bbox_inches="tight"); plt.show()

def breakeven(a, method="Gemini few-shot"):
    hit = curve[curve[a] >= acc.loc[method, a]]
    return int(hit["labelled_titles"].iloc[0]) if len(hit) else None

print("Labelled titles the baseline needs to match Gemini few-shot:")
for a in ATTRS:
    b = breakeven(a)
    print(f"  {a:12s} {b:>7,}" if b else f"  {a:12s}  never (LLM stays ahead)")

# %% [markdown]
# ## 9. Error analysis
# Where does the LLM go wrong, and is it the model or the labels?

# %%
best_llm = "Gemini few-shot" if acc.loc["Gemini few-shot"].mean() >= acc.loc["Gemini zero-shot"].mean() else "Gemini zero-shot"
p = METHODS[best_llm]
errors = test[["title"] + ATTRS].copy()
for a in ATTRS:
    errors[f"pred_{a}"] = p[a]
wrong = errors[(errors["usage"] != errors["pred_usage"]) | (errors["articleType"] != errors["pred_articleType"])]
wrong.to_csv(f"{RESULTS_DIR}/llm_errors.csv", index=False)
print(f"{best_llm}: {len(wrong)} of {len(test)} products have a wrong usage or articleType")
wrong.sample(min(12, len(wrong)), random_state=SEED)[["title", "articleType", "pred_articleType", "usage", "pred_usage"]]

# %%
labels = ALLOWED["usage"]
cm = confusion_matrix(test["usage"], p["usage"].fillna("__missing__"), labels=labels)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel(f"Predicted ({best_llm})"); ax.set_ylabel("Catalogue label"); ax.set_title("Usage: where the mistakes are")
ax.grid(False)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/usage_confusion.png", dpi=120); plt.show()

# %% [markdown]
# ## 10. Tag new products
# The same function can tag titles that have no labels, such as new listings.

# %%
new_titles = ["Roadster Men Olive Green Slim Fit Cargo Trousers", "Biba Women Mustard Printed Anarkali Kurta",
              "Nike Unisex Black Running Shoes", "Fossil Women Rose Gold Analogue Watch",
              "HRX Men Grey Dry-Fit Training T-shirt"]
new = pd.DataFrame({"title": new_titles})
tagged, _ = tag_with_llm(llm, new, "new_products", examples=few_shot_examples, use_cache=False)
pd.concat([new, tagged], axis=1)

# %% [markdown]
# ## 11. Summary
# Fill this in from your own run:
#
# - **Accuracy:** Gemini few-shot ___% / zero-shot ___% vs TF-IDF baseline ___% ("all 4 correct": ___% vs ___%).
# - **Labelled data:** the baseline needed about ___ labelled titles to match the LLM on ___.
# - **Hardest attribute:** ___. Many "errors" are ___ (e.g. ambiguous or noisy catalogue labels).
# - **Cost and speed:** ___ API calls, ___ tokens, ___ s total for 300 products.
# - **Recommendation:** use the LLM for ___ (cold start, new categories), and a trained model for ___.
