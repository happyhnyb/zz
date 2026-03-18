"""
LIME Stability Explorer — Streamlit Demo
=========================================
Interactive demo for the LIME stability thesis project.

Tabs
----
1. Overview       — project summary and design
2. Run Experiment — configure and run a single experiment live
3. Stability View — visualise results from a completed run
4. Compare Models — side-by-side model stability comparison
5. Correlation Lab — inject correlation and watch stability degrade

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import random
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from scipy import stats as scipy_stats
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils import resample

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LIME Stability Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colours ───────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#2196F3",
    "Random Forest":       "#4CAF50",
    "MLP":                 "#FF5722",
}

# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (self-contained — no imports from project modules)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading Breast Cancer dataset…")
def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = pd.Series((data.target == 0).astype(int), name="target")
    return X, y, data.feature_names.tolist()


@st.cache_data(show_spinner="Loading Adult Income dataset…")
def load_adult_income_data():
    try:
        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame.copy()
        y = (df["class"].astype(str).str.strip()
             .str.replace(".", "", regex=False)
             .map({"<=50K": 0, ">50K": 1}))
        valid = y.notna()
        X = df.drop(columns=["class"])[valid].reset_index(drop=True)
        y = y[valid].astype(int).reset_index(drop=True)
        return X, y
    except Exception as e:
        return None, None


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = [
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), num_cols),
    ]
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    return ColumnTransformer(transformers, remainder="drop",
                             verbose_feature_names_out=True)


def build_model(name: str, seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    else:
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=seed)


def add_noise(X: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    if sigma == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    for col in Xn.select_dtypes(include=[np.number]).columns:
        std = float(Xn[col].std(ddof=0))
        if std > 0:
            Xn[col] = Xn[col].astype(float) + rng.normal(0, sigma * std, len(Xn))
    return Xn


def apply_imbalance(X: pd.DataFrame, y: pd.Series,
                    maj_ratio: float, seed: int):
    vc = y.value_counts()
    maj, min_ = vc.idxmax(), vc.idxmin()
    X_maj, y_maj = X[y == maj], y[y == maj]
    X_min, y_min = X[y == min_], y[y == min_]
    n_maj = len(X_maj)
    desired = max(10, min(int(round(n_maj * (1 - maj_ratio) / maj_ratio)), len(X_min)))
    Xm, ym = resample(X_min, y_min, replace=False, n_samples=desired, random_state=seed)
    Xo = pd.concat([X_maj, Xm]).sample(frac=1, random_state=seed)
    yo = y.loc[Xo.index]
    return Xo.reset_index(drop=True), yo.reset_index(drop=True)


def inject_correlation(X: pd.DataFrame, rho: float, seed: int = 42):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    n_feat = min(4, len(num_cols) - len(num_cols) % 2)
    if n_feat < 2:
        return X.copy(), {}
    selected = X[num_cols].var(ddof=0).nlargest(n_feat).index.tolist()
    vals = X[selected].values.astype(float)
    N = len(vals)
    # Rank transform
    Z = np.empty_like(vals)
    for j in range(vals.shape[1]):
        ranks = scipy_stats.rankdata(vals[:, j], method="average")
        Z[:, j] = scipy_stats.norm.ppf(ranks / (N + 1))
    # Cholesky
    Sigma = np.eye(n_feat)
    for k in range(0, n_feat, 2):
        Sigma[k, k+1] = Sigma[k+1, k] = rho
    L = np.linalg.cholesky(Sigma)
    Z_corr = Z @ L.T
    # Inverse transform
    Xo = X.copy()
    for i, col in enumerate(selected):
        orig = np.sort(X[col].values.astype(float))
        u = scipy_stats.norm.cdf(Z_corr[:, i])
        idx = np.clip((u * (N - 1)).astype(int), 0, N - 1)
        Xo[col] = orig[idx]
    achieved = {}
    for k in range(0, n_feat, 2):
        f1, f2 = selected[k], selected[k+1]
        r = float(np.corrcoef(Xo[f1], Xo[f2])[0, 1])
        achieved[f"{f1[:12]}↔{f2[:12]}"] = round(r, 3)
    return Xo, achieved


def resolve_name(name: str, cat_cols: list) -> str:
    if "__" in name:
        name = name.split("__", 1)[1]
    for c in cat_cols:
        if name.startswith(c + "_") or name.startswith(c + "=") or name == c:
            return c
    for op in [" <= ", " > ", " < ", " >= ", "="]:
        if op in name:
            return name.split(op)[0].strip()
    return name.strip()


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def spearman_sim(ra: dict, rb: dict, k: int) -> float:
    feats = sorted(set(ra) | set(rb))
    if len(feats) < 2:
        return 1.0
    a = [ra.get(f, k+1) for f in feats]
    b = [rb.get(f, k+1) for f in feats]
    if a == b:
        return 1.0
    c, _ = spearmanr(a, b)
    return 0.0 if np.isnan(c) else float(c)


def to_rank(exp):
    s = sorted(exp, key=lambda x: abs(x[1]), reverse=True)
    return {n: i+1 for i, (n, _) in enumerate(s)}


def run_lime_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_runs: int,
    top_k: int,
    n_instances: int,
    num_samples: int,
    seed: int,
    variation: str,
    variation_value: float,
    progress_bar=None,
):
    """Run one complete LIME stability experiment and return results dict."""

    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Apply variation
    corr_meta = {}
    if variation == "noise":
        X = add_noise(X, variation_value, seed)
    elif variation == "imbalance":
        try:
            X, y = apply_imbalance(X, y, variation_value, seed)
        except Exception:
            pass
    elif variation == "correlation":
        X, corr_meta = inject_correlation(X, variation_value, seed)

    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed)

    # Build + fit pipeline
    prep = build_preprocessor(X_tr)
    clf  = build_model(model_name, seed)
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    # Metrics
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]
    perf = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "f1":       round(f1_score(y_te, y_pred, zero_division=0), 4),
        "auc":      round(roc_auc_score(y_te, y_proba), 4),
    }

    # Transform for LIME
    Xtr_t = prep.transform(X_tr)
    Xte_t = prep.transform(X_te)
    fnames = prep.get_feature_names_out().tolist()
    cat_idx = [i for i, n in enumerate(fnames) if n.startswith("cat__")]

    explainer = LimeTabularExplainer(
        training_data=Xtr_t,
        feature_names=fnames,
        class_names=["class_0", "class_1"],
        categorical_features=cat_idx if cat_idx else None,
        mode="classification",
        discretize_continuous=True,
        discretizer="quartile",
        random_state=seed,
    )
    predict_fn = pipe.named_steps["clf"].predict_proba

    # Select instances spread across confidence levels
    n_inst = min(n_instances, len(Xte_t))
    sorted_idx = np.argsort(y_proba)
    band = max(1, len(sorted_idx) // 3)
    pools = [sorted_idx[:band], sorted_idx[band:2*band], sorted_idx[2*band:]]
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(p, size=min(n_inst//3 + (1 if i < n_inst%3 else 0), len(p)), replace=False)
        for i, p in enumerate(pools)
    ])[:n_inst]

    instance_results = []
    total = len(chosen)

    for step, idx in enumerate(chosen):
        if progress_bar:
            progress_bar.progress((step + 1) / total,
                                  text=f"Explaining instance {step+1}/{total}…")

        instance = Xte_t[int(idx)]
        label    = int(pipe.predict(X_te.iloc[[int(idx)]])[0])

        exps = []
        for run in range(n_runs):
            s = int(seed) + int(idx) * 100 + run
            np.random.seed(s); random.seed(s)
            e = explainer.explain_instance(
                instance, predict_fn, num_features=top_k,
                num_samples=num_samples, labels=(label,))
            raw = [(resolve_name(n, cat_cols), float(w))
                   for n, w in e.as_list(label=label)]
            raw.sort(key=lambda x: abs(x[1]), reverse=True)
            exps.append(raw[:top_k])

        jacs, spes = [], []
        for ea, eb in combinations(exps, 2):
            sa, sb = {n for n,_ in ea}, {n for n,_ in eb}
            ra, rb = to_rank(ea), to_rank(eb)
            jacs.append(jaccard(sa, sb))
            spes.append(spearman_sim(ra, rb, top_k))

        # Swap rate for correlation
        swap_rate = 0.0
        if corr_meta and variation == "correlation":
            pair_key = list(corr_meta.keys())[0]
            f1_name, f2_name = pair_key.split("↔")
            f1_name, f2_name = f1_name.strip(), f2_name.strip()
            total_p, swap_p = 0, 0
            for ea, eb in combinations(exps, 2):
                sa = {n for n,_ in ea}; sb = {n for n,_ in eb}
                # partial match on truncated names
                a1 = any(f1_name in n for n in sa)
                a2 = any(f2_name in n for n in sa)
                b1 = any(f1_name in n for n in sb)
                b2 = any(f2_name in n for n in sb)
                if a1 or a2 or b1 or b2:
                    total_p += 1
                    if (a1 and not a2 and b2 and not b1) or \
                       (a2 and not a1 and b1 and not b2):
                        swap_p += 1
            swap_rate = swap_p / total_p if total_p > 0 else 0.0

        # Top features across all runs (frequency count)
        feat_counts: dict[str, int] = {}
        for exp in exps:
            for fn, _ in exp:
                feat_counts[fn] = feat_counts.get(fn, 0) + 1

        instance_results.append({
            "idx":          int(idx),
            "label":        label,
            "mean_jaccard": float(np.mean(jacs)) if jacs else 0.0,
            "std_jaccard":  float(np.std(jacs, ddof=1)) if len(jacs) > 1 else 0.0,
            "mean_spearman":float(np.mean(spes)) if spes else 0.0,
            "swap_rate":    swap_rate,
            "feat_counts":  feat_counts,
            "all_exps":     exps,
        })

    return {
        "model":        model_name,
        "variation":    variation,
        "var_value":    variation_value,
        "performance":  perf,
        "instances":    instance_results,
        "corr_meta":    corr_meta,
        "mean_jaccard": float(np.mean([r["mean_jaccard"] for r in instance_results])),
        "mean_spearman":float(np.mean([r["mean_spearman"] for r in instance_results])),
        "n_runs":       n_runs,
        "top_k":        top_k,
    }


def classify_trust(jac, spe, auc, jt=0.7, st=0.7, at=0.75):
    stable = jac >= jt and spe >= st
    if stable and auc < at:  return "⚠️ Misleading"
    if stable:                return "✅ Trustworthy"
    if (jac >= 0.5 or spe >= 0.5) and auc >= at:
        return "🟡 Conditionally trustworthy"
    return "❌ Unreliable"


# ══════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_stability_per_instance(results: dict):
    instances = results["instances"]
    idxs  = [r["idx"] for r in instances]
    jacs  = [r["mean_jaccard"]  for r in instances]
    spes  = [r["mean_spearman"] for r in instances]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"{results['model']} — {results['variation']} "
        f"(level={results['var_value']}) — Per-Instance Stability",
        fontsize=11, fontweight="bold")

    for ax, vals, label, color in [
        (axes[0], jacs, "Jaccard Similarity",    "#2196F3"),
        (axes[1], spes, "Spearman Correlation",  "#4CAF50"),
    ]:
        ax.bar(range(len(vals)), vals, color=color, alpha=0.75, width=0.6)
        ax.axhline(np.mean(vals), color="black", ls="--", lw=1.5,
                   label=f"Mean = {np.mean(vals):.3f}")
        ax.axhline(0.7, color="red", ls=":", lw=1.2, label="Threshold (0.7)")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Instance Index", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(idxs, rotation=45, fontsize=7)

    plt.tight_layout()
    return fig_to_buf(fig)


def plot_feature_frequency(results: dict, instance_idx: int = 0):
    r = results["instances"][instance_idx]
    counts = r["feat_counts"]
    n_runs = results["n_runs"]
    top_n  = 10

    sorted_feats = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats, cnts  = zip(*sorted_feats) if sorted_feats else ([], [])
    freqs = [c / n_runs for c in cnts]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2196F3" if f >= 0.7 else "#FF9800" if f >= 0.4 else "#F44336"
              for f in freqs]
    bars = ax.barh(range(len(feats)), freqs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([f[:35] for f in feats], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction of LIME runs containing feature", fontsize=9)
    ax.set_title(f"Feature Consistency — Instance {r['idx']} "
                 f"(Jaccard={r['mean_jaccard']:.3f})", fontsize=10, fontweight="bold")
    ax.axvline(0.7, color="red",  ls=":", lw=1.2, label="0.7 threshold")
    ax.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax.set_xlim(0, 1.1)
    ax.legend(fontsize=8)

    for bar, freq in zip(bars, freqs):
        ax.text(freq + 0.02, bar.get_y() + bar.get_height()/2,
                f"{freq:.2f}", va="center", fontsize=7)
    plt.tight_layout()
    return fig_to_buf(fig)


def plot_comparison_bars(results_list: list):
    labels = [f"{r['model']}\n{r['variation']}={r['var_value']}" for r in results_list]
    jacs   = [r["mean_jaccard"]  for r in results_list]
    spes   = [r["mean_spearman"] for r in results_list]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.5), 5))
    w = 0.35
    b1 = ax.bar(x - w/2, jacs, w, label="Jaccard",  alpha=0.85,
                color=[MODEL_COLORS.get(r["model"], "#888") for r in results_list])
    b2 = ax.bar(x + w/2, spes, w, label="Spearman", alpha=0.5,
                color=[MODEL_COLORS.get(r["model"], "#888") for r in results_list],
                hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0.7, color="red", ls=":", lw=1.2, label="Threshold (0.7)")
    ax.set_ylabel("Mean Stability Score", fontsize=10)
    ax.set_title("Model / Condition Comparison", fontsize=11, fontweight="bold")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h:.2f}", ha="center", fontsize=7)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_to_buf(fig)


def plot_heatmap_inline(results_list: list):
    rows = []
    for r in results_list:
        rows.append({
            "Condition": f"{r['variation']}={r['var_value']}",
            "Model":     r["model"],
            "Jaccard":   r["mean_jaccard"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    try:
        pivot = df.pivot_table(index="Condition", columns="Model",
                               values="Jaccard", aggfunc="mean")
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8, max(3, len(pivot)*0.6)))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0, vmax=1, linewidths=0.4,
                cbar_kws={"label": "Mean Jaccard"})
    ax.set_title("Stability Heatmap", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig_to_buf(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "results_store" not in st.session_state:
    st.session_state.results_store = []   # list of result dicts
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/200px-Scikit_learn_logo_small.svg.png", width=60)
    st.title("LIME Stability Explorer")
    st.caption("Master's Thesis Demo — Gamze YILMAZ, Gisma University 2026")
    st.divider()

    st.subheader("⚙️ Experiment Settings")

    dataset_choice = st.selectbox("Dataset", ["Breast Cancer", "Adult Income"])
    model_choice   = st.selectbox("Model",   ["Logistic Regression", "Random Forest", "MLP"])
    variation      = st.selectbox("Variation", ["baseline", "noise", "imbalance", "correlation"])

    var_value = 0.0
    if variation == "noise":
        var_value = st.slider("Noise level (σ)", 0.0, 0.5, 0.1, 0.05)
    elif variation == "imbalance":
        var_value = st.slider("Majority class fraction", 0.5, 0.95, 0.80, 0.05)
    elif variation == "correlation":
        var_value = st.slider("Target Pearson ρ", 0.0, 0.95, 0.80, 0.05)

    st.divider()
    st.subheader("🔬 LIME Parameters")
    n_runs      = st.slider("LIME runs per instance", 5, 30, 10)
    top_k       = st.slider("Top-k features",          3,  8,  5)
    n_instances = st.slider("Test instances",           3, 20,  8)
    num_samples = st.select_slider("Neighbourhood samples",
                                   options=[200, 500, 1000, 2000], value=500)
    seed        = st.number_input("Random seed", value=42, min_value=0)

    st.divider()
    run_btn = st.button("▶  Run Experiment", type="primary", use_container_width=True)
    clear_btn = st.button("🗑  Clear History", use_container_width=True)
    if clear_btn:
        st.session_state.results_store = []
        st.session_state.last_result   = None
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "🧪 Run & Results",
    "📊 Stability View",
    "⚖️ Compare",
    "🔗 Correlation Lab",
])

# ── Tab 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.title("🔍 LIME Explanation Stability Explorer")
    st.markdown("""
    This interactive demo accompanies the master's thesis:

    > **"Stability of LIME Explanations Under Controlled Dataset Variations:
    An Empirical Study Across Machine Learning Models"**
    > — Gamze YILMAZ, Gisma University of Applied Sciences, Winter 2026

    ---
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datasets", "2", "Adult Income + Breast Cancer")
    c2.metric("Models", "3", "LR · RF · MLP")
    c3.metric("Variations", "4", "Baseline · Noise · Imbalance · Correlation")
    c4.metric("Total Experiments", "180", "3 seeds × 30 instances × 20 runs")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Research Questions")
        st.markdown("""
        **RQ1** — How stable are LIME explanations when repeated on the same instance?

        **RQ2** — How does stability differ across LR, RF, and MLP models?

        **RQ3** — How do noise, class imbalance, and feature correlation affect stability?

        **RQ4** — Under what conditions can LIME be considered *trustworthy*?
        """)

        st.subheader("📐 Stability Metrics")
        st.markdown(
            "| Metric | Formula | Range |\n"
            "|--------|---------|-------|\n"
            "| **Jaccard** | A intersect B / A union B | [0, 1] |\n"
            "| **Spearman** | rank corr, penalty rank k+1 for absent features | [-1, 1] |\n"
            "| **Swap Rate** | fraction of run-pairs where correlated features swap | [0, 1] |\n"
        )

    with col2:
        st.subheader("🏷️ Trustworthiness Classification")
        trust_data = {
            "Label": ["✅ Trustworthy", "🟡 Conditionally trustworthy",
                      "❌ Unreliable", "⚠️ Misleading"],
            "Jaccard": ["≥ 0.70", "≥ 0.50", "< 0.50", "≥ 0.70"],
            "Spearman": ["≥ 0.70", "≥ 0.50", "any", "≥ 0.70"],
            "AUC": ["≥ 0.75", "≥ 0.75", "any", "< 0.75"],
        }
        st.dataframe(pd.DataFrame(trust_data), hide_index=True, use_container_width=True)

        st.subheader("⚙️ How to Use")
        st.markdown("""
        1. **Configure** an experiment in the sidebar
        2. Click **▶ Run Experiment**
        3. Explore results in **🧪 Run & Results**
        4. Compare multiple runs in **⚖️ Compare**
        5. Study correlation effects in **🔗 Correlation Lab**
        """)


# ── Tab 2: Run & Results ──────────────────────────────────────────────────────
with tab2:
    # Run experiment if button pressed
    if run_btn:
        with st.spinner("Loading dataset…"):
            if dataset_choice == "Breast Cancer":
                X, y, _ = load_breast_cancer_data()
            else:
                X, y = load_adult_income_data()
                if X is None:
                    st.error("Adult Income dataset requires internet access (OpenML). "
                             "Please use Breast Cancer instead.")
                    st.stop()

        pb = st.progress(0, text="Starting…")

        with st.spinner("Running LIME experiment…"):
            result = run_lime_experiment(
                X=X, y=y,
                model_name=model_choice,
                n_runs=n_runs,
                top_k=top_k,
                n_instances=n_instances,
                num_samples=num_samples,
                seed=int(seed),
                variation=variation,
                variation_value=var_value,
                progress_bar=pb,
            )

        pb.empty()
        st.session_state.last_result = result
        st.session_state.results_store.append(result)
        st.success(f"Experiment complete! {len(result['instances'])} instances explained.")

    result = st.session_state.last_result

    if result is None:
        st.info("👈 Configure an experiment in the sidebar and click **▶ Run Experiment**.")
    else:
        # KPI row
        perf = result["performance"]
        jac  = result["mean_jaccard"]
        spe  = result["mean_spearman"]
        trust = classify_trust(jac, spe, perf["auc"])

        st.subheader(f"Results — {result['model']} | {result['variation']} = {result['var_value']}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy",   f"{perf['accuracy']:.3f}")
        c2.metric("F1",         f"{perf['f1']:.3f}")
        c3.metric("AUC",        f"{perf['auc']:.3f}")
        c4.metric("Mean Jaccard",  f"{jac:.3f}",
                  delta=f"{jac - 0.7:+.3f} vs threshold",
                  delta_color="normal" if jac >= 0.7 else "inverse")
        c5.metric("Mean Spearman", f"{spe:.3f}",
                  delta=f"{spe - 0.7:+.3f} vs threshold",
                  delta_color="normal" if spe >= 0.7 else "inverse")
        c6.metric("Trustworthiness", trust)

        st.divider()

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Per-Instance Stability")
            buf = plot_stability_per_instance(result)
            st.image(buf, use_container_width=True)

        with col2:
            st.subheader("Instance Detail")
            instances = result["instances"]
            inst_labels = [f"Instance {r['idx']} (Jac={r['mean_jaccard']:.2f})"
                           for r in instances]
            sel = st.selectbox("Select instance", inst_labels)
            sel_idx = inst_labels.index(sel)
            r = instances[sel_idx]

            st.markdown(f"""
            - **Predicted label:** `{r['label']}`
            - **Mean Jaccard:** `{r['mean_jaccard']:.4f}`
            - **Std Jaccard:** `{r['std_jaccard']:.4f}`
            - **Mean Spearman:** `{r['mean_spearman']:.4f}`
            - **Swap Rate:** `{r['swap_rate']:.4f}`
            """)

            buf2 = plot_feature_frequency(result, sel_idx)
            st.image(buf2, use_container_width=True)

        st.divider()

        # Raw data table
        with st.expander("📄 Raw instance results table"):
            rows = [{
                "Instance": r["idx"],
                "Pred Label": r["label"],
                "Mean Jaccard": round(r["mean_jaccard"], 4),
                "Std Jaccard": round(r["std_jaccard"], 4),
                "Mean Spearman": round(r["mean_spearman"], 4),
                "Swap Rate": round(r["swap_rate"], 4),
            } for r in result["instances"]]
            df_raw = pd.DataFrame(rows)
            st.dataframe(df_raw, use_container_width=True, hide_index=True)

            csv = df_raw.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv,
                               file_name="lime_stability_results.csv",
                               mime="text/csv")


# ── Tab 3: Stability View ─────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Detailed Stability Analysis")

    result = st.session_state.last_result
    if result is None:
        st.info("Run an experiment first.")
    else:
        instances = result["instances"]

        col1, col2 = st.columns(2)

        with col1:
            # Distribution of per-instance Jaccard
            jacs = [r["mean_jaccard"] for r in instances]
            spes = [r["mean_spearman"] for r in instances]

            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, vals, label, color in [
                (axes[0], jacs, "Jaccard Similarity",   "#2196F3"),
                (axes[1], spes, "Spearman Correlation", "#4CAF50"),
            ]:
                ax.hist(vals, bins=min(10, len(vals)), color=color, alpha=0.75,
                        edgecolor="white")
                ax.axvline(np.mean(vals), color="black", ls="--",
                           label=f"Mean={np.mean(vals):.3f}")
                ax.axvline(0.7, color="red", ls=":", label="Threshold")
                ax.set_xlabel(label, fontsize=9)
                ax.set_ylabel("Instances", fontsize=9)
                ax.set_title(f"Distribution of {label}", fontsize=10)
                ax.legend(fontsize=8)
            plt.tight_layout()
            st.image(fig_to_buf(fig), use_container_width=True)

        with col2:
            # Scatter: Jaccard vs Spearman
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(jacs, spes, c="#2196F3", alpha=0.7, s=60, edgecolors="white")
            ax.axhline(0.7, color="red",  ls=":", lw=1)
            ax.axvline(0.7, color="red",  ls=":", lw=1)
            ax.set_xlabel("Mean Jaccard Similarity", fontsize=10)
            ax.set_ylabel("Mean Spearman Correlation", fontsize=10)
            ax.set_title("Jaccard vs. Spearman per Instance", fontsize=10, fontweight="bold")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            # Quadrant labels
            ax.text(0.85, 0.85, "Stable", transform=ax.transAxes,
                    color="green", fontsize=8, ha="center")
            ax.text(0.15, 0.15, "Unstable", transform=ax.transAxes,
                    color="red", fontsize=8, ha="center")
            plt.tight_layout()
            st.image(fig_to_buf(fig), use_container_width=True)

        # All runs for one instance
        st.divider()
        st.subheader("Explanation Runs for Selected Instance")
        inst_labels = [f"Instance {r['idx']}" for r in instances]
        sel = st.selectbox("Instance", inst_labels, key="stab_inst")
        sel_idx = inst_labels.index(sel)
        r = instances[sel_idx]

        all_exps = r["all_exps"]
        n_show   = min(len(all_exps), 10)

        feat_matrix = []
        all_feats = sorted({fn for exp in all_exps for fn, _ in exp})

        for i, exp in enumerate(all_exps[:n_show]):
            fd = {fn: w for fn, w in exp}
            row = {"Run": i+1}
            for f in all_feats:
                row[f[:20]] = round(fd.get(f, 0.0), 4)
            feat_matrix.append(row)

        df_runs = pd.DataFrame(feat_matrix).set_index("Run")
        st.caption(f"LIME weight matrix — first {n_show} of {len(all_exps)} runs "
                   f"(top {result['top_k']} features per run, 0 = absent)")
        st.dataframe(df_runs.style.background_gradient(cmap="RdBu_r", axis=None,
                                                        vmin=-0.3, vmax=0.3),
                     use_container_width=True)


# ── Tab 4: Compare ────────────────────────────────────────────────────────────
with tab4:
    st.subheader("⚖️ Multi-Run Comparison")

    store = st.session_state.results_store
    if len(store) == 0:
        st.info("Run at least one experiment first.")
    elif len(store) == 1:
        st.info("Run more experiments to compare. Use the sidebar to change model/variation.")
    else:
        st.caption(f"{len(store)} experiments in history.")

        # Comparison bar chart
        buf = plot_comparison_bars(store)
        st.image(buf, use_container_width=True)

        # Heatmap
        buf2 = plot_heatmap_inline(store)
        if buf2:
            st.image(buf2, use_container_width=True)

        # Summary table
        st.divider()
        rows = []
        for r in store:
            trust = classify_trust(r["mean_jaccard"], r["mean_spearman"],
                                   r["performance"]["auc"])
            rows.append({
                "Model":        r["model"],
                "Variation":    r["variation"],
                "Level":        r["var_value"],
                "AUC":          round(r["performance"]["auc"], 3),
                "F1":           round(r["performance"]["f1"], 3),
                "Mean Jaccard": round(r["mean_jaccard"], 3),
                "Mean Spearman":round(r["mean_spearman"], 3),
                "Trustworthiness": trust,
            })
        df_comp = pd.DataFrame(rows)

        def color_trust(val):
            if "Trustworthy" in val and "Conditionally" not in val:
                return "background-color: #d4edda"
            elif "Conditionally" in val:
                return "background-color: #fff3cd"
            elif "Unreliable" in val:
                return "background-color: #f8d7da"
            else:
                return "background-color: #e2d9f3"

        st.dataframe(
            df_comp.style.applymap(color_trust, subset=["Trustworthiness"]),
            use_container_width=True, hide_index=True,
        )

        csv_comp = df_comp.to_csv(index=False)
        st.download_button("⬇ Download comparison CSV", csv_comp,
                           file_name="lime_comparison.csv", mime="text/csv")


# ── Tab 5: Correlation Lab ────────────────────────────────────────────────────
with tab5:
    st.subheader("🔗 Feature Correlation Lab")
    st.markdown("""
    See how injecting correlation between feature pairs degrades LIME stability.
    The **swap rate** measures how often LIME attributes importance to different
    members of a correlated pair across runs.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        rho_levels = st.multiselect(
            "ρ levels to test",
            options=[0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            default=[0.0, 0.5, 0.8, 0.95],
        )
        corr_model = st.selectbox("Model", ["Logistic Regression", "Random Forest", "MLP"],
                                  key="corr_model")
        corr_runs  = st.slider("LIME runs", 5, 20, 8, key="corr_runs")
        corr_insts = st.slider("Instances", 3, 10, 5, key="corr_insts")
        corr_btn   = st.button("▶ Run Correlation Sweep", type="primary",
                               use_container_width=True)

    with col2:
        if "corr_sweep" not in st.session_state:
            st.session_state.corr_sweep = []

        if corr_btn and len(rho_levels) > 0:
            X_bc, y_bc, _ = load_breast_cancer_data()
            sweep_results = []
            prog = st.progress(0, text="Running correlation sweep…")

            for i, rho in enumerate(sorted(rho_levels)):
                prog.progress((i+1)/len(rho_levels), text=f"Testing ρ = {rho}…")
                r = run_lime_experiment(
                    X=X_bc, y=y_bc,
                    model_name=corr_model,
                    n_runs=corr_runs,
                    top_k=5,
                    n_instances=corr_insts,
                    num_samples=500,
                    seed=42,
                    variation="correlation" if rho > 0 else "baseline",
                    variation_value=rho,
                )
                sweep_results.append({
                    "rho":          rho,
                    "mean_jaccard": r["mean_jaccard"],
                    "mean_spearman":r["mean_spearman"],
                    "mean_swap":    np.mean([i["swap_rate"] for i in r["instances"]]),
                    "auc":          r["performance"]["auc"],
                })
            prog.empty()
            st.session_state.corr_sweep = sweep_results

        sweep = st.session_state.corr_sweep

        if sweep:
            df_sw = pd.DataFrame(sweep)
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            ax1.plot(df_sw["rho"], df_sw["mean_jaccard"],  "o-",
                     color="#2196F3", lw=2, ms=7, label="Mean Jaccard")
            ax1.plot(df_sw["rho"], df_sw["mean_spearman"], "s--",
                     color="#4CAF50", lw=2, ms=7, label="Mean Spearman")
            ax2.plot(df_sw["rho"], df_sw["mean_swap"],     "^:",
                     color="#FF5722", lw=2, ms=7, label="Swap Rate")

            ax1.axhline(0.7, color="red", ls=":", lw=1, alpha=0.6)
            ax1.set_xlabel("Target Pearson ρ", fontsize=11)
            ax1.set_ylabel("Stability Score", fontsize=11)
            ax2.set_ylabel("Swap Rate", fontsize=11, color="#FF5722")
            ax1.set_ylim(-0.1, 1.1)
            ax2.set_ylim(0, 1.0)
            ax1.set_title(
                f"Correlation Sweep — {corr_model} on Breast Cancer",
                fontsize=12, fontweight="bold")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

            plt.tight_layout()
            st.image(fig_to_buf(fig), use_container_width=True)

            st.dataframe(df_sw.round(4), use_container_width=True, hide_index=True)

            if len(df_sw) >= 2:
                baseline_jac = float(df_sw[df_sw["rho"] == 0.0]["mean_jaccard"].values[0]) \
                    if 0.0 in df_sw["rho"].values else float(df_sw.iloc[0]["mean_jaccard"])
                max_rho = float(df_sw.iloc[-1]["rho"])
                max_jac = float(df_sw.iloc[-1]["mean_jaccard"])
                max_swap = float(df_sw.iloc[-1]["mean_swap"])
                drop = baseline_jac - max_jac

                st.info(
                    f"**Finding:** At ρ = {max_rho}, mean Jaccard drops to **{max_jac:.3f}** "
                    f"(from {baseline_jac:.3f} at baseline, Δ = {drop:+.3f}). "
                    f"Swap rate reaches **{max_swap:.3f}**, indicating LIME is "
                    f"{'frequently' if max_swap > 0.3 else 'occasionally'} "
                    f"swapping attribution between correlated feature pairs."
                )
        else:
            st.info("Click **▶ Run Correlation Sweep** to see how correlation degrades stability.")
