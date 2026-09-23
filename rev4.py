"""
================================================================================
reviewer4_analysis.py — ADD-ON for the second revision round (Reviewer 4)
================================================================================

Pure post-processing. Fits NO models and changes NO reported value. It reads the
prediction files written by the patched reproduce_all.py and produces the
statistical support requested by Reviewer 4 and by Reviewer 1.

Prerequisite
------------
  1) apply the three small patches to reproduce_all.py (prediction capture)
  2) python reproduce_all.py        -> writes results/predictions_models.csv
                                                results/predictions_normalization.csv
  3) python reviewer4_analysis.py   -> writes results/reviewer4_supplementary.xlsx

Runtime: 2-4 minutes. No model fitting, no tuning, no re-training.

Output sheets
-------------
  S1a  Paired bootstrap: DeltaMAE of every Table-5 configuration vs. CatBoost
       -> R4-2. Source of the single new column added to Table 5.
  S1b  Paired bootstrap on squared-error differences (boosting models)
       -> R4-2, second metric explicitly requested.
  S2   Paired bootstrap across normalization strategies, common TL units
       -> R4-2. Reported in text, not as a new table.
  S3   Segment error AND SIGNED BIAS: seller type, fuel type, age band, quintile
       -> R4-3. Detects systematic over/undervaluation.
  S4   Fallback split: new series of a known brand vs. entirely new brand
       -> R4-1. Already Table 11; here with bias and bootstrap CIs.
  S5   Bootstrap CIs for the price-decomposition components
       -> R4-2, third part ("the same analysis for the decomposition").
================================================================================
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error,
)

from reproduce_all import CONFIG, preprocess

N_BOOT = 2000
BOOT_SEED = 20260713
OUT_XLSX = "reviewer4_supplementary.xlsx"
REF_MODEL = "CatBoost"
REF_STRATEGY = "USD/TRY"


# ============================================================================
# Bootstrap helpers
# ============================================================================

def _boot_indices(n):
    """Deterministic resample indices. The generator is re-seeded on every call,
    so all comparisons below share identical bootstrap replicates."""
    rng = np.random.default_rng(BOOT_SEED)
    for _ in range(N_BOOT):
        yield rng.integers(0, n, n)


def paired_bootstrap(y_true, pred_a, pred_b, metric="mae"):
    """Paired bootstrap of the per-observation loss difference (a minus b).

    Negative -> model a has the lower error. A 95% interval containing zero
    means the two are statistically indistinguishable on this test set.
    """
    y_true = np.asarray(y_true, float)
    pred_a = np.asarray(pred_a, float)
    pred_b = np.asarray(pred_b, float)

    if metric == "mae":
        d = np.abs(pred_a - y_true) - np.abs(pred_b - y_true)
    elif metric == "sq":
        d = (pred_a - y_true) ** 2 - (pred_b - y_true) ** 2
    else:
        raise ValueError(metric)

    boot = np.fromiter((d[i].mean() for i in _boot_indices(len(d))),
                       dtype=float, count=N_BOOT)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)


def bootstrap_mean(x):
    """One-sample bootstrap CI for a mean (segment bias)."""
    x = np.asarray(x, float)
    boot = np.fromiter((x[i].mean() for i in _boot_indices(len(x))),
                       dtype=float, count=N_BOOT)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return x.mean(), lo, hi


def _ci(lo, hi, dec=1):
    return f"[{lo:.{dec}f}, {hi:.{dec}f}]"


def _load(fname):
    path = os.path.join(CONFIG["output_dir"], fname)
    if not os.path.exists(path):
        raise SystemExit(
            f"\nMissing {path}\n"
            "Apply the prediction-capture patches to reproduce_all.py and run it first.\n")
    return pd.read_csv(path)


# ============================================================================
# S1 — model comparison  (seed-averaged per-observation loss)
# ============================================================================

def _model_groups(dfp):
    """Group 'Model|seedN' columns by model, preserving file order."""
    groups, order = {}, []
    for c in dfp.columns:
        if c == "actual_USD":
            continue
        name = c.split("|")[0]
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(c)
    return order, groups


def _loss(y, cols, dfp, metric):
    """Per-observation loss, averaged over seeds.

    The mean of this vector equals the seed-averaged MAE / MSE reported in
    Table 5 exactly, so the bootstrap point estimate is arithmetically
    consistent with the tabulated values.
    """
    e = np.stack([dfp[c].values - y for c in cols])
    return (np.abs(e) if metric == "mae" else e ** 2).mean(axis=0)


def paired_bootstrap_loss(loss_a, loss_b):
    """Paired bootstrap of a loss difference (a minus b).
    Negative -> a has the lower error. CI containing zero -> indistinguishable."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    boot = np.fromiter((d[i].mean() for i in _boot_indices(len(d))),
                       dtype=float, count=N_BOOT)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return d.mean(), lo, hi, min(p, 1.0)


def paired_bootstrap_rmse(loss_a, loss_b):
    """Paired bootstrap of an RMSE difference (same resamples, USD units)."""
    a, b = np.asarray(loss_a, float), np.asarray(loss_b, float)
    boot = np.fromiter((np.sqrt(a[i].mean()) - np.sqrt(b[i].mean())
                        for i in _boot_indices(len(a))), dtype=float, count=N_BOOT)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return np.sqrt(a.mean()) - np.sqrt(b.mean()), lo, hi, min(p, 1.0)


def s1_models(dfp):
    print("\n[S1] Paired bootstrap across Table-5 configurations ...")
    y = dfp["actual_USD"].values
    order, groups = _model_groups(dfp)
    if REF_MODEL not in groups:
        raise SystemExit(f"{REF_MODEL} columns not found in predictions file.")

    mae = {m: _loss(y, groups[m], dfp, "mae") for m in order}
    sq = {m: _loss(y, groups[m], dfp, "sq") for m in order}

    rows = []
    for m in order:
        base = {"Model": m, "Seeds": len(groups[m]),
                "Test MAE (USD)": round(mae[m].mean(), 2),
                "Test RMSE (USD)": round(float(np.sqrt(sq[m].mean())), 2)}
        if m == REF_MODEL:
            rows.append({**base, "delta MAE vs CatBoost (USD)": 0.0,
                         "95% CI": "reference", "p": "",
                         "Distinguishable from CatBoost": "reference"})
            continue
        d, lo, hi, p = paired_bootstrap_loss(mae[m], mae[REF_MODEL])
        rows.append({**base, "delta MAE vs CatBoost (USD)": round(d, 2),
                     "95% CI": _ci(lo, hi, 2), "p": round(p, 4),
                     "Distinguishable from CatBoost": "no" if lo <= 0 <= hi else "yes"})
        print(f"    {m:20s} dMAE={d:9.2f}  CI={_ci(lo, hi, 2)}  p={p:.4f}")

    sq_rows = []
    for m in order:
        if m == REF_MODEL:
            continue
        d, lo, hi, p = paired_bootstrap_loss(sq[m], sq[REF_MODEL])
        dr, rlo, rhi, rp = paired_bootstrap_rmse(sq[m], sq[REF_MODEL])
        sq_rows.append({
            "Comparison": f"{m} - CatBoost",
            "delta mean squared error (USD^2)": round(d, 1),
            "MSE 95% CI": _ci(lo, hi, 0),
            "MSE p": round(p, 4),
            "delta RMSE (USD)": round(dr, 2),
            "RMSE 95% CI": _ci(rlo, rhi, 2),
            "RMSE p": round(rp, 4),
            "Distinguishable": "no" if lo <= 0 <= hi else "yes",
        })
    return pd.DataFrame(rows), pd.DataFrame(sq_rows)


# ============================================================================
# S2 — normalization strategies (common TL units)
# ============================================================================

def s2_normalization(dfn):
    print("\n[S2] Paired bootstrap across normalization strategies (TL) ...")
    y = dfn["actual_TL"].values
    strategies = [c for c in dfn.columns if c != "actual_TL"]
    rows = []
    for name in strategies:
        mae = mean_absolute_error(y, dfn[name].values)
        if name == REF_STRATEGY:
            rows.append({"Strategy": name, "Test MAE (TL)": round(mae),
                         "delta MAE vs USD/TRY (TL)": 0, "95% CI": "reference",
                         "p": "", "Distinguishable": "reference"})
            continue
        dm, lo, hi, p = paired_bootstrap(y, dfn[name].values, dfn[REF_STRATEGY].values, "mae")
        rows.append({"Strategy": name, "Test MAE (TL)": round(mae),
                     "delta MAE vs USD/TRY (TL)": round(dm),
                     "95% CI": _ci(lo, hi, 0), "p": round(p, 4),
                     "Distinguishable": "no" if lo <= 0 <= hi else "yes"})
        print(f"    {name:12s} dMAE={dm:10.0f} TL  CI={_ci(lo, hi, 0)}")
    return pd.DataFrame(rows)


# ============================================================================
# S3 / S4 — segment error and signed bias
# ============================================================================

def _segment_row(label, yt, yp, n_total):
    err = yp - yt                       # positive -> the model OVERVALUES
    bias, blo, bhi = bootstrap_mean(err)
    return {
        "Segment": label,
        "N": int(len(yt)),
        "Share (%)": round(len(yt) / n_total * 100, 2),
        "Mean actual (USD)": round(float(yt.mean())),
        "MAE (USD)": round(mean_absolute_error(yt, yp)),
        "RMSE (USD)": round(float(np.sqrt(mean_squared_error(yt, yp)))),
        "MAPE (%)": round(mean_absolute_percentage_error(yt, yp) * 100, 2),
        "Mean signed error (USD)": round(bias),
        "Bias 95% CI (USD)": _ci(blo, bhi, 0),
        "Median signed error (%)": round(float(np.median(err / yt * 100)), 2),
        "Systematic bias": "none" if blo <= 0 <= bhi else ("overvalues" if bias > 0 else "undervalues"),
    }


def s3_segments(cfg, y_te, y_pred):
    print("\n[S3] Segment error and signed bias ...")
    da = pd.read_csv(cfg["aralik_csv"], encoding="utf-8-sig", sep=",")
    if len(da) != len(y_te):
        raise SystemExit("December CSV and prediction file are not row-aligned.")
    yt, yp = np.asarray(y_te, float), np.asarray(y_pred, float)
    n = len(yt)
    age = cfg["ref_year"] - da["yil"].values

    seg_defs = [
        ("Seller type", np.asarray(da["kimden"].astype(str))),
        ("Fuel type", np.asarray(da["yakit_tipi"].astype(str))),
        ("Transmission", np.asarray(da["vites_tipi"].astype(str))),
        ("Age band (years)", np.asarray(pd.cut(
            age, bins=[-1, 3, 5, 8, 12, 200],
            labels=["<=3", "4-5", "6-8", "9-12", ">=13"]).astype(str))),
        ("Price quintile", np.asarray(pd.qcut(
            yt, 5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4",
                           "Q5 (most expensive)"]).astype(str))),
    ]

    blocks = []
    for block, keys in seg_defs:
        rows = []
        for level in pd.unique(keys):
            if str(level).lower() in ("nan", "none", ""):
                continue
            mask = (keys == level)
            if mask.sum() < 30:                    # too small for a stable interval
                continue
            rows.append(_segment_row(str(level), yt[mask], yp[mask], n))
        df = pd.DataFrame(rows).sort_values("N", ascending=False)
        df.insert(0, "Breakdown", block)
        blocks.append(df)
        print(f"    {block:20s} {len(df)} levels")
    return pd.concat(blocks, ignore_index=True)


def s4_fallback(cfg, y_te, y_pred):
    print("\n[S4] Fallback split (new series vs. new brand) ...")
    dh = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(cfg["aralik_csv"], encoding="utf-8-sig", sep=",")
    pairs = set(zip(dh["marka"], dh["seri"]))
    brands = set(dh["marka"].unique())
    status = np.array(["known" if (m, s) in pairs
                       else ("fb_series" if m in brands else "fb_brand")
                       for m, s in zip(da["marka"], da["seri"])])
    yt, yp = np.asarray(y_te, float), np.asarray(y_pred, float)
    n = len(yt)

    groups = [
        ("All test data", np.ones(n, dtype=bool)),
        ("Known brand-series", status == "known"),
        ("Fallback (all)", status != "known"),
        ("New series of a known brand (brand-mean fallback)", status == "fb_series"),
        ("Entirely new brand (global-mean fallback)", status == "fb_brand"),
    ]
    rows = []
    for name, mask in groups:
        if mask.sum() < 5:
            continue
        r = _segment_row(name, yt[mask], yp[mask], n)
        r["R2"] = round(r2_score(yt[mask], yp[mask]), 4)
        r["MAE (TL)"] = round(mean_absolute_error(yt[mask], yp[mask]) * cfg["USD_ARA"])  # convert before rounding (matches Table 11)
        rows.append(r)
    return pd.DataFrame(rows)


# ============================================================================
# S5 — bootstrap CIs for the decomposition
# ============================================================================

def s5_decomposition(cfg):
    print("\n[S5] Bootstrap CIs for the decomposition components ...")
    Xh, Xa, yh, ya = preprocess(cfg, cfg["USD_HAZ"], cfg["USD_ARA"])
    bins = [-1, 3, 5, 8, 12, 200]
    labels = ["<=3", "4-5", "6-8", "9-12", ">=13"]
    h = pd.DataFrame({"band": pd.cut(Xh["age"].values, bins=bins, labels=labels),
                      "usd": yh.values})
    a = pd.DataFrame({"band": pd.cut(Xa["age"].values, bins=bins, labels=labels),
                      "usd": ya.values})

    def like_for_like(hh, aa):
        """June-weighted within-band mean USD price ratio (December / June)."""
        w = hh.groupby("band", observed=False).size().reindex(labels).fillna(0.0)
        hu = hh.groupby("band", observed=False)["usd"].mean().reindex(labels)
        au = aa.groupby("band", observed=False)["usd"].mean().reindex(labels)
        ok = hu.notna() & au.notna() & (w > 0)
        if not ok.any():
            return np.nan
        ww = w[ok] / w[ok].sum()
        return float((ww * (au[ok] / hu[ok])).sum())

    pt_l4l = like_for_like(h, a)
    pt_usd = float(a["usd"].mean() / h["usd"].mean())

    rng = np.random.default_rng(BOOT_SEED)
    b_l4l, b_usd, b_comp = [], [], []
    hv, av = h.values, a.values
    for _ in range(N_BOOT):
        hs = pd.DataFrame(hv[rng.integers(0, len(h), len(h))], columns=["band", "usd"])
        as_ = pd.DataFrame(av[rng.integers(0, len(a), len(a))], columns=["band", "usd"])
        hs["usd"] = hs["usd"].astype(float)
        as_["usd"] = as_["usd"].astype(float)
        l4l = like_for_like(hs, as_)
        u = float(as_["usd"].mean() / hs["usd"].mean())
        b_l4l.append(l4l)
        b_usd.append(u)
        b_comp.append(u / l4l)

    def pct(arr, point):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return round((point - 1) * 100, 2), round((lo - 1) * 100, 2), round((hi - 1) * 100, 2)

    cur = cfg["USD_ARA"] / cfg["USD_HAZ"]
    rows = [{"Component": "Currency (deterministic constant)",
             "Point estimate (%)": round((cur - 1) * 100, 2),
             "95% CI (%)": "n/a", "Distinguishable from zero": "n/a"}]
    for label, arr, p0 in [
        ("USD residual after currency (mean USD price, Dec/Jun)", b_usd, pt_usd),
        ("Like-for-like within age band (June-weighted)", b_l4l, pt_l4l),
        ("Composition (residual / like-for-like)", b_comp, pt_usd / pt_l4l),
    ]:
        p, lo, hi = pct(arr, p0)
        rows.append({"Component": label, "Point estimate (%)": p,
                     "95% CI (%)": _ci(lo, hi, 2),
                     "Distinguishable from zero": "no" if lo <= 0 <= hi else "yes"})
        print(f"    {label:52s} {p:+.2f}%  {_ci(lo, hi, 2)}")
    return pd.DataFrame(rows)


# ============================================================================
# Writer
# ============================================================================

def save(sheets):
    path = os.path.join(CONFIG["output_dir"], OUT_XLSX)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet, title, df in sheets:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name=sheet, index=False, header=False)
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)
    print(f"\nWritten: {path}")


def main():
    t0 = time.time()
    print("=" * 78)
    print("REVIEWER 4 SUPPLEMENTARY ANALYSIS  (post-processing only, no model fitting)")
    print(f"bootstrap replicates: {N_BOOT}    seed: {BOOT_SEED}")
    print("=" * 78)

    dfp = _load("predictions_models.csv")
    dfn = _load("predictions_normalization.csv")

    s1, s1b = s1_models(dfp)
    s2 = s2_normalization(dfn)
    y_te = dfp["actual_USD"].values
    y_cb = dfp[f"{REF_MODEL}|seed42"].values   # canonical fit, = Tables 10-11
    s3 = s3_segments(CONFIG, y_te, y_cb)
    s4 = s4_fallback(CONFIG, y_te, y_cb)
    s5 = s5_decomposition(CONFIG)

    save([
        ("S1a - Model bootstrap",
         "Table S1a: Paired bootstrap of test-MAE differences against CatBoost "
         "(December test set, 2000 replicates)", s1),
        ("S1b - Squared error",
         "Table S1b: Paired bootstrap of squared-error and RMSE differences against CatBoost", s1b),
        ("S2 - Normalization",
         "Table S2: Paired bootstrap across normalization strategies, common TL units", s2),
        ("S3 - Segment bias",
         "Table S3: Prediction error and signed bias by market segment", s3),
        ("S4 - Fallback split",
         "Table S4: Fallback decomposition - new series of a known brand vs. "
         "entirely new brand", s4),
        ("S5 - Decomposition CI",
         "Table S5: Bootstrap confidence intervals for the price-decomposition components", s5),
    ])

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min\n")
    print("--- quote these in the response letter -----------------------------")
    print(s1[["Model", "delta MAE vs CatBoost (USD)", "95% CI",
              "Distinguishable from CatBoost"]].to_string(index=False))
    print()
    print(s4[["Segment", "N", "R2", "MAPE (%)", "Median signed error (%)",
              "Systematic bias"]].to_string(index=False))
    print()
    print(s5.to_string(index=False))


if __name__ == "__main__":
    main()