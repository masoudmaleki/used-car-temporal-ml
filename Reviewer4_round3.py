"""
================================================================================
reviewer4_round3.py — ADD-ON for the third revision round (Reviewer 4)
================================================================================

Runs alongside reproduce_all.py and imports the SAME pipeline functions, so every
number below comes from the identical preprocessing, encoding and model
configuration that produced the manuscript tables. reproduce_all.py is NOT
modified and no previously reported value is recomputed or overwritten.

  T1  Alternative loss functions and prediction intervals            -> R4-1
      Squared error (baseline) vs. log-price, Huber (delta swept over
      1x, 3x and 10x the baseline TRAINING MAE) and quantile-median
      objectives; absolute AND relative error per price quintile, plus
      signed bias per quintile. A 90% interval from quantile regression
      (alpha = 0.05 / 0.95), with coverage and width reported separately
      for ordinary listings and unfamiliar brand-series cases.

  T2  Segment-based fallback for unseen brands                       -> R4-2
      Replaces the global training mean with a smoothed segment cell
      (body type x age band x engine-size band) built from TRAINING data
      only, with a documented ladder down to the global mean. The SAME
      fitted model scores both encodings, so the comparison isolates the
      encoding change and needs no retraining. Interval width and coverage
      are reported by training support and under both encodings.

  T3  Operational cost of each learner                               -> R4-3
      Training wall time, serialized model size, scoring throughput for
      1,000 listings, and the implied cost of periodic retraining.

  T4  Conformal calibration of the valuation range                   -> R4-1
      Conformalized quantile regression (Romano et al., 2019): the June
      training set is split into a fitting fold and a calibration fold,
      and the interval is widened by the calibration-fold conformity
      quantile. Reports in-period coverage (a correctness check on the
      procedure) and out-of-period coverage, which quantifies what the
      six-month shift costs in calibration.

Output sheets and the manuscript
--------------------------------
  S6 (T1a), S7 (T1b), S8 (T1c), S9 (T2a), S10 (T2b), S11 (T4), S12 (T3b), Table 13 (T3a)

Run
---
  python "Reviewer4 round3.py"          (same folder and environment as
                                       reproduce_all.py; do not upgrade any
                                       package between the two runs)

Writes  results/reviewer4_round3.xlsx  and prints a quotable summary.
Runtime: roughly 2-3 minutes.
================================================================================
"""

import os
import platform
import tempfile
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

OUT_XLSX = "reviewer4_round3.xlsx"
INTERVAL = (0.05, 0.95)          # 90% nominal prediction interval
HUBER_MULTIPLIERS = (1, 3, 10)   # delta = multiplier x baseline TRAINING MAE
CAL_FRACTION = 0.25              # share of June held out for conformal calibration
CAL_SEED = 42
N_TIMING_REPEATS = 7
SCORE_BATCH = 1000

AGE_BINS = [-1, 3, 5, 8, 12, 200]
AGE_LABELS = ["<=3", "4-5", "6-8", "9-12", ">=13"]
QUINTILE_LABELS = ["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (most expensive)"]


# ============================================================================
# Shared helpers
# ============================================================================

def _metrics(y, p):
    return {
        "R2": r2_score(y, p),
        "MAE (USD)": mean_absolute_error(y, p),
        "RMSE (USD)": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE (%)": mean_absolute_percentage_error(y, p) * 100,
        "Median signed error (%)": float(np.median((p - y) / y * 100)),
    }


def _cb(cfg, loss=None, **kw):
    from catboost import CatBoostRegressor
    p = dict(iterations=cfg["cb_iterations"], depth=6, learning_rate=0.05,
             verbose=0, random_state=42)
    if loss:
        p["loss_function"] = loss
    p.update(kw)
    return CatBoostRegressor(**p)


def fallback_status(cfg):
    """Per-row fallback status of the December test set, in file order.

    Identical construction to table11_robustness() in reproduce_all.py:
      known      -> (brand, series) seen in June
      fb_marka   -> brand seen, series new  (brand-mean fallback)
      fb_global  -> brand entirely new      (global-mean fallback)
    """
    dh = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(cfg["aralik_csv"], encoding="utf-8-sig", sep=",")
    pairs = set(zip(dh["marka"], dh["seri"]))
    brands = set(dh["marka"].unique())
    status = np.array([
        "known" if (m, s) in pairs else ("fb_marka" if m in brands else "fb_global")
        for m, s in zip(da["marka"], da["seri"])
    ])
    return status


def _package_versions():
    """Library versions for Supplementary Table S12 (XGBoost is version-sensitive)."""
    rows = []
    for mod, label in (("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit-learn"),
                       ("xgboost", "xgboost"), ("lightgbm", "lightgbm"), ("catboost", "catboost"),
                       ("shap", "shap")):
        try:
            rows.append({"Item": f"{label} version", "Value": __import__(mod).__version__})
        except Exception:
            pass
    return rows


def _quintiles(y):
    return np.asarray(pd.qcut(y, 5, labels=QUINTILE_LABELS).astype(str))


def _cov_row(label, y, lo, hi, msk):
    msk = np.asarray(msk, dtype=bool)
    if msk.sum() < 5:
        return None
    inside = (y[msk] >= lo[msk]) & (y[msk] <= hi[msk])
    width = hi[msk] - lo[msk]
    return {
        "Group": label,
        "N": int(msk.sum()),
        "Coverage (%)": round(float(inside.mean() * 100), 2),
        "Mean interval width (USD)": round(float(width.mean())),
        "Median interval width (USD)": round(float(np.median(width))),
        "Median width / price (%)": round(float(np.median(width / y[msk]) * 100), 1),
    }


# ============================================================================
# T1 — alternative objectives and prediction intervals  (R4-1)
# ============================================================================

def t1_objectives(cfg, status):
    print("\n[T1] Alternative objectives and 90% prediction intervals ...")
    X_tr, X_te, y_tr, y_te = preprocess(cfg, cfg["USD_HAZ"], cfg["USD_ARA"])
    Xtr, Xte = X_tr.values, X_te.values
    ytr, yte = y_tr.values, y_te.values

    preds = {}

    # --- baseline: squared error (identical to the manuscript's CatBoost) ---
    t0 = time.time()
    m = _cb(cfg)
    m.fit(Xtr, ytr)
    preds["Squared error (baseline)"] = m.predict(Xte)
    train_mae = mean_absolute_error(ytr, m.predict(Xtr))
    print(f"    baseline                R2={r2_score(yte, preds['Squared error (baseline)']):.4f}"
          f"  ({time.time()-t0:.0f}s)   train MAE={train_mae:,.0f} USD")

    # --- log-price target, back-transformed with exp ---
    t0 = time.time()
    m_log = _cb(cfg)
    m_log.fit(Xtr, np.log(ytr))
    preds["Log-price target"] = np.exp(m_log.predict(Xte))
    print(f"    log-price               R2={r2_score(yte, preds['Log-price target']):.4f}"
          f"  ({time.time()-t0:.0f}s)")

    # --- Huber, delta swept relative to the baseline TRAINING MAE ---
    for mult in HUBER_MULTIPLIERS:
        delta = float(round(train_mae * mult))
        t0 = time.time()
        # Newton leaf estimation diverges for Huber (zero Hessian outside delta);
        # gradient leaf estimation gives stable fits.
        mh = _cb(cfg, loss=f"Huber:delta={delta}", leaf_estimation_method="Gradient")
        mh.fit(Xtr, ytr)
        key = f"Huber (delta={delta:.0f} USD = {mult}x train MAE)"
        preds[key] = mh.predict(Xte)
        print(f"    huber delta={delta:<8.0f}    R2={r2_score(yte, preds[key]):.4f}"
              f"  ({time.time()-t0:.0f}s)")

    # --- quantile regression: median plus the 90% interval bounds ---
    qmodels = {}
    for a in (INTERVAL[0], 0.50, INTERVAL[1]):
        t0 = time.time()
        mq = _cb(cfg, loss=f"Quantile:alpha={a}")
        mq.fit(Xtr, ytr)
        qmodels[a] = mq
        print(f"    quantile alpha={a:<5}    ({time.time()-t0:.0f}s)")
    qpred = {a: qmodels[a].predict(Xte) for a in qmodels}
    preds["Quantile (median)"] = qpred[0.50]

    lo = np.minimum(qpred[INTERVAL[0]], qpred[INTERVAL[1]])
    hi = np.maximum(qpred[INTERVAL[0]], qpred[INTERVAL[1]])

    overall = pd.DataFrame(
        [{"Objective": k, **_metrics(yte, v)} for k, v in preds.items()]
    ).round(4)

    q = _quintiles(yte)
    rows = []
    for name, p in preds.items():
        for lab in QUINTILE_LABELS:
            msk = q == lab
            rows.append({
                "Objective": name,
                "Price quintile": lab,
                "N": int(msk.sum()),
                "Mean actual (USD)": round(float(yte[msk].mean())),
                "MAE (USD)": round(mean_absolute_error(yte[msk], p[msk])),
                "RMSE (USD)": round(float(np.sqrt(mean_squared_error(yte[msk], p[msk])))),
                "MAPE (%)": round(mean_absolute_percentage_error(yte[msk], p[msk]) * 100, 2),
                "Median signed error (%)": round(float(np.median((p[msk] - yte[msk]) / yte[msk] * 100)), 2),
                "Share over-valued (%)": round(float((p[msk] > yte[msk]).mean() * 100), 1),
            })
    per_quintile = pd.DataFrame(rows)

    cov = [_cov_row("All test listings", yte, lo, hi, np.ones(len(yte), dtype=bool))]
    for lab in QUINTILE_LABELS:
        cov.append(_cov_row(f"  {lab}", yte, lo, hi, q == lab))
    cov.append(_cov_row("Ordinary listings (known brand-series)", yte, lo, hi, status == "known"))
    cov.append(_cov_row("Unfamiliar brand-series (all fallback)", yte, lo, hi, status != "known"))
    cov.append(_cov_row("  New series of a known brand", yte, lo, hi, status == "fb_marka"))
    cov.append(_cov_row("  Entirely new brand", yte, lo, hi, status == "fb_global"))
    coverage = pd.DataFrame([c for c in cov if c])

    return overall, per_quintile, coverage, (lo, hi), qmodels, (X_tr, X_te, y_tr, y_te)


# ============================================================================
# T2 — segment-based fallback for unseen brands  (R4-2)
# ============================================================================

def _smooth(cell_sum, cell_n, prior, k):
    return (cell_sum + k * prior) / (cell_n + k)


def build_enriched_test_matrix(cfg, status, X_te):
    """Re-encode TE_ms for rows whose brand is absent from training.

    Ladder, estimated on JUNE (training) data only:
        1. body type x age band x engine-size band
        2. body type x age band
        3. age band
        4. global training mean          (the current behaviour)
    Every other row keeps exactly the encoding used in the manuscript.
    """
    k = cfg["te_smoothing_k"]
    dh = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(cfg["aralik_csv"], encoding="utf-8-sig", sep=",")
    if len(da) != len(X_te):
        raise SystemExit("December CSV and test matrix are not row-aligned.")

    tr = pd.DataFrame({
        "price": dh["fiyat"].values / cfg["USD_HAZ"],
        "kasa": dh["kasa_tipi"].astype(str).values,
        "age": cfg["ref_year"] - dh["yil"].values,
        "cc": dh["motor_hacmi"].values,
    })
    te = pd.DataFrame({
        "kasa": da["kasa_tipi"].astype(str).values,
        "age": cfg["ref_year"] - da["yil"].values,
        "cc": da["motor_hacmi"].values,
    })

    edges = np.unique(np.nanquantile(tr["cc"], [0, .25, .5, .75, 1.0]))
    edges[0], edges[-1] = -np.inf, np.inf
    for d in (tr, te):
        d["ageb"] = pd.cut(d["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        d["ccb"] = pd.cut(d["cc"], bins=edges, labels=False, include_lowest=True)
        d["ccb"] = d["ccb"].fillna(-1).astype(int).astype(str)

    mu_global = float(tr["price"].mean())

    def tbl(keys):
        g = tr.groupby(keys, observed=True)["price"].agg(["sum", "count"])
        return _smooth(g["sum"], g["count"], mu_global, k), g["count"]

    lvl3, n3 = tbl(["kasa", "ageb", "ccb"])
    lvl2, n2 = tbl(["kasa", "ageb"])
    lvl1, _ = tbl(["ageb"])

    idx3 = pd.MultiIndex.from_arrays([te["kasa"], te["ageb"], te["ccb"]])
    idx2 = pd.MultiIndex.from_arrays([te["kasa"], te["ageb"]])
    v = pd.Series(lvl3.reindex(idx3).values, dtype=float)
    v = v.fillna(pd.Series(lvl2.reindex(idx2).values, dtype=float))
    v = v.fillna(pd.Series(lvl1.reindex(te["ageb"]).values, dtype=float))
    v = v.fillna(mu_global)
    support = (pd.Series(n3.reindex(idx3).values, dtype=float)
               .fillna(pd.Series(n2.reindex(idx2).values, dtype=float))
               .fillna(0.0))

    mask = status == "fb_global"
    X_new = X_te.copy()
    X_new.loc[X_new.index[mask], "TE_ms"] = v.values[mask]
    return X_new, mask, v.values, support.values, mu_global


def t2_enriched_fallback(cfg, status, data, interval, qmodels):
    print("\n[T2] Segment-based fallback for unseen brands ...")
    X_tr, X_te, y_tr, y_te = data
    yte = y_te.values
    X_new, mask, v, support, mu_global = build_enriched_test_matrix(cfg, status, X_te)

    print(f"    rows re-encoded: {int(mask.sum())} (entirely new brands)")
    print(f"    global mean {mu_global:,.0f} USD -> segment cells "
          f"{v[mask].min():,.0f}-{v[mask].max():,.0f} USD, "
          f"median support {np.median(support[mask]):.0f} training listings")

    m = _cb(cfg)
    m.fit(X_tr.values, y_tr)
    p_old = m.predict(X_te.values)
    p_new = m.predict(X_new.values)

    groups = [
        ("All test listings", np.ones(len(yte), dtype=bool)),
        ("Known brand-series", status == "known"),
        ("Fallback (all)", status != "known"),
        ("New series of a known brand", status == "fb_marka"),
        ("Entirely new brand", status == "fb_global"),
    ]
    rows = []
    for label, msk in groups:
        if msk.sum() < 5:
            continue
        for tag, p in (("Global mean (current)", p_old),
                       ("Segment cell (proposed)", p_new)):
            rows.append({"Group": label, "Fallback rule": tag, "N": int(msk.sum()),
                         **{kk: round(vv, 4) for kk, vv in _metrics(yte[msk], p[msk]).items()}})
    comparison = pd.DataFrame(rows)

    # intervals under both encodings
    lo_o, hi_o = interval
    qn = {a: qmodels[a].predict(X_new.values) for a in qmodels}
    lo_n = np.minimum(qn[INTERVAL[0]], qn[INTERVAL[1]])
    hi_n = np.maximum(qn[INTERVAL[0]], qn[INTERVAL[1]])

    sup_all = np.full(len(yte), np.nan)
    sup_all[mask] = support[mask]
    bands = [
        ("Known brand-series", status == "known"),
        ("New series of a known brand", status == "fb_marka"),
        ("Entirely new brand (all)", mask),
        ("  segment support >= 500", mask & (sup_all >= 500)),
        ("  segment support 100-499", mask & (sup_all >= 100) & (sup_all < 500)),
        ("  segment support < 100", mask & (sup_all < 100)),
    ]
    srows = []
    for label, msk in bands:
        msk = np.asarray(msk, dtype=bool)
        if msk.sum() < 5:
            continue
        a = _cov_row(label, yte, lo_o, hi_o, msk)
        b = _cov_row(label, yte, lo_n, hi_n, msk)
        if a is None or b is None:
            continue
        srows.append({
            "Group": label, "N": a["N"],
            "Coverage, global-mean encoding (%)": a["Coverage (%)"],
            "Coverage, segment-cell encoding (%)": b["Coverage (%)"],
            "Median width, global-mean (USD)": a["Median interval width (USD)"],
            "Median width, segment-cell (USD)": b["Median interval width (USD)"],
            "Median width / price, segment-cell (%)": b["Median width / price (%)"],
        })
    support_table = pd.DataFrame(srows)
    return comparison, support_table


# ============================================================================
# T3 — operational cost of each learner  (R4-3)
# ============================================================================

def t3_operational(cfg, data):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    import joblib

    print("\n[T3] Operational cost (training, size, scoring throughput) ...")
    X_tr, X_te, y_tr, y_te = data
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)
    Xtr, Xte = X_tr.values, X_te.values
    cbi, mlp_it = cfg["cb_iterations"], cfg["mlp_max_iter"]
    s = 42

    specs = [
        ("CatBoost", lambda: CatBoostRegressor(iterations=cbi, depth=6, learning_rate=0.05,
                                               verbose=0, random_state=s), False, "cbm"),
        ("XGBoost", lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, random_state=s,
                                         verbosity=0, n_jobs=-1), False, "json"),
        ("LightGBM", lambda: LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8, random_state=s,
                                           verbose=-1, n_jobs=-1), False, "txt"),
        ("Random Forest", lambda: RandomForestRegressor(n_estimators=200, max_depth=15,
                                                        min_samples_leaf=5, random_state=s,
                                                        n_jobs=-1), False, "joblib"),
        ("MLP", lambda: MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=mlp_it,
                                     early_stopping=True, random_state=s,
                                     learning_rate_init=0.001), True, "joblib"),
    ]

    batch_idx = np.arange(SCORE_BATCH)
    rows = []
    for name, factory, needs_sc, fmt in specs:
        A, B = (Xtr_sc, Xte_sc) if needs_sc else (Xtr, Xte)
        m = factory()
        t0 = time.perf_counter()
        m.fit(A, y_tr)
        fit_s = time.perf_counter() - t0

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"model.{fmt}")
            if fmt == "cbm":
                m.save_model(path)
            elif fmt == "json":
                m.get_booster().save_model(path)   # booster save works across xgboost/sklearn versions
            elif fmt == "txt":
                m.booster_.save_model(path)
            else:
                joblib.dump(m, path, compress=0)
            size_mb = os.path.getsize(path) / 1024 ** 2

        batch = B[batch_idx]
        m.predict(batch)                                   # warm-up, not timed
        times = []
        for _ in range(N_TIMING_REPEATS):
            t0 = time.perf_counter()
            m.predict(batch)
            times.append(time.perf_counter() - t0)
        score_ms = float(np.median(times)) * 1000

        rows.append({
            "Model": name,
            "Test R2 (seed 42)": round(r2_score(y_te, m.predict(B)), 4),
            "Training time (s)": round(fit_s, 1),
            "Model size (MB)": round(size_mb, 2),
            f"Scoring {SCORE_BATCH} listings (ms)": round(score_ms, 1),
            "Throughput (listings/s)": int(SCORE_BATCH / (score_ms / 1000)),
            "Annual retraining cost (s, 12 refits)": round(fit_s * 12, 1),
        })
        print(f"    {name:14s} fit={fit_s:7.1f}s  size={size_mb:6.2f}MB  "
              f"score1k={score_ms:7.1f}ms")

    env = pd.DataFrame([
        {"Item": "Processor", "Value": platform.processor() or platform.machine()},
        {"Item": "Logical cores", "Value": os.cpu_count()},
        {"Item": "Platform", "Value": platform.platform()},
        {"Item": "Python", "Value": platform.python_version()},
        {"Item": "Training rows", "Value": int(len(X_tr))},
        {"Item": "Features", "Value": int(X_tr.shape[1])},
        {"Item": "Scoring batch", "Value": SCORE_BATCH},
        {"Item": "Timing repeats (median reported)", "Value": N_TIMING_REPEATS},
        *_package_versions(),
        {"Item": "Note", "Value": "Single-seed fit; Table 5 reports the mean of three seeds."},
    ])
    return pd.DataFrame(rows), env


# ============================================================================
# T4 — conformal calibration of the valuation range  (R4-1)
# ============================================================================

def t4_conformal(cfg, status, data):
    """Conformalized quantile regression (Romano, Patterson & Candes, 2019).

    The June training set is split into a fitting fold and a calibration fold.
    Quantile models are fitted on the first; on the second the conformity score
    s_i = max(lo_i - y_i, y_i - hi_i) is computed and its (1-alpha) empirical
    quantile q-hat widens the interval symmetrically.

    In-period coverage on the calibration fold is a correctness check on the
    procedure. Out-of-period coverage on December quantifies what the six-month
    shift costs in calibration.
    """
    print("\n[T4] Conformal calibration of the 90% valuation range ...")
    X_tr, X_te, y_tr, y_te = data
    Xtr, ytr = X_tr.values, y_tr.values
    Xte, yte = X_te.values, y_te.values

    rng = np.random.default_rng(CAL_SEED)
    perm = rng.permutation(len(ytr))
    n_cal = int(round(len(ytr) * CAL_FRACTION))
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

    qm = {}
    for a in (INTERVAL[0], INTERVAL[1]):
        t0 = time.time()
        m = _cb(cfg, loss=f"Quantile:alpha={a}")
        m.fit(Xtr[fit_idx], ytr[fit_idx])
        qm[a] = m
        print(f"    fit alpha={a:<5} on {len(fit_idx):,} rows  ({time.time()-t0:.0f}s)")

    def bounds(X):
        a = qm[INTERVAL[0]].predict(X)
        b = qm[INTERVAL[1]].predict(X)
        return np.minimum(a, b), np.maximum(a, b)

    lo_c, hi_c = bounds(Xtr[cal_idx])
    y_c = ytr[cal_idx]
    scores = np.maximum(lo_c - y_c, y_c - hi_c)
    nominal = round((INTERVAL[1] - INTERVAL[0]), 10)
    level = min(1.0, np.ceil((n_cal + 1) * nominal) / n_cal)
    qhat = float(np.quantile(scores, level, method="higher"))
    print(f"    calibration rows={n_cal:,}  q-hat={qhat:,.0f} USD "
          f"(level={level:.4f})")

    lo_te, hi_te = bounds(Xte)
    variants = {
        "Uncalibrated quantile interval": (lo_te, hi_te),
        "Conformalized interval": (lo_te - qhat, hi_te + qhat),
    }

    rows = []
    # in-period correctness check on the held-out calibration fold
    for tag, (l, h) in (("Uncalibrated quantile interval", (lo_c, hi_c)),
                        ("Conformalized interval", (lo_c - qhat, hi_c + qhat))):
        r = _cov_row("June calibration fold (in-period)", y_c, l, h,
                     np.ones(len(y_c), dtype=bool))
        rows.append({"Interval": tag, **r})  # always non-empty

    groups = [
        ("December test set (out-of-period)", np.ones(len(yte), dtype=bool)),
        ("  Ordinary listings (known brand-series)", status == "known"),
        ("  Unfamiliar brand-series (all fallback)", status != "known"),
        ("  Entirely new brand", status == "fb_global"),
    ]
    q = _quintiles(yte)
    groups += [(f"  {lab}", q == lab) for lab in QUINTILE_LABELS]

    for tag, (l, h) in variants.items():
        for label, msk in groups:
            r = _cov_row(label, yte, l, h, msk)
            if r is None:
                continue
            rows.append({"Interval": tag, **r})

    out = pd.DataFrame(rows)
    out = out[["Interval", "Group", "N", "Coverage (%)",
               "Median interval width (USD)", "Median width / price (%)"]]
    return out, qhat


# ============================================================================
# Writer
# ============================================================================

def save(sheets, cfg):
    path = os.path.join(cfg["output_dir"], OUT_XLSX)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet, title, df in sheets:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name=sheet, index=False, header=False)
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)
    print(f"\nWritten: {path}")


def main():
    t0 = time.time()
    print("=" * 78)
    print("REVIEWER 4 — THIRD ROUND ANALYSIS")
    print(f"nominal interval: {round((INTERVAL[1]-INTERVAL[0])*100)}%  "
          f"(alpha = {INTERVAL[0]} / {INTERVAL[1]})")
    print("=" * 78)
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    status = fallback_status(CONFIG)
    overall, per_quintile, coverage, interval, qmodels, data = t1_objectives(CONFIG, status)
    fb_cmp, support_tbl = t2_enriched_fallback(CONFIG, status, data, interval, qmodels)
    ops, env = t3_operational(CONFIG, data)
    conf, qhat = t4_conformal(CONFIG, status, data)

    save([
        ("S6 - Objectives", "Table S6: Alternative training objectives, December test set", overall),
        ("S7 - By quintile", "Table S7: Absolute and relative error by price quintile, per objective", per_quintile),
        ("S8 - Interval coverage", "Table S8: Nominal 90% prediction-interval coverage and width", coverage),
        ("S9 - Fallback rule", "Table S9: Global-mean vs. segment-cell fallback for unseen brands", fb_cmp),
        ("S10 - Support bands", "Table S10: Interval coverage and width by training support, both encodings", support_tbl),
        ("S11 - Conformal", f"Table S11: Conformalized quantile intervals (q-hat = {qhat:,.0f} USD)", conf),
        ("S12 - Environment", "Table S12: Benchmark environment", env),
        ("Table 13 - Operational cost", "Table 13: Training time, model size and scoring throughput", ops),
    ], CONFIG)

    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min\n")
    print("--- quote these in the response letter " + "-" * 38)
    print("\n[S6] objectives\n", overall.to_string(index=False))
    print("\n[S7] median signed error by quintile")
    print(per_quintile.pivot(index="Price quintile", columns="Objective",
                             values="Median signed error (%)")
          .reindex(QUINTILE_LABELS).to_string())
    print("\n[S8] interval coverage\n", coverage.to_string(index=False))
    print("\n[S9] fallback rule\n", fb_cmp.to_string(index=False))
    print("\n[S10] support bands\n", support_tbl.to_string(index=False))
    print("\n[Table 13] operational cost\n", ops.to_string(index=False))
    print(f"\n[S11] conformal calibration, q-hat = {qhat:,.0f} USD\n", conf.to_string(index=False))


if __name__ == "__main__":
    main()