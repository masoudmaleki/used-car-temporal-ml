"""
================================================================================
Exchange Rate Pass-Through in Used Vehicle Pricing
A Temporal Machine Learning Framework — FULL REPRODUCIBILITY SCRIPT
================================================================================

Single-file pipeline that regenerates EVERY table and figure reported in the
manuscript, writing all tables to one formatted Excel workbook and saving all
figures as 300-DPI PNGs.

Manuscript tables reproduced by this script
--------------------------------------------
  Table 1  Sensitivity to target-encoding smoothing k          -> table1_k_sensitivity()
  Table 2  Evaluation metric definitions (definitional)        -> static reference sheet
  Table 3  Descriptive statistics + macro indicators           -> table3_descriptive()
  Table 4  Normalization strategy comparison (CatBoost)        -> table4_normalization()
  Table 5  Model comparison (11 configurations)               -> table5_models()
  Table 6  Hyperparameter tuning, MLP and Random Forest       -> table6_hyperparameter_tuning()
  Table 7  Period-dummy analysis                               -> table7_period_dummy()
  Table 8  Age-distribution shift                              -> tables_8_9_age()
  Table 9  Within-age-band USD price comparison                -> tables_8_9_age()
  Table 10 Prediction error by price quintile                  -> table10_quintiles()
  Table 11 Robustness: known vs. fallback brand-series         -> table11_robustness()
  Table 12 Positioning vs. prior work (definitional)           -> static reference sheet
  Section 6.5 robustness to the actual December exchange rate  -> dec_rate_robustness()

  Table 13 and Supplementary Tables S6-S12 are produced by "Reviewer4 round3.py";
  Supplementary Tables S1a-S5 by rev4.py (both import CONFIG and preprocess from here).

Figures reproduced (styled: Times New Roman, bold axis labels, no titles, 300 DPI)
  Figure 2   R2 across five normalization strategies (CV/Train/Test)
  Figure 7   SHAP beeswarm (train / test)
  Figure 9   Predicted vs. actual (thousand USD)
  Figure 10  Residual distribution (thousand USD)
  Figure 8   SHAP dependence — brand-series target encoding
  Figure     CatBoost native feature importance (top 15)

Requirements
------------
  pip install pandas numpy scikit-learn xgboost lightgbm catboost shap matplotlib openpyxl

Data files expected alongside this script (relative paths, see CONFIG)
  Haziran_2025_son.csv   (cleaned June training set,   34,509 rows)
  Aralık_2025_son.csv    (cleaned December test set,   37,062 rows)
  Haziran_2025.csv       (raw June file, used ONLY to count raw records for Table 3)  [optional]
  Aralık_2025.xlsx       (raw December file, used ONLY to count raw records)          [optional]

Run
---
  python reproduce_all.py
        -> results/reproduction_all_tables.xlsx  and  results/*.png

  REPRO_FAST=1 python reproduce_all.py
        -> quick smoke test (reduced iterations/seeds); DOES NOT reproduce paper values.

--------------------------------------------------------------------------------
IMPORTANT REPRODUCIBILITY NOTES
--------------------------------------------------------------------------------
* ENCODING (corrected in the revision). The June file obtained from Kaggle stored
  Turkish characters double-encoded ("DÃ¼z", "TofaÅŸ") and spelled the brand
  "MINI", while the December scrape is proper UTF-8 ("Düz", "Tofaş", "Mini").
  Run fix_encoding.py once before this script. preprocess() now stops with an
  error if any categorical value is not in its category list, so a silent
  mapping of unknown categories to the reference level cannot recur.

* CatBoost iterations are set in CONFIG. All predictive results (Tables 1, 4, 5,
  10, 11 and the figures) use cb_iterations = 1000. The period-dummy DIAGNOSTIC
  (Table 7) uses cb_iterations_period = 500; see the note in table7_period_dummy().

* XGBoost results depend slightly on the installed XGBoost version (differences in
  the fourth decimal of R2). CatBoost, LightGBM, Random Forest and the linear models
  reproduce exactly across environments. Package versions used for the paper are
  listed in Supplementary Table S12.
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


# ============================================================================
# 0. CONFIGURATION  (single source of truth — no magic numbers below)
# ============================================================================

CONFIG = {
    # --- data files ---
    "haziran_csv":  "Haziran_2025_son.csv",
    "aralik_csv":   "Aralık_2025_son.csv",
    "haziran_raw":  "Haziran_2025.csv",     # optional, raw-record count only
    "aralik_raw":   "Aralık_2025.xlsx",     # optional, raw-record count only

    # --- output ---
    "output_dir":   "results",
    "output_excel": "reproduction_all_tables.xlsx",

    # --- macroeconomic deflators (Haziran / Aralik) ---
    "USD_HAZ": 39.2062,  "USD_ARA": 42.4321,
    "EUR_HAZ": 46.8545,  "EUR_ARA": 49.2998,
    "GOLD_HAZ": 4146.56, "GOLD_ARA": 5781.25,
    "CPI_HAZ": 3400.32,  "CPI_ARA": 3847.35,

    # --- preprocessing / evaluation ---
    "te_smoothing_k":   50,
    "cv_folds":         5,
    "cv_random_state":  42,
    "seeds":            [42, 123, 789],
    "ref_year":         2025,

    # --- model budgets (parametrized so runs are reproducible & configurable) ---
    "cb_iterations":        1000,   # CatBoost, all predictive models + figures
    "cb_iterations_period": 500,    # CatBoost, Table 7 period-dummy diagnostic ONLY
    "mlp_max_iter":         500,
    "tune_n_iter": 25,

    # --- Table 1 sweep ---
    "k_list": [10, 20, 50, 100, 200],
}

# Normalization strategies for Table 4 (name, June rate, December rate).
CONFIG["strategies"] = [
    ("USD/TRY",    CONFIG["USD_HAZ"],  CONFIG["USD_ARA"]),
    ("EUR/TRY",    CONFIG["EUR_HAZ"],  CONFIG["EUR_ARA"]),
    ("CPI (TÜFE)", CONFIG["CPI_HAZ"],  CONFIG["CPI_ARA"]),
    ("Raw TL",     1.0,                1.0),
    ("Gold/TRY",   CONFIG["GOLD_HAZ"], CONFIG["GOLD_ARA"]),
]

# ---- optional FAST smoke-test overrides (NOT for reproducing paper values) ----
if os.environ.get("REPRO_FAST"):
    CONFIG["cb_iterations"] = 40
    CONFIG["cb_iterations_period"] = 40
    CONFIG["mlp_max_iter"] = 60
    CONFIG["seeds"] = [42]
    CONFIG["k_list"] = [10, 50]
    CONFIG["strategies"] = CONFIG["strategies"][:2]
    CONFIG["tune_n_iter"] = 3

    print(">>> REPRO_FAST enabled: reduced budgets — smoke test only, values NOT paper-accurate.\n")

os.makedirs(CONFIG["output_dir"], exist_ok=True)


# ============================================================================
# 1. PREPROCESSING  (identical to the original pipeline — do not alter encoding)
# ============================================================================

def preprocess(cfg, norm_haz, norm_ara):
    """
    Feature engineering + smoothed target encoding + train-median NaN filling.

    Both CSV files must be proper UTF-8 (run fix_encoding.py first).
    Train = June 2025, Test = December 2025.
    """
    df1 = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    df2 = pd.read_csv(cfg["aralik_csv"],  encoding="utf-8-sig", sep=",")

    # -- guard: brand names that differ only in spelling/case across periods --
    june_brands = set(df1["marka"].unique())
    june_fold = {b.casefold() for b in june_brands}
    clash = sorted(b for b in df2["marka"].unique()
                   if b not in june_brands and b.casefold() in june_fold)
    if clash:
        raise ValueError(f"Brand spelling differs between periods: {clash}. "
                         "Run fix_encoding.py first.")

    # -- feature engineering --
    for df in (df1, df2):
        df["age"] = cfg["ref_year"] - df["yil"]
        df["log_km"] = np.log1p(df["kilometre"])
        df["power_per_cc"] = df["motor_gucu"] / df["motor_hacmi"]
        df.drop(columns=["yil", "kilometre", "motor_gucu", "motor_hacmi",
                         "model", "id", "renk"], inplace=True)

    # -- categorical encoding (original categories, original byte encoding) --
    cats_kimden = ["Sahibinden", "Galeriden", "Yetkili Bayiden", "Rent a Car"]
    vites_cats  = ["Otomatik", "Düz", "Yarı Otomatik"]
    yakit_cats  = ["Benzin", "Dizel", "LPG & Benzin", "Hibrit", "Elektrik"]
    kasa_cats   = ["Sedan", "Hatchback/5", "Cabrio", "Coupe", "SUV",
                   "Hatchback/3", "Station wagon", "Roadster",
                   "-", "MPV", "Pick-up"]

    # -- guard: an unknown category would silently become the reference level --
    for tag, df in (("June", df1), ("December", df2)):
        for col, cats in (("vites_tipi", vites_cats), ("yakit_tipi", yakit_cats),
                          ("kasa_tipi", kasa_cats), ("kimden", cats_kimden)):
            unknown = sorted(set(df[col].dropna().unique()) - set(cats))
            if unknown:
                raise ValueError(f"{tag} {col}: unknown categories {unknown}. "
                                 "Run fix_encoding.py first.")

    for df in (df1, df2):
        df["vites_tipi"] = pd.Categorical(df["vites_tipi"], categories=vites_cats)
        df["yakit_tipi"] = pd.Categorical(df["yakit_tipi"], categories=yakit_cats)
        df["kasa_tipi"]  = pd.Categorical(df["kasa_tipi"],  categories=kasa_cats)
        df["kimden"]     = pd.Categorical(df["kimden"],      categories=cats_kimden)

    df1 = pd.get_dummies(df1, columns=["vites_tipi", "yakit_tipi", "kasa_tipi", "kimden"], drop_first=True)
    df2 = pd.get_dummies(df2, columns=["vites_tipi", "yakit_tipi", "kasa_tipi", "kimden"], drop_first=True)

    # -- normalize target --
    pcol = "price_norm"
    df1[pcol] = df1["fiyat"] / norm_haz
    df2[pcol] = df2["fiyat"] / norm_ara

    # -- smoothed target encoding (train-only statistics) --
    k = cfg["te_smoothing_k"]
    mu_global = df1[pcol].mean()

    marka_stats = (
        df1.groupby("marka")[pcol].agg(["mean", "count"])
        .rename(columns={"mean": "mu_marka", "count": "n_marka"})
    )
    ms_stats = (
        df1.groupby(["marka", "seri"])[pcol].agg(["mean", "count"])
        .rename(columns={"mean": "mu_ms", "count": "n_ms"})
    )
    ms_stats = ms_stats.merge(marka_stats["mu_marka"], left_on="marka",
                              right_index=True, how="left")
    ms_stats["TE_ms"] = (
        ms_stats["n_ms"] * ms_stats["mu_ms"] + k * ms_stats["mu_marka"]
    ) / (ms_stats["n_ms"] + k)

    df1 = df1.merge(ms_stats[["TE_ms"]], left_on=["marka", "seri"], right_index=True, how="left")
    df2 = df2.merge(ms_stats[["TE_ms"]], left_on=["marka", "seri"], right_index=True, how="left")
    df2 = df2.merge(marka_stats[["mu_marka"]], left_on="marka", right_index=True, how="left")
    df2["TE_ms"] = df2["TE_ms"].fillna(df2["mu_marka"])
    df2["TE_ms"] = df2["TE_ms"].fillna(mu_global)

    # -- NaN filling with train medians --
    num_cols = df1.select_dtypes(include=["int64", "float64"]).columns
    num_cols = num_cols.intersection(df2.columns)
    medians = df1[num_cols].median()
    df1[num_cols] = df1[num_cols].fillna(medians)
    df2[num_cols] = df2[num_cols].fillna(medians)

    df1.drop(columns=["marka", "seri"], inplace=True)
    df2.drop(columns=["marka", "seri", "mu_marka"], inplace=True, errors="ignore")

    # -- split --
    y_train = df1[pcol].copy()
    y_test  = df2[pcol].copy()
    X_train = df1.drop(columns=[pcol, "fiyat"]).copy()
    X_test  = df2.drop(columns=[pcol, "fiyat"]).copy()

    common = X_train.columns.intersection(X_test.columns)
    return X_train[common], X_test[common], y_train, y_test


def _calc(y_true, y_pred):
    return {
        "R2":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


# ============================================================================
# 2. TABLE 1 — target-encoding smoothing sensitivity
# ============================================================================

def table1_k_sensitivity(cfg):
    from catboost import CatBoostRegressor
    print("\n[Table 1] Target-encoding smoothing sensitivity ...")
    rows = []
    for k in cfg["k_list"]:
        c = dict(cfg); c["te_smoothing_k"] = k
        X_tr, X_te, y_tr, y_te = preprocess(c, cfg["USD_HAZ"], cfg["USD_ARA"])
        m = CatBoostRegressor(iterations=cfg["cb_iterations"], depth=6,
                              learning_rate=0.05, verbose=0, random_state=42)
        m.fit(X_tr.values, y_tr)
        tr = r2_score(y_tr, m.predict(X_tr.values))
        te = r2_score(y_te, m.predict(X_te.values))
        rows.append({"k": k, "Train R2": tr, "Test R2": te, "Delta R2": tr - te})
        print(f"    k={k:>3} | Train={tr:.4f} Test={te:.4f} d={tr-te:.4f}")
    return pd.DataFrame(rows)


# ============================================================================
# 3. TABLE 3 — descriptive statistics (computed from data, not hardcoded)
# ============================================================================

def _count_raw_records(cfg):
    """True vehicle counts from the raw source files (data rows, header excluded).
    Falls back to documented constants if the raw files are absent."""
    haz = None
    try:
        with open(cfg["haziran_raw"], "rb") as f:
            haz = sum(1 for _ in f) - 1            # newlines minus header
    except Exception:
        pass
    ara = None
    try:
        import openpyxl
        wb = openpyxl.load_workbook(cfg["aralik_raw"], read_only=True)
        ara = wb.active.max_row - 1                # rows minus header
        wb.close()
    except Exception:
        pass
    # documented fallbacks (raw data rows, header excluded)
    return (haz if haz is not None else 50755,
            ara if ara is not None else 52051)


def table3_descriptive(cfg):
    print("\n[Table 3] Descriptive statistics ...")
    dh = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(cfg["aralik_csv"],  encoding="utf-8-sig", sep=",")
    raw_h, raw_a = _count_raw_records(cfg)

    def chg(a, b):
        return f"+{(b / a - 1) * 100:.1f}%"

    ph, pa = dh["fiyat"], da["fiyat"]
    bs_h = dh[["marka", "seri"]].drop_duplicates().shape[0]
    bs_a = da[["marka", "seri"]].drop_duplicates().shape[0]

    rows = [
        {"Parameter": "Source",              "June 2025": "arabam.com",   "December 2025": "arabam.com",   "Change": "—"},
        {"Parameter": "Raw Records",         "June 2025": raw_h,          "December 2025": raw_a,          "Change": chg(raw_h, raw_a)},
        {"Parameter": "After Cleaning",      "June 2025": len(dh),        "December 2025": len(da),        "Change": chg(len(dh), len(da))},
        {"Parameter": "Mean Price (TL)",     "June 2025": round(ph.mean()),   "December 2025": round(pa.mean()),   "Change": chg(ph.mean(), pa.mean())},
        {"Parameter": "Median Price (TL)",   "June 2025": round(ph.median()), "December 2025": round(pa.median()), "Change": chg(ph.median(), pa.median())},
        {"Parameter": "Min Price (TL)",      "June 2025": round(ph.min()),    "December 2025": round(pa.min()),    "Change": "—"},
        {"Parameter": "Max Price (TL)",      "June 2025": round(ph.max()),    "December 2025": round(pa.max()),    "Change": "—"},
        {"Parameter": "Unique Brands",       "June 2025": dh["marka"].nunique(), "December 2025": da["marka"].nunique(), "Change": "—"},
        {"Parameter": "Brand-Series Combos", "June 2025": bs_h,           "December 2025": bs_a,           "Change": "—"},
        {"Parameter": "",                    "June 2025": "",             "December 2025": "",             "Change": ""},
        {"Parameter": "USD/TRY Rate",        "June 2025": cfg["USD_HAZ"], "December 2025": cfg["USD_ARA"], "Change": chg(cfg["USD_HAZ"], cfg["USD_ARA"])},
        {"Parameter": "EUR/TRY Rate",        "June 2025": cfg["EUR_HAZ"], "December 2025": cfg["EUR_ARA"], "Change": chg(cfg["EUR_HAZ"], cfg["EUR_ARA"])},
        {"Parameter": "Gold (TL/gram)",      "June 2025": cfg["GOLD_HAZ"],"December 2025": cfg["GOLD_ARA"],"Change": chg(cfg["GOLD_HAZ"], cfg["GOLD_ARA"])},
        {"Parameter": "CPI Index",           "June 2025": cfg["CPI_HAZ"], "December 2025": cfg["CPI_ARA"], "Change": chg(cfg["CPI_HAZ"], cfg["CPI_ARA"])},
    ]
    df = pd.DataFrame(rows)
    print(f"    N (clean): {len(dh):,} / {len(da):,}  | raw: {raw_h:,} / {raw_a:,}")
    return df


# ============================================================================
# 4. TABLE 4 — normalization strategy comparison (CatBoost)
# ============================================================================

def table4_normalization(cfg):
    """
    CV loop uses cb_iterations (=1000), consistent with the final fit and with
    Table 5.
    """
    from catboost import CatBoostRegressor
    print("\n[Table 4] Normalization strategy comparison ...")
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])
    rows = []
    preds = {}
    for name, nh, na in cfg["strategies"]:
        t0 = time.time()
        X_tr, X_te, y_tr, y_te = preprocess(cfg, nh, na)

        cv = []
        for tri, vai in kf.split(X_tr):
            m = CatBoostRegressor(iterations=cfg["cb_iterations"], depth=6,
                                  learning_rate=0.05, verbose=0, random_state=42)
            m.fit(X_tr.values[tri], y_tr.values[tri])
            cv.append(r2_score(y_tr.values[vai], m.predict(X_tr.values[vai])))

        m_full = CatBoostRegressor(iterations=cfg["cb_iterations"], depth=6,
                                   learning_rate=0.05, verbose=0, random_state=42)
        m_full.fit(X_tr.values, y_tr)
        p_te = m_full.predict(X_te.values)
        tr = _calc(y_tr, m_full.predict(X_tr.values))
        te = _calc(y_te, p_te)
        preds[name] = p_te * na
        preds["actual_TL"] = y_te.values * na

        rows.append({
            "Strategy": name, "Rate (Jun)": nh, "Rate (Dec)": na,
            "Rate Change (%)": (na / nh - 1) * 100,
            "CV R2": np.mean(cv), "CV R2 std": np.std(cv),
            "Train R2": tr["R2"], "Test R2": te["R2"], "Δ(R2)": tr["R2"] - te["R2"],
            "Test MAPE": te["MAPE"], "Test MAE": te["MAE"], "Test RMSE": te["RMSE"],
            "Test MAE (TL)": te["MAE"] * na, "Test RMSE (TL)": te["RMSE"] * na,
        })
        print(f"    {name:12s} | Test={te['R2']:.4f} CV={np.mean(cv):.4f} "
              f"MAPE={te['MAPE']*100:.2f}% | {time.time()-t0:.0f}s")
    df = pd.DataFrame(rows).sort_values("Test R2", ascending=False).reset_index(drop=True)
    return df, pd.DataFrame(preds)


# ============================================================================
# 5. TABLE 5 — model comparison (9 models + 2 ablations)  [+ returns USD data]
# ============================================================================

def table5_models(cfg, norm_haz, norm_ara):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor

    print("\n[Table 5] Model comparison (USD/TRY) ...")
    X_tr, X_te, y_tr, y_te = preprocess(cfg, norm_haz, norm_ara)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])
    cbi = cfg["cb_iterations"]
    mlp_it = cfg["mlp_max_iter"]

    # (name, factory(seed), needs_scaling, deterministic, feature_mode)
    model_defs = [
        ("Linear Regression", lambda s: LinearRegression(),                                      True,  True,  None),
        ("Ridge",             lambda s: Ridge(alpha=10),                                          True,  True,  None),
        ("Lasso",             lambda s: Lasso(alpha=0.1),                                         True,  True,  None),
        ("ElasticNet",        lambda s: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),       True,  True,  None),
        ("Random Forest",     lambda s: RandomForestRegressor(n_estimators=200, max_depth=15,
                                                              min_samples_leaf=5, random_state=s, n_jobs=-1), False, False, None),
        ("XGBoost",           lambda s: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                                     subsample=0.8, colsample_bytree=0.8, random_state=s,
                                                     verbosity=0, n_jobs=-1),                     False, False, None),
        ("LightGBM",          lambda s: LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                                      subsample=0.8, colsample_bytree=0.8, random_state=s,
                                                      verbose=-1, n_jobs=-1),                     False, False, None),
        ("CatBoost",          lambda s: CatBoostRegressor(iterations=cbi, depth=6, learning_rate=0.05,
                                                          verbose=0, random_state=s),             False, False, None),
        ("MLP",               lambda s: MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=mlp_it,
                                                     early_stopping=True, random_state=s,
                                                     learning_rate_init=0.001),                   True,  False, None),
        ("CatBoost_NO_TE",    lambda s: CatBoostRegressor(iterations=cbi, depth=6, learning_rate=0.05,
                                                          verbose=0, random_state=s),             False, False, "NO_TE"),
        ("Linear_TE_only",    lambda s: LinearRegression(),                                       False, True,  "TE_ONLY"),
    ]

    rows = []
    preds = {}
    for name, factory, needs_sc, is_det, mode in model_defs:
        t0 = time.time()
        if mode == "NO_TE":
            Xtr, Xte = X_tr.drop(columns=["TE_ms"]).values, X_te.drop(columns=["TE_ms"]).values
        elif mode == "TE_ONLY":
            Xtr, Xte = X_tr[["TE_ms"]].values, X_te[["TE_ms"]].values
        elif needs_sc:
            Xtr, Xte = X_tr_sc, X_te_sc
        else:
            Xtr, Xte = X_tr.values, X_te.values

        cv = []
        for tri, vai in kf.split(Xtr):
            mc = factory(42); mc.fit(Xtr[tri], y_tr.values[tri])
            cv.append(r2_score(y_tr.values[vai], mc.predict(Xtr[vai])))

        seeds = [42] if is_det else cfg["seeds"]
        tr_r2, te_r2, tr_mae, te_mae, tr_rmse, te_rmse, tr_mape, te_mape = ([] for _ in range(8))
        for s in seeds:
            m = factory(s); m.fit(Xtr, y_tr)
            pt, pe = m.predict(Xtr), m.predict(Xte)
            preds[f"{name}|seed{s}"] = pe
            tr_r2.append(r2_score(y_tr, pt));   te_r2.append(r2_score(y_te, pe))
            tr_mae.append(mean_absolute_error(y_tr, pt)); te_mae.append(mean_absolute_error(y_te, pe))
            tr_rmse.append(np.sqrt(mean_squared_error(y_tr, pt))); te_rmse.append(np.sqrt(mean_squared_error(y_te, pe)))
            tr_mape.append(mean_absolute_percentage_error(y_tr, pt)); te_mape.append(mean_absolute_percentage_error(y_te, pe))

        rows.append({
            "Model": name, "CV R2": np.mean(cv), "CV R2 std": np.std(cv),
            "Train R2": np.mean(tr_r2), "Test R2": np.mean(te_r2),
            "Test R2 std": np.std(te_r2) if len(te_r2) > 1 else 0.0,
            "Train MAE": np.mean(tr_mae), "Test MAE": np.mean(te_mae),
            "Train RMSE": np.mean(tr_rmse), "Test RMSE": np.mean(te_rmse),
            "Train MAPE": np.mean(tr_mape), "Test MAPE": np.mean(te_mape),
            "Test MAE (TL)": np.mean(te_mae) * norm_ara, "Test RMSE (TL)": np.mean(te_rmse) * norm_ara,
            "Δ(R2)": np.mean(tr_r2) - np.mean(te_r2),
        })
        print(f"    {name:20s} | CV={np.mean(cv):.4f} Test={np.mean(te_r2):.4f} "
              f"MAPE={np.mean(te_mape)*100:.2f}% | {time.time()-t0:.0f}s")
    preds["actual_USD"] = y_te.values
    return pd.DataFrame(rows), X_tr, X_te, y_tr, y_te, pd.DataFrame(preds)


# ============================================================================
# 6. TABLE 6 — period-dummy analysis
# ============================================================================

def table7_period_dummy(cfg, X_tr, X_te, y_tr, y_te):
    """
    A period indicator is appended to the pooled June+December feature matrix and
    its permutation/importance rank is inspected.

    Only XGBoost and CatBoost are reported (as in the manuscript). LightGBM is
    excluded because its raw split-count importance is not comparable in scale to
    the other learners' importances and would be misleading in the same ranking.

    NOTE: CatBoost here uses cb_iterations_period (=500), a DIAGNOSTIC budget. The
    period rank/importance and the "period is negligible" conclusion are
    insensitive to the exact budget.
    """
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    print("\n[Table 7] Period-dummy analysis ...")

    Xa = X_tr.copy(); Xa["period"] = 0
    Xb = X_te.copy(); Xb["period"] = 1
    X_all = pd.concat([Xa, Xb], ignore_index=True)
    y_all = pd.concat([y_tr, y_te], ignore_index=True)
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])

    specs = [
        ("XGBoost", XGBRegressor, dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbosity=0, n_jobs=-1)),
        ("CatBoost", CatBoostRegressor, dict(iterations=cfg["cb_iterations_period"], depth=6,
                                             learning_rate=0.05, verbose=0, random_state=42)),
    ]
    rows = []
    for name, cls, params in specs:
        cv = []
        for tri, vai in kf.split(X_all):
            m = cls(**params); m.fit(X_all.values[tri], y_all.values[tri])
            cv.append(r2_score(y_all.values[vai], m.predict(X_all.values[vai])))
        m_full = cls(**params); m_full.fit(X_all.values, y_all.values)
        imp = (m_full.feature_importances_ if hasattr(m_full, "feature_importances_")
               else m_full.get_feature_importance())
        fi = pd.DataFrame({"feature": X_all.columns, "importance": imp}).sort_values(
            "importance", ascending=False).reset_index(drop=True)
        p_imp = float(fi.loc[fi["feature"] == "period", "importance"].iloc[0])
        p_rank = int(fi.index[fi["feature"] == "period"][0]) + 1
        rows.append({
            "Model": name, "Combined CV R2": np.mean(cv), "CV R2 std": np.std(cv),
            "Period Importance": p_imp, "Period Rank": p_rank, "Total Features": len(fi),
        })
        print(f"    {name:10s} | CV={np.mean(cv):.4f} period_imp={p_imp:.4f} "
              f"rank={p_rank}/{len(fi)}")
    return pd.DataFrame(rows)


# ============================================================================
# 7. TABLES 8 & 9 — age distribution & within-band USD prices
# ============================================================================

def tables_8_9_age(cfg):
    print("\n[Tables 8 & 9] Age distribution & within-band USD prices ...")
    Xh, Xa, yh, ya = preprocess(cfg, cfg["USD_HAZ"], cfg["USD_ARA"])
    bins = [-1, 3, 5, 8, 12, 200]
    labels = ["≤3", "4–5", "6–8", "9–12", "≥13"]
    h = pd.DataFrame({"age": Xh["age"].values, "usd": yh.values})
    a = pd.DataFrame({"age": Xa["age"].values, "usd": ya.values})
    h["band"] = pd.cut(h["age"], bins=bins, labels=labels)
    a["band"] = pd.cut(a["age"], bins=bins, labels=labels)

    hj = (h.groupby("band", observed=False).size() / len(h) * 100).reindex(labels)
    aj = (a.groupby("band", observed=False).size() / len(a) * 100).reindex(labels)
    hu = h.groupby("band", observed=False)["usd"].mean().reindex(labels)
    au = a.groupby("band", observed=False)["usd"].mean().reindex(labels)

    t7 = pd.DataFrame({"Age band (years)": labels,
                       "June (%)": hj.round(2).values,
                       "December (%)": aj.round(2).values})
    t7 = pd.concat([t7, pd.DataFrame([{
        "Age band (years)": "Mean age",
        "June (%)": round(float(h["age"].mean()), 2),
        "December (%)": round(float(a["age"].mean()), 2)}])], ignore_index=True)

    t8 = pd.DataFrame({"Age band": labels,
                       "June (USD)": hu.round().values,
                       "December (USD)": au.round().values,
                       "Within-band Δ (%)": ((au / hu - 1) * 100).round(2).values})
    print(f"    mean age June/Dec: {h['age'].mean():.2f} / {a['age'].mean():.2f}")
    return t7, t8


# ============================================================================
# 8. Canonical CatBoost fit on USD data (shared by Tables 10, 11 and figures)
# ============================================================================

def fit_canonical_usd(cfg, X_tr, X_te, y_tr):
    """Single seed-42 CatBoost(iterations=cb_iterations) on USD data — the model
    used for Table 10, Table 11 and all diagnostic figures. Fit once, reuse."""
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=cfg["cb_iterations"], depth=6,
                          learning_rate=0.05, verbose=0, random_state=42)
    m.fit(X_tr.values, y_tr)
    return m, m.predict(X_te.values)


# ============================================================================
# 9. TABLE 10 — prediction error by price quintile
# ============================================================================

def table10_quintiles(y_test, y_pred):
    print("\n[Table 10] Error by price quintile ...")
    yt = y_test.values
    q = pd.qcut(yt, 5, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4", "Q5 (most expensive)"])
    rows = []
    for g in q.categories:
        mask = (q == g)
        rows.append({
            "Price quintile": g, "N": int(mask.sum()),
            "Mean (USD)": round(yt[mask].mean()),
            "MAE (USD)": round(mean_absolute_error(yt[mask], y_pred[mask])),
            "RMSE (USD)": round(np.sqrt(mean_squared_error(yt[mask], y_pred[mask]))),
            "MAPE (%)": round(mean_absolute_percentage_error(yt[mask], y_pred[mask]) * 100, 2),
        })
    return pd.DataFrame(rows)  # within-stratum R2 omitted (limited target range)


# ============================================================================
# 10. TABLE 11 — robustness: known vs. fallback brand-series
# ============================================================================

def table11_robustness(cfg, y_test, y_pred):
    print("\n[Table 11] Robustness (known vs. fallback) ...")
    dh = pd.read_csv(cfg["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(cfg["aralik_csv"],  encoding="utf-8-sig", sep=",")
    haz_pairs = set(zip(dh["marka"], dh["seri"]))
    haz_markas = set(dh["marka"].unique())
    status = np.array([
        "known" if (m, s) in haz_pairs else ("fb_marka" if m in haz_markas else "fb_global")
        for m, s in zip(da["marka"], da["seri"])
    ])
    N = len(status)
    groups = [
        ("All Test Data",                              np.ones(N, dtype=bool)),
        ("Known (brand–series seen in training)",      status == "known"),
        ("Fallback (all)",                             status != "known"),
        ("Brand fallback (known brand, new series)",   status == "fb_marka"),
        ("Global fallback (unknown brand)",            status == "fb_global"),
    ]
    yt = y_test.values
    rows = []
    for name, mask in groups:
        if mask.sum() == 0:
            continue
        r2 = r2_score(yt[mask], y_pred[mask])
        mae = mean_absolute_error(yt[mask], y_pred[mask])
        rmse = np.sqrt(mean_squared_error(yt[mask], y_pred[mask]))
        mape = mean_absolute_percentage_error(yt[mask], y_pred[mask])
        rows.append({
            "Group": name, "N": int(mask.sum()), "N (%)": round(mask.sum() / N * 100, 1),
            "R2": round(r2, 4), "MAE": round(mae, 1), "RMSE": round(rmse, 1),
            "MAPE (%)": round(mape * 100, 2),
            "MAE (TL)": round(mae * cfg["USD_ARA"]), "RMSE (TL)": round(rmse * cfg["USD_ARA"]),
        })
    return pd.DataFrame(rows)

# ============================================================================
# 10b. TABLE 6 — HYPERPARAMETER TUNING, MLP & Random Forest  (Reviewer 3-G, 2-Q17/Q18)
# ============================================================================

def table6_hyperparameter_tuning(cfg, norm_haz, norm_ara, t5):
    """
    Randomised search (n_iter candidates, 5-fold CV on the TRAIN set) for the two
    parameter-sensitive models. Gradient-boosting models are intentionally NOT
    retuned: three independent implementations already converge to the same test
    R2, so the ceiling is set by data/features, not model-specific hyperparameters.
    Default (untuned) Test R2 values are read from the MLP/RF rows of Table 5
    (passed in as t5), so they always match the current run.
    """
    def _default(name):
        return round(float(t5.loc[t5["Model"] == name, "Test R2"].iloc[0]), 4)
    from sklearn.model_selection import RandomizedSearchCV, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    print("\n[Table 6] Randomised search: MLP and Random Forest ...")

    X_tr, X_te, y_tr, y_te = preprocess(cfg, norm_haz, norm_ara)
    kf = KFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["cv_random_state"])
    n_iter = cfg["tune_n_iter"]
    rows = []

    # --- MLP (scaled inputs) ---
    sc = StandardScaler(); Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
    mlp_grid = {
        "hidden_layer_sizes": [(128, 64, 32), (256, 128, 64), (128, 64), (64, 32), (256, 128, 64, 32)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [5e-4, 1e-3, 3e-3],
        "activation": ["relu", "tanh"],
    }
    mlp = RandomizedSearchCV(
        MLPRegressor(max_iter=cfg["mlp_max_iter"], early_stopping=True, random_state=42),
        mlp_grid, n_iter=n_iter, cv=kf, scoring="r2", n_jobs=-1, random_state=42)
    mlp.fit(Xtr, y_tr)
    rows.append({"Model": "MLP", "Default Test R2": _default("MLP"),
                 "Tuned CV R2": round(mlp.best_score_, 4),
                 "Tuned Test R2": round(r2_score(y_te, mlp.predict(Xte)), 4),
                 "Best params": str(mlp.best_params_)})
    print(f"    MLP  tuned Test R2={r2_score(y_te, mlp.predict(Xte)):.4f} (default {_default('MLP')})")

    # --- Random Forest (unscaled inputs) ---
    rf_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [12, 15, 20, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": ["sqrt", 0.5, 1.0],
    }
    rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        rf_grid, n_iter=n_iter, cv=kf, scoring="r2", n_jobs=-1, random_state=42)
    rf.fit(X_tr.values, y_tr)
    rows.append({"Model": "Random Forest", "Default Test R2": _default("Random Forest"),
                 "Tuned CV R2": round(rf.best_score_, 4),
                 "Tuned Test R2": round(r2_score(y_te, rf.predict(X_te.values)), 4),
                 "Best params": str(rf.best_params_)})
    print(f"    RF   tuned Test R2={r2_score(y_te, rf.predict(X_te.values)):.4f} (default {_default('Random Forest')})")
    return pd.DataFrame(rows)


# ============================================================================
# 10c. DECEMBER-RATE ROBUSTNESS  (Reviewer 3-E)
# ============================================================================

def dec_rate_robustness(cfg, norm_haz):
    """Robustness of the December deflator: forward-looking end-of-November proxy
    (42.4321) vs. the actual end-of-December rate (42.9340)."""
    from catboost import CatBoostRegressor
    print("\n[Dec-rate] December deflator robustness ...")
    rows = []
    for label, ara in [("End-of-November proxy (42.4321)", cfg["USD_ARA"]),
                       ("Actual end-of-December (42.9340)", 42.9340)]:
        X_tr, X_te, y_tr, y_te = preprocess(cfg, norm_haz, ara)
        m = CatBoostRegressor(iterations=cfg["cb_iterations"], depth=6,
                              learning_rate=0.05, verbose=0, random_state=42)
        m.fit(X_tr.values, y_tr); yp = m.predict(X_te.values)
        rows.append({"December deflator": label,
                     "Test R2": round(r2_score(y_te, yp), 4),
                     "MAE (USD)": round(mean_absolute_error(y_te, yp)),
                     "MAPE (%)": round(mean_absolute_percentage_error(y_te, yp) * 100, 2)})
    return pd.DataFrame(rows)
# ============================================================================
# 11. FIGURES  (Times New Roman, bold axis labels, no titles, thousand-USD scaling, 300 DPI)
# ============================================================================

# English rename map for publication-quality figures (keys keep original bytes).
RENAME_MAP = {
    "degisen_sayisi": "Replaced_parts", "boyali_sayisi": "Repainted_parts",
    "age": "Vehicle_age", "log_km": "Log_mileage", "power_per_cc": "Power_density",
    "vites_tipi_Düz": "Transmission_manual", "vites_tipi_Yarı Otomatik": "Transmission_semi_auto",
    "yakit_tipi_Dizel": "Fuel_diesel", "yakit_tipi_LPG & Benzin": "Fuel_LPG",
    "yakit_tipi_Hibrit": "Fuel_hybrid", "yakit_tipi_Elektrik": "Fuel_electric",
    "kasa_tipi_Hatchback/5": "Body_hatchback5", "kasa_tipi_Cabrio": "Body_cabrio",
    "kasa_tipi_Coupe": "Body_coupe", "kasa_tipi_SUV": "Body_SUV",
    "kasa_tipi_Hatchback/3": "Body_hatchback3", "kasa_tipi_Station wagon": "Body_wagon",
    "kasa_tipi_Roadster": "Body_roadster", "kasa_tipi_-": "Body_other",
    "kasa_tipi_MPV": "Body_MPV", "kasa_tipi_Pick-up": "Body_pickup",
    "kimden_Galeriden": "Seller_dealer", "kimden_Yetkili Bayiden": "Seller_authorized",
    "kimden_Rent a Car": "Seller_rental", "TE_ms": "TE_brand_series",
}


def make_figures(cfg, table4_df, model, X_tr, X_te, y_tr, y_te):
    print("\n[Figures] Generating and saving PNGs ...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.family"] = "Times New Roman"
    out = cfg["output_dir"]

    def _bold_axes(ax):
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold")
        ax.set_ylabel(ax.get_ylabel(), fontweight="bold")

    # ---- Figure 2: normalization R2 comparison (CV / Train / Test) ----
    d = table4_df.set_index("Strategy")
    names = list(d.index)
    x = np.arange(len(names)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, d["CV R2"].values,   w, label="Cross-validation")
    ax.bar(x,     d["Train R2"].values, w, label="Train")
    ax.bar(x + w, d["Test R2"].values,  w, label="Test")
    ax.set_xticks(x); ax.set_xticklabels(names)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylabel("R² Score", fontweight="bold"); ax.set_ylim(0, 1.0)
    ax.legend(prop={"weight": "bold"})
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig2_normalization_r2.png"), dpi=300)
    plt.close()

    # SHAP-based figures use the shared canonical model (English feature names).
    try:
        import shap
        Xtr = X_tr.rename(columns=RENAME_MAP)
        Xte = X_te.rename(columns=RENAME_MAP)
        explainer = shap.TreeExplainer(model)

        for tag, X in (("train", Xtr), ("test", Xte)):
            sv = explainer.shap_values(X)
            imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}
                               ).sort_values("mean_abs_shap", ascending=False)
            imp.to_csv(os.path.join(out, f"shap_{tag}.csv"), index=False)
            fig, _ = plt.subplots(figsize=(10, 7))
            shap.summary_plot(sv, X, show=False)
            axs = plt.gca()
            axs.set_xlabel(axs.get_xlabel(), fontweight="bold")
            axs.set_ylabel(axs.get_ylabel(), fontweight="bold")
            for lbl in axs.get_yticklabels():
                lbl.set_fontweight("bold")
            for cb in fig.axes:
                if cb is not axs:
                    cb.set_ylabel(cb.get_ylabel(), fontweight="bold")
                    for lbl in cb.get_yticklabels():
                        lbl.set_fontweight("bold")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"shap_{tag}.png"), dpi=300)
            plt.close()

        # SHAP dependence — brand-series target encoding
        sv_test = explainer.shap_values(Xte)
        fig, _ = plt.subplots(figsize=(7, 5))
        shap.dependence_plot("TE_brand_series", sv_test, Xte, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out, "shap_dependence_TE_brand_series.png"), dpi=300)
        plt.close()
    except Exception as e:
        print(f"    [warn] SHAP figures skipped: {e}")

    # ---- Predicted vs. actual (thousand USD; the target is USD-normalized) ----
    y_pred = model.predict(X_te.values)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_te / 1000, y_pred / 1000, alpha=0.3, s=5)
    lims = [min(y_te.min(), y_pred.min()) / 1000, max(y_te.max(), y_pred.max()) / 1000]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Price (thousand USD)", fontweight="bold")
    ax.set_ylabel("Predicted Price (thousand USD)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "pred_vs_actual.png"), dpi=300)
    plt.close()

    # ---- Residual distribution (thousand USD) ----
    resid = (y_te.values - y_pred) / 1000
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=80, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Residual (thousand USD)", fontweight="bold")
    ax.set_ylabel("Frequency", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "residuals.png"), dpi=300)
    plt.close()

    # ---- CatBoost native feature importance (top 15) ----
    fi = pd.DataFrame({"feature": X_tr.columns, "importance": model.get_feature_importance()}
                      ).sort_values("importance", ascending=False)
    fi["feature"] = fi["feature"].replace(RENAME_MAP)
    fi.to_csv(os.path.join(out, "catboost_feature_importance.csv"), index=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi["feature"][:15], fi["importance"][:15])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score", fontweight="bold")
    ax.set_ylabel("Feature", fontweight="bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "feature_importance.png"), dpi=300)
    plt.close()
    print(f"    figures + SHAP CSVs saved under {out}/")


# ============================================================================
# 12. EXCEL WRITER  (one formatted sheet per table)
# ============================================================================

def save_excel(sheets, cfg):
    """sheets: list of (sheet_name, title, dataframe_or_text)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    print("\n[Excel] Writing workbook ...")
    wb = Workbook()
    hfill = PatternFill("solid", fgColor="2F5496")
    hfont = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    dfont = Font(name="Arial", size=10)
    tfont = Font(name="Arial", bold=True, size=13, color="2F5496")
    brd = Border(left=Side("thin"), right=Side("thin"), top=Side("thin"), bottom=Side("thin"))

    def write_df(ws, df, title):
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max(len(df.columns), 1))
        ws.cell(row=1, column=1, value=title).font = tfont
        r = 3
        for c, col in enumerate(df.columns, 1):
            cell = ws.cell(row=r, column=c, value=str(col))
            cell.fill = hfill; cell.font = hfont; cell.border = brd
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        r += 1
        for _, rowdata in df.iterrows():
            for c, val in enumerate(rowdata, 1):
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                cell = ws.cell(row=r, column=c, value=val)
                cell.font = dfont; cell.border = brd
                cell.alignment = Alignment(horizontal="center")
                if isinstance(val, float):
                    if abs(val) < 1:
                        cell.number_format = "0.0000"
                    elif abs(val) < 100:
                        cell.number_format = "0.00"
                    else:
                        cell.number_format = "#,##0"
            r += 1
        # column widths
        for c, col in enumerate(df.columns, 1):
            width = max(len(str(col)) + 2,
                        *(len(str(v)) + 2 for v in df.iloc[:, c - 1].tolist())) if len(df) else len(str(col)) + 2
            ws.column_dimensions[get_column_letter(c)].width = min(max(width, 10), 40)

    def write_text(ws, title, text):
        ws.cell(row=1, column=1, value=title).font = tfont
        for i, line in enumerate(text.strip().split("\n"), start=3):
            ws.cell(row=i, column=1, value=line).font = dfont
        ws.column_dimensions["A"].width = 100

    first = True
    for name, title, payload in sheets:
        ws = wb.active if first else wb.create_sheet()
        ws.title = name[:31]
        first = False
        if isinstance(payload, pd.DataFrame):
            write_df(ws, payload, title)
        else:
            write_text(ws, title, payload)

    path = os.path.join(cfg["output_dir"], cfg["output_excel"])
    wb.save(path)
    print(f"    saved: {path}")
    return path


# Definitional (non-computed) reference sheets so the workbook contains Tables 1-12.
TABLE2_TEXT = """
Table 2 — Evaluation metrics used in this study (definitional; no computed values).
  R2   = 1 - Σ(y_i - ŷ_i)^2 / Σ(y_i - ȳ)^2        (explained variance)
  MAE  = (1/n) Σ |y_i - ŷ_i|                        (average absolute error)
  RMSE = sqrt( (1/n) Σ (y_i - ŷ_i)^2 )              (penalizes large errors)
  MAPE = (1/n) Σ |(y_i - ŷ_i) / y_i|                (relative error)
"""

TABLE12_TEXT = """
Table 12 — Positioning of the proposed framework against representative prior work
(definitional / narrative; no computed values). See manuscript Section 7.2 for the
comparison across: temporal (train-then-shift) evaluation, currency normalization
as the temporal-alignment mechanism, exchange-rate pass-through decomposition, and
target-encoding robustness under unseen brand-series combinations.
"""


# ============================================================================
# 13. MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("FULL REPRODUCTION — all tables + figures")
    print("=" * 80)

    usd_h, usd_a = CONFIG["USD_HAZ"], CONFIG["USD_ARA"]

    t1  = table1_k_sensitivity(CONFIG)
    t3  = table3_descriptive(CONFIG)
    t4, norm_preds = table4_normalization(CONFIG)
    t5, X_tr, X_te, y_tr, y_te, model_preds = table5_models(CONFIG, usd_h, usd_a)
    t7  = table7_period_dummy(CONFIG, X_tr, X_te, y_tr, y_te)
    t8, t9 = tables_8_9_age(CONFIG)

    model, y_pred = fit_canonical_usd(CONFIG, X_tr, X_te, y_tr)   # shared model
    t10 = table10_quintiles(y_te, y_pred)
    t11 = table11_robustness(CONFIG, y_te, y_pred)
    model_preds.to_csv(os.path.join(CONFIG["output_dir"], "predictions_models.csv"), index=False)
    norm_preds.to_csv(os.path.join(CONFIG["output_dir"], "predictions_normalization.csv"), index=False)

    t6 = table6_hyperparameter_tuning(CONFIG, usd_h, usd_a, t5)
    t_dec = dec_rate_robustness(CONFIG, usd_h)

    make_figures(CONFIG, t4, model, X_tr, X_te, y_tr, y_te)

    sheets = [
        ("Table 1 - k sensitivity", "Table 1: Sensitivity to target-encoding smoothing k (CatBoost, USD/TRY)", t1),
        ("Table 2 - Metrics",       "Table 2: Evaluation metrics", TABLE2_TEXT),
        ("Table 3 - Descriptives",  "Table 3: Descriptive statistics and macroeconomic indicators", t3),
        ("Table 4 - Normalization", "Table 4: Normalization strategy comparison (CatBoost)", t4),
        ("Table 5 - Models",        "Table 5: Model comparison under USD/TRY normalization", t5),
        ("Table 6 - Tuning",        "Table 6: Hyperparameter tuning (randomised search), MLP and Random Forest", t6),
        ("Table 7 - Period dummy",  "Table 7: Period-dummy analysis (combined June–December)", t7),
        ("Table 8 - Age dist",      "Table 8: Age-distribution shift between June and December 2025", t8),
        ("Table 9 - Within-band",   "Table 9: Within-age-band USD price comparison across periods", t9),
        ("Table 10 - Quintiles",    "Table 10: Prediction error by price quintile (December test set)", t10),
        ("Table 11 - Robustness",   "Table 11: Robustness — known vs. fallback brand-series combinations", t11),
        ("Table 12 - Positioning",  "Table 12: Positioning against prior work", TABLE12_TEXT),
        ("Sec 6.5 - Dec-rate",      "Section 6.5: December-deflator robustness, end-of-November proxy vs. actual end-of-December", t_dec),
    ]
    save_excel(sheets, CONFIG)

    print(f"\n{'=' * 80}\nDONE in {(time.time() - t0) / 60:.1f} min — outputs in {CONFIG['output_dir']}/\n{'=' * 80}")


if __name__ == "__main__":
    main()
