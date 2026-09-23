"""leave-one-brand-out cold-start simulation (Supplementary Table S13).

Reviewer #4 asked whether a richer fallback than the global training mean helps
when a brand is entirely unseen. Only seven December vehicles belong to genuinely
new brands, which is far too few to answer the question. This script therefore
creates cold-start cases at scale from brands that ARE present in training.

For every brand b that occurs in the June training data:

  1. all June rows of brand b are removed from the training set;
  2. the brand-series target encoding, the global training mean and the
     segment-cell ladder (body type x age band x engine-size band, with a ladder
     to coarser cells, Section 7.3) are re-estimated on the reduced June data,
     so that nothing about brand b can leak into the encoder;
  3. the reference model (CatBoost, depth 6, lr 0.05, 1000 iterations, seed 42)
     and the two quantile models of the 90% valuation range are refitted on the
     reduced data;
  4. the December vehicles of brand b are scored by that model twice: once with
     the global-mean fallback (the rule used in the manuscript) and once with the
     segment-cell fallback proposed in Section 7.3.

Pooling over all held-out brands gives every December vehicle of a known brand
exactly one cold-start prediction under each rule, which is what Table S13
reports. The manuscript's own predictions (with the true brand-series encoding)
are reported alongside as the warm-start reference.

Input : Haziran_2025_son.csv, Aralık_2025_son.csv (UTF-8, i.e. after fix_encoding.py)
        and reproduce_all.py in the same folder (imported, not modified).
Output: results/reviewer4_round4.xlsx, sheet "S13 - Leave-brand-out".
Runtime: roughly 10-15 minutes.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

import reproduce_all as R           # the published pipeline: CONFIG + preprocess

CFG = R.CONFIG
K = CFG["te_smoothing_k"]
INTERVAL = (0.05, 0.95)             # same 90% range as Supplementary Table S8
AGE_BINS = [-1, 3, 5, 8, 12, 200]
AGE_LABELS = ["<=3", "4-5", "6-8", "9-12", ">=13"]
OUT_XLSX = os.path.join(CFG["output_dir"], "reviewer4_round4.xlsx")
PER_BRAND_MIN_N = 30                 # brands listed individually in Table S13b


def _cb(loss=None):
    from catboost import CatBoostRegressor
    p = dict(iterations=CFG["cb_iterations"], depth=6, learning_rate=0.05,
             verbose=0, random_state=42)
    if loss:
        p["loss_function"] = loss
    return CatBoostRegressor(**p)


def _metrics(y, p):
    err = p - y                                   # positive -> over-valuation
    return {
        "R2": r2_score(y, p),
        "MAE (USD)": mean_absolute_error(y, p),
        "RMSE (USD)": np.sqrt(mean_squared_error(y, p)),
        "MAPE (%)": mean_absolute_percentage_error(y, p) * 100,
        "Median signed error (%)": float(np.median(err / y * 100)),
    }


def _coverage(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    return {"Coverage of 90% range (%)": float(inside.mean() * 100),
            "Median width / price (%)": float(np.median(width / y) * 100)}


def _te_tables(marka, seri, price):
    """Smoothed brand-series target encoding, identical to Section 4.2."""
    d = pd.DataFrame({"marka": marka, "seri": seri, "p": price})
    mu_global = float(d["p"].mean())
    mu_marka = d.groupby("marka")["p"].mean()
    g = d.groupby(["marka", "seri"])["p"].agg(["mean", "count"])
    g = g.join(mu_marka.rename("mu_marka"), on="marka")
    te = (g["count"] * g["mean"] + K * g["mu_marka"]) / (g["count"] + K)
    return mu_global, te


def _segment_tables(kasa, ageb, ccb, price, mu_global):
    """Segment-cell ladder of Section 7.3, estimated on the reduced June data."""
    d = pd.DataFrame({"kasa": kasa, "ageb": ageb, "ccb": ccb, "p": price})

    def tbl(keys):
        g = d.groupby(keys, observed=True)["p"].agg(["sum", "count"])
        return (g["sum"] + K * mu_global) / (g["count"] + K)

    return tbl(["kasa", "ageb", "ccb"]), tbl(["kasa", "ageb"]), tbl(["ageb"])


def main():
    t0 = time.time()
    print("=" * 78)
    print("LEAVE-ONE-BRAND-OUT COLD-START SIMULATION (Table S13)")
    print("=" * 78)

    X_tr, X_te, y_tr, y_te = R.preprocess(CFG, CFG["USD_HAZ"], CFG["USD_ARA"])
    dh = pd.read_csv(CFG["haziran_csv"], encoding="utf-8-sig", sep=",")
    da = pd.read_csv(CFG["aralik_csv"], encoding="utf-8-sig", sep=",")
    if len(dh) != len(X_tr) or len(da) != len(X_te):
        raise SystemExit("CSV files and design matrices are not row-aligned.")

    y_tr_v, y_te_v = y_tr.values, y_te.values
    price_tr = dh["fiyat"].values / CFG["USD_HAZ"]

    # segment-cell keys (band edges from the June data, as in Section 7.3)
    edges = np.unique(np.nanquantile(dh["motor_hacmi"].values, [0, .25, .5, .75, 1.0]))
    edges[0], edges[-1] = -np.inf, np.inf

    def keys(df):
        age = CFG["ref_year"] - df["yil"].values
        ageb = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS).astype(str)
        ccb = pd.cut(df["motor_hacmi"].values, bins=edges, labels=False,
                     include_lowest=True)
        ccb = pd.Series(ccb).fillna(-1).astype(int).astype(str).values
        return df["kasa_tipi"].astype(str).values, np.asarray(ageb), ccb

    kasa_tr, ageb_tr, ccb_tr = keys(dh)
    kasa_te, ageb_te, ccb_te = keys(da)

    brand_tr = dh["marka"].values
    brand_te = da["marka"].values
    june_brands = set(brand_tr)

    # warm-start reference: the manuscript's own model and predictions
    print("\nfitting the reference (warm-start) model ...")
    ref_model, ref_pred = R.fit_canonical_usd(CFG, X_tr, X_te, y_tr)
    ref_q = {}
    for a in INTERVAL:
        m = _cb(loss=f"Quantile:alpha={a}")
        m.fit(X_tr.values, y_tr_v)
        ref_q[a] = m.predict(X_te.values)
    ref_lo = np.minimum(ref_q[INTERVAL[0]], ref_q[INTERVAL[1]])
    ref_hi = np.maximum(ref_q[INTERVAL[0]], ref_q[INTERVAL[1]])

    # rows that can be simulated: December vehicles whose brand exists in June
    sim_rows = np.array([b in june_brands for b in brand_te])
    brands = sorted({b for b in brand_te[sim_rows]})
    print(f"{sim_rows.sum():,} of {len(da):,} December vehicles belong to "
          f"{len(brands)} brands seen in June; each brand is held out in turn.")

    pred_A = np.full(len(da), np.nan)      # global-mean fallback (current rule)
    pred_B = np.full(len(da), np.nan)      # segment-cell fallback (Section 7.3)
    lo_A = np.full(len(da), np.nan); hi_A = np.full(len(da), np.nan)
    lo_B = np.full(len(da), np.nan); hi_B = np.full(len(da), np.nan)

    for i, b in enumerate(brands, 1):
        keep = brand_tr != b                       # June rows that remain
        test_idx = np.where(brand_te == b)[0]      # December rows of this brand

        mu_global, te_ms = _te_tables(brand_tr[keep], dh["seri"].values[keep],
                                      price_tr[keep])
        lvl3, lvl2, lvl1 = _segment_tables(kasa_tr[keep], ageb_tr[keep],
                                           ccb_tr[keep], price_tr[keep], mu_global)

        # training matrix with the re-estimated encoding
        Xb = X_tr[keep].copy()
        idx = pd.MultiIndex.from_arrays([brand_tr[keep], dh["seri"].values[keep]])
        Xb["TE_ms"] = te_ms.reindex(idx).values
        yb = y_tr_v[keep]

        # the held-out brand's December rows under the two fallback rules
        Xa = X_te.iloc[test_idx].copy()
        XA = Xa.copy(); XA["TE_ms"] = mu_global
        v = pd.Series(lvl3.reindex(pd.MultiIndex.from_arrays(
            [kasa_te[test_idx], ageb_te[test_idx], ccb_te[test_idx]])).values, dtype=float)
        v = v.fillna(pd.Series(lvl2.reindex(pd.MultiIndex.from_arrays(
            [kasa_te[test_idx], ageb_te[test_idx]])).values, dtype=float))
        v = v.fillna(pd.Series(lvl1.reindex(ageb_te[test_idx]).values, dtype=float))
        v = v.fillna(mu_global)
        XB = Xa.copy(); XB["TE_ms"] = v.values

        model = _cb().fit(Xb.values, yb)
        pred_A[test_idx] = model.predict(XA.values)
        pred_B[test_idx] = model.predict(XB.values)

        qb = {}
        for a in INTERVAL:
            mq = _cb(loss=f"Quantile:alpha={a}").fit(Xb.values, yb)
            qb[a] = (mq.predict(XA.values), mq.predict(XB.values))
        lo_A[test_idx] = np.minimum(qb[INTERVAL[0]][0], qb[INTERVAL[1]][0])
        hi_A[test_idx] = np.maximum(qb[INTERVAL[0]][0], qb[INTERVAL[1]][0])
        lo_B[test_idx] = np.minimum(qb[INTERVAL[0]][1], qb[INTERVAL[1]][1])
        hi_B[test_idx] = np.maximum(qb[INTERVAL[0]][1], qb[INTERVAL[1]][1])

        print(f"  [{i:>2}/{len(brands)}] {b:<18} held out: "
              f"{keep.sum():,} train rows, {len(test_idx):,} test vehicles "
              f"({time.time()-t0:.0f}s)")

    m = sim_rows
    y = y_te_v[m]
    pooled = pd.DataFrame([
        {"Group": "All simulated cold-start vehicles", "Encoding": "Known brand-series (warm start)",
         "N": int(m.sum()), **_metrics(y, ref_pred[m]), **_coverage(y, ref_lo[m], ref_hi[m])},
        {"Group": "All simulated cold-start vehicles", "Encoding": "Global mean (current rule)",
         "N": int(m.sum()), **_metrics(y, pred_A[m]), **_coverage(y, lo_A[m], hi_A[m])},
        {"Group": "All simulated cold-start vehicles", "Encoding": "Segment cell (Section 7.3)",
         "N": int(m.sum()), **_metrics(y, pred_B[m]), **_coverage(y, lo_B[m], hi_B[m])},
    ]).round(4)

    # per brand: does the richer fallback help, and how far is the brand from the market mean?
    mu_market = float(price_tr.mean())
    brand_mean = pd.Series(price_tr).groupby(brand_tr).mean()
    per = []
    for b in sorted(set(brand_te[m])):
        s = (brand_te == b)
        if s.sum() < PER_BRAND_MIN_N:
            continue
        ys = y_te_v[s]
        mk = lambda p: mean_absolute_percentage_error(ys, p[s]) * 100
        cov = lambda lo, hi: float(((ys >= lo[s]) & (ys <= hi[s])).mean() * 100)
        per.append({
            "Brand": b, "N (December)": int(s.sum()),
            "Brand mean price / market mean": brand_mean[b] / mu_market,
            "MAPE, warm start (%)": mk(ref_pred),
            "MAPE, global mean (%)": mk(pred_A),
            "MAPE, segment cell (%)": mk(pred_B),
            "Gain from segment cell (pp)": mk(pred_A) - mk(pred_B),
            "Coverage, global mean (%)": cov(lo_A, hi_A),
            "Coverage, segment cell (%)": cov(lo_B, hi_B),
        })
    by_brand = pd.DataFrame(per).sort_values("Brand mean price / market mean").round(3)

    far = by_brand[(by_brand["Brand mean price / market mean"] > 1.5) |
                   (by_brand["Brand mean price / market mean"] < 1 / 1.5)]
    near = by_brand.drop(far.index)
    r = float(np.corrcoef(by_brand["Gain from segment cell (pp)"],
                          np.abs(np.log(by_brand["Brand mean price / market mean"])))[0, 1])
    summary = pd.DataFrame([
        {"Statistic": "Brands with mean price far from the market mean (ratio >1.5 or <0.67)",
         "Value": f"{len(far)} brands; segment cell better in {(far['Gain from segment cell (pp)'] > 0).sum()}; "
                  f"mean gain {far['Gain from segment cell (pp)'].mean():.1f} pp of MAPE"},
        {"Statistic": "Brands with mean price close to the market mean",
         "Value": f"{len(near)} brands; segment cell better in {(near['Gain from segment cell (pp)'] > 0).sum()}; "
                  f"mean gain {near['Gain from segment cell (pp)'].mean():.1f} pp of MAPE"},
        {"Statistic": "Correlation of the gain with |log(brand mean / market mean)|",
         "Value": f"r = {r:.2f} over {len(by_brand)} brands"},
    ])

    os.makedirs(CFG["output_dir"], exist_ok=True)
    sheets = [
        ("S13 - Leave-brand-out",
         "Table S13: Leave-one-brand-out cold-start simulation (each brand is removed from training "
         "in turn; its December vehicles are priced by a model that never saw the brand)", pooled),
        ("S13b - By brand",
         f"Table S13b: Cold-start error by held-out brand (brands with at least {PER_BRAND_MIN_N} "
         "December vehicles), ordered by brand price level relative to the market mean", by_brand),
        ("S13c - Summary", "Table S13c: When does the segment-cell fallback help?", summary),
    ]
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        for sheet, title, df in sheets:
            pd.DataFrame([[title]]).to_excel(xw, sheet_name=sheet, index=False, header=False)
            df.to_excel(xw, sheet_name=sheet, index=False, startrow=2)

    pd.DataFrame({"brand": brand_te, "actual_USD": y_te_v, "simulated": sim_rows,
                  "pred_known": ref_pred, "lo_known": ref_lo, "hi_known": ref_hi,
                  "pred_global_mean": pred_A, "lo_global_mean": lo_A, "hi_global_mean": hi_A,
                  "pred_segment_cell": pred_B, "lo_segment_cell": lo_B, "hi_segment_cell": hi_B}).to_csv(
        os.path.join(CFG["output_dir"], "predictions_leave_brand_out.csv"), index=False)

    print("\n" + pooled.to_string(index=False))
    print("\n" + summary.to_string(index=False))
    print(f"\nWritten: {OUT_XLSX}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
