# Exchange Rate Pass-Through in Used Vehicle Pricing — reproduction package

Code and data for:

> Melek, N., Melek, M., Güvercin, M. *Exchange Rate Pass-Through in Used Vehicle Pricing: A Temporal Machine Learning Framework with Repeated Cross-Sectional Evidence from Turkey.* Applied Soft Computing (under review).

The repository reproduces every table and figure of the paper and of its supplementary material from the two source datasets. Models are trained on June 2025 listings and evaluated on December 2025 listings; prices are deflated by an external reference (USD/TRY, EUR/TRY, CPI, raw TL, gold) before modelling.

---

## 1. Requirements

Python 3.13.3 was used for all reported results. The package versions are those recorded in Supplementary Table S12:

```
numpy==2.3.2
pandas==2.3.1
scikit-learn==1.7.1
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.8
shap==0.50.0
matplotlib
openpyxl
```

```bash
pip install -r requirements.txt
```

Results are deterministic for a fixed package set: all models are fitted with fixed seeds (42, 123, 789) and all bootstrap procedures with seed 20260713. The reported results were produced on Windows 11 (Supplementary Table S12). With the package versions above, CatBoost, LightGBM, Random Forest, the MLP and the linear models give the same results on Linux as on Windows. XGBoost does not: its random row and column subsampling does not draw the same samples on the two platforms for the same seed, so on Linux it fits a different, equally valid model. Its Test R² stays at 0.961, but its RMSE differs by about 15 USD, and in the paired bootstrap of Supplementary Table S1b its RMSE gap to CatBoost becomes +46 USD (95% CI [2, 93], p = 0.04) instead of +32 USD (95% CI [−8, 74], p = 0.13). Differences of this size are within the variation of XGBoost across its three seeds (Table 5).

## 2. Data

Two raw source files are the input of the whole pipeline; everything else is generated.

| File | Source | In this repository |
|---|---|---|
| `car_price_prediction.csv` | public Kaggle dataset (see below), 50,755 listings | no — download it from Kaggle |
| `Aralık_2025.xlsx` | collected by the authors from arabam.com in December 2025, 52,051 listings | yes |

June 2025 source dataset:

> Tanrıverdi, M. (2025). *Used Car Prices Dataset Turkey (arabam.com).* Kaggle. https://www.kaggle.com/datasets/mehmettanriverdi/used-car-prices-dataset-turkey-arabam-com (accessed 10 April 2026).

Download the June file from the Kaggle page, keep its name `car_price_prediction.csv`, place it in this folder, and check that it is the file used in the paper:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `car_price_prediction.csv` | 6,088,499 | `78b8d8e1b6efb209b333b86e25b705da9824a6f123d0598f976f7c1da4cccc0b` |
| `Aralık_2025.xlsx` | 4,155,327 | `99aa8018807a72082704047a72d93aca202c4e30126c62f1dec8e01d1ad8f77b` |

(Windows: `certutil -hashfile car_price_prediction.csv SHA256`; Linux/macOS: `sha256sum car_price_prediction.csv`.) Do not open the file in a spreadsheet program and save it again: that converts values such as the engine size `1.8` in the model designation or the Saab series `9-3` into dates. `prepare_data.py` warns if it detects such values.

`prepare_data.py` turns the two raw files into the two analysis files used by every other script:

* `Haziran_2025_son.csv` — 34,509 June listings (training period)
* `Aralık_2025_son.csv` — 37,062 December listings (test period)

Cleaning follows Section 4.2 of the paper: records with missing brand, series, model designation or price are dropped; model year 1980 or later; mileage at most 600,000 km; at most eight painted and at most eight replaced parts; prices outside the 2nd–99th percentiles and engine power and displacement outside the 1st–99th percentiles are removed, each filter applied to the rows that survive the previous one. The script prints the resulting record counts, brand counts and mean prices and compares them with Table 3 of the paper, so a wrong input file is detected immediately.

Both analysis files contain vehicle attributes (model year, mileage, engine power, engine displacement, body type, transmission, fuel type, colour, brand, series, model designation, numbers of painted and replaced parts), a four-level seller-category label and the listed price. Listing identifiers, URLs, seller names and contact details and geographic fields are not used.

### Encoding and the earlier preprocessing

The Kaggle file is UTF-8 and comma-separated, and `prepare_data.py` reads it as it is. Our first analysis used a copy that had been opened and saved again in a spreadsheet program. That round trip changed the delimiter, converted 1,815 model designations and 12 series names into dates (for example the Saab series `9-3` became `9.Mar`), and wrote the month abbreviations containing ğ or ş as single ISO-8859-9 bytes, so the copy no longer decoded as UTF-8. Our preprocessing then read the copy as ISO-8859-9, which double-encoded every other Turkish string (`Düz` became `DÃ¼z`, `Tofaş` became `TofaÅŸ`), so that identical categories no longer matched between the June and December files. Reading the original file removes all of these effects. The brands `MINI` (June) and `Mini` (December) denote the same brand and are harmonised to `Mini`.

`fix_encoding.py` repairs a June file produced by the earlier preprocessing and is kept for provenance; it is not needed if you run `prepare_data.py`. The analysis scripts refuse to run on unharmonised input: they raise an exception if a category present in either file is absent from the declared category list, or if two brand spellings differ only in case or encoding.

## 3. How to run

Place the scripts and the two raw data files in the same folder and run, in this order:

```bash
python prepare_data.py             #  <1 min   raw files -> the two analysis files
python reproduce_all.py            # ~40 min   Tables 1-12, Section 6.5, Figures 2, 6, 7a, 7b, 8, 9, 10
python make_figures_3_4_5.py       #  <1 min   Figures 3, 4, 5a, 5b
python rev4.py                     #  ~2 min   Supplementary Tables S1a-S5
python reviewer4_round3.py         # ~15 min   Supplementary Tables S6-S12 and Table 13
python leave_brand_out.py          # ~15 min   Supplementary Tables S13, S13b
```

All outputs are written to `results/`. The four result workbooks of the reported run are included in `results/` of this repository, so every value in the paper can be checked without re-running anything. `reproduce_all.py` also accepts a smoke-test mode (`REPRO_FAST=1`) with reduced budgets; those values are not the reported ones.

### Expected output

If the environment is correct, the printed values include:

| Check | Value |
|---|---|
| CatBoost, USD/TRY, Test R² (Table 4) | 0.9618 |
| CatBoost, USD/TRY, Test MAPE | 10.74 % |
| Cross-validated R², all deflators (Table 4) | 0.9662 ± 0.0037 |
| June / December records after cleaning (Table 3) | 34,509 / 37,062 |
| MLP after tuning, Test R² (Table 6) | 0.9362 |
| Random Forest after tuning, Test R² (Table 6) | 0.9575 |
| Huber, δ = 1×/3×/10× train MAE, Test R² (Table S6) | 0.8814 / 0.9470 / 0.9594 |
| LightGBM training time (Table 13) | 0.4 s on the machine of Table S12 |
| Leave-one-brand-out, global-mean fallback, R² (Table S13) | 0.4977 |

Timings (Table 13) are re-measured on every run and depend on the machine and its load; only the model sizes and the R² column of Table 13 are reproducible exactly.

## 4. What each script produces

| Script | Output |
|---|---|
| `prepare_data.py` | `Haziran_2025_son.csv`, `Aralık_2025_son.csv` (the two analysis files) |
| `fix_encoding.py` | kept for provenance: repairs a June file produced by the earlier preprocessing run |
| `reproduce_all.py` | `results/reproduction_all_tables.xlsx` (Tables 1–12 and the Section 6.5 robustness check), `results/predictions_models.csv`, `results/predictions_normalization.csv`, `results/shap_train.csv`, `results/shap_test.csv`, `results/catboost_feature_importance.csv`, Figures 2, 6, 7a, 7b, 8, 9, 10 |
| `make_figures_3_4_5.py` | Figures 3, 4, 5a, 5b, drawn from `reproduction_all_tables.xlsx` |
| `rev4.py` | `results/reviewer4_supplementary.xlsx` (Tables S1a, S1b, S2, S3, S4, S5) |
| `reviewer4_round3.py` | `results/reviewer4_round3.xlsx` (Tables S6–S12 and Table 13) |
| `leave_brand_out.py` | `results/reviewer4_round4.xlsx` (Tables S13, S13b), `results/predictions_leave_brand_out.csv` |

Every value reported in the paper can be traced to a cell of one of these workbooks.

## 5. Method notes

* **Target encoding.** Brand–series prices are encoded with a smoothed mean, k = 50, estimated on the June data only. Unseen combinations fall back to the brand mean and, for an unseen brand, to the global training mean.
* **Temporal protocol.** December prices are deflated with the end-of-November reference value, so no information from the test period enters the pipeline.
* **Valuation range.** Quantile regression at α = 0.05 and α = 0.95, calibrated by conformalized quantile regression on a held-out June fold.
* **Leave-one-brand-out simulation.** Each brand present in June is removed from training in turn; the encoder, the reference model and the two quantile models are refitted without it, and that brand's December vehicles are then priced as if the brand had never been seen. This yields 37,055 cold-start cases instead of the seven genuinely new-brand vehicles in the test period.

## 6. Licence and citation

Code: MIT licence (see `LICENSE`).
December 2025 data (`Aralık_2025.xlsx`): Creative Commons Attribution 4.0 (CC BY 4.0).
June 2025 data: not redistributed here; see the Kaggle dataset above and its licence terms.

If you use this material, please cite the paper and, for the June data, the Kaggle dataset.
