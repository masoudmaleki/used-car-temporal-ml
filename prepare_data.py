"""prepare_data.py - builds the two analysis files from the raw sources.

Inputs (place them in this folder)
----------------------------------
car_price_prediction.csv   June 2025 listings, downloaded unchanged from the Kaggle
                           dataset cited in the README (UTF-8, comma-separated).
Aralık_2025.xlsx           December 2025 listings, collected by the authors.

Outputs
-------
Haziran_2025_son.csv   34,509 cleaned June listings    (training period)
Aralık_2025_son.csv    37,062 cleaned December listings (test period)

Cleaning, as described in Section 4.2 of the paper
--------------------------------------------------
1. rows with a missing brand, series, model designation or price are dropped;
2. model year 1980 or later;
3. mileage at most 600,000 km;
4. at most eight painted and at most eight replaced parts;
5. prices outside the 2nd-99th percentiles are dropped;
6. engine power and engine displacement outside the 1st-99th percentiles are dropped
   (applied in this order, each on the rows that survive the previous step).

The brand "MINI" in the June file and "Mini" in the December file denote the same
brand and are harmonised to "Mini".

Do not open the Kaggle file in a spreadsheet program and save it again
-----------------------------------------------------------------------
A spreadsheet round trip converts values such as the engine size "1.8" in the model
designation, or the Saab series "9-3", into dates ("1.Ağu", "9.Mar"), changes the
delimiter, and can write some characters in a legacy encoding. The script reads the
original file and warns if it finds date-like series or model values.

Usage
-----
    python prepare_data.py
"""
import re
import sys

import pandas as pd

JUNE_RAW, DEC_RAW = "car_price_prediction.csv", "Aralık_2025.xlsx"
JUNE_OUT, DEC_OUT = "Haziran_2025_son.csv", "Aralık_2025_son.csv"

KEY_COLUMNS = ["marka", "seri", "model", "fiyat"]
# values printed in Table 3 of the paper; the script checks itself against them
EXPECTED = {
    "June": dict(raw=50755, clean=34509, brands=45, combos=362, mean_price=853859),
    "December": dict(raw=52051, clean=37062, brands=49, combos=349, mean_price=1010748),
}
DATE_LIKE = re.compile(r"^\d{1,2}\.(Oca|Şub|Mar|Nis|May|Haz|Tem|Ağu|Eyl|Eki|Kas|Ara)$")


def _is_stray(s):
    return isinstance(s, str) and any("\udc80" <= ch <= "\udcff" for ch in s)


def _repair_stray_bytes(s):
    """Decode single legacy (ISO-8859-9) bytes that survive a UTF-8 read."""
    if not _is_stray(s):
        return s
    return "".join(bytes([ord(ch) - 0xDC00]).decode("iso-8859-9")
                   if "\udc80" <= ch <= "\udcff" else ch for ch in s)


def read_june(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode("utf-8-sig", errors="replace")
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig",
                     encoding_errors="surrogateescape")
    stray = 0
    for col in df.columns:
        if df[col].dtype == object:
            stray += int(df[col].map(_is_stray).sum())
            df[col] = df[col].map(_repair_stray_bytes)
    dates = int(sum(df[c].astype(str).str.match(DATE_LIKE).sum() for c in ("seri", "model")))
    if sep != "," or stray or dates:
        print(f"  WARNING: this does not look like the original Kaggle file "
              f"(delimiter '{sep}', {stray} values in a legacy encoding, "
              f"{dates} date-like series/model values). Download it again.")
    return df


def clean(df):
    df = df.dropna(subset=KEY_COLUMNS)
    df = df[df["yil"] >= 1980]
    df = df[df["kilometre"] <= 600000]
    df = df[df["boyali_sayisi"] <= 8]
    df = df[df["degisen_sayisi"] <= 8]
    lo, hi = df["fiyat"].quantile(0.02), df["fiyat"].quantile(0.99)
    df = df[(df["fiyat"] >= lo) & (df["fiyat"] <= hi)]
    for col in ("motor_gucu", "motor_hacmi"):
        df = df[df[col].between(df[col].quantile(0.01), df[col].quantile(0.99))]
    return df


def report(label, raw_n, df):
    exp = EXPECTED[label]
    got = dict(raw=raw_n, clean=len(df), brands=df["marka"].nunique(),
               combos=df[["marka", "seri"]].drop_duplicates().shape[0],
               mean_price=int(round(df["fiyat"].mean())))
    print(f"  {label}: raw {got['raw']:,} -> clean {got['clean']:,}; "
          f"{got['brands']} brands, {got['combos']} brand-series combinations, "
          f"mean price {got['mean_price']:,} TL")
    bad = {k: (got[k], exp[k]) for k in exp if got[k] != exp[k]}
    if bad:
        print(f"  WARNING: {label} differs from Table 3 of the paper: "
              + "; ".join(f"{k} {g} instead of {e}" for k, (g, e) in bad.items()))
    return not bad


def main():
    print("Preparing the June 2025 training file ...")
    june_raw = read_june(JUNE_RAW)
    june = clean(june_raw)
    n_mini = int((june["marka"] == "MINI").sum())
    june.loc[june["marka"] == "MINI", "marka"] = "Mini"
    print(f"  brand 'MINI' harmonised to 'Mini' in {n_mini} rows")
    ok_june = report("June", len(june_raw), june)
    june.to_csv(JUNE_OUT, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("\nPreparing the December 2025 test file ...")
    dec_raw = pd.read_excel(DEC_RAW)
    dec = clean(dec_raw)
    ok_dec = report("December", len(dec_raw), dec)
    dec.to_csv(DEC_OUT, index=False, encoding="utf-8-sig", lineterminator="\n")

    print(f"\nWritten {JUNE_OUT} and {DEC_OUT}.")
    if not (ok_june and ok_dec):
        print("At least one check failed: the source files are not the ones used in the paper.")
        sys.exit(1)
    print("All checks against Table 3 passed.")


if __name__ == "__main__":
    main()