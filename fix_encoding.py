"""
fix_encoding.py - repairs the June 2025 file before running reproduce_all.py

Problem found in the audit
--------------------------
The June file (Kaggle source) stores Turkish characters double-encoded
("DÃ¼z", "TofaÅŸ", "YarÄ± Otomatik"), whereas the December scrape is
proper UTF-8 ("Düz", "Tofaş", "Yarı Otomatik"). In addition, the June file spells
the brand "MINI" and the December file "Mini". Consequences in the published
pipeline:
  * every manual / semi-automatic car in the December test set was encoded as
    automatic (both transmission dummies = 0 for all 37,062 test rows);
  * Tofaş (690 test rows) and Mini (256 test rows) were treated as brands never
    seen in training and received the global-mean target encoding.

What this script does
---------------------
1. Reverses the double encoding in every text column of the June file
   (latin-1 -> utf-8 round trip; values that are already correct are left alone).
2. Harmonises the brand name "MINI" -> "Mini".
3. Verifies that the category sets of both files now match.
The original file is kept as Haziran_2025_son_ORIGINAL.csv.

Run once, in the folder that contains the two CSV files:
    python fix_encoding.py
"""
import os
import shutil

import pandas as pd

JUNE, DEC = "Haziran_2025_son.csv", "Aralık_2025_son.csv"
BACKUP = "Haziran_2025_son_ORIGINAL.csv"


def repair(s):
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def main():
    src = BACKUP if os.path.exists(BACKUP) else JUNE
    dh = pd.read_csv(src, encoding="utf-8-sig")
    da = pd.read_csv(DEC, encoding="utf-8-sig")

    text_cols = [c for c in dh.columns if not pd.api.types.is_numeric_dtype(dh[c])]
    for c in text_cols:
        before = dh[c].copy()
        dh[c] = dh[c].map(repair)
        print(f"  {c:12s}: {int((before != dh[c]).sum()):6d} values repaired")
    n_mini = int((dh["marka"] == "MINI").sum())
    dh.loc[dh["marka"] == "MINI", "marka"] = "Mini"
    print(f"  marka       : {n_mini:6d} 'MINI' -> 'Mini'")

    for c in ["vites_tipi", "yakit_tipi", "kasa_tipi", "kimden"]:
        only_june = set(dh[c]) - set(da[c])
        only_dec = set(da[c]) - set(dh[c])
        print(f"  check {c:11s} only-June={sorted(only_june)} only-Dec={sorted(only_dec)}")
    if src == JUNE:
        shutil.copyfile(JUNE, BACKUP)
    dh.to_csv(JUNE, index=False, encoding="utf-8-sig")
    print(f"\nWritten {JUNE} (original kept as {BACKUP}).")


if __name__ == "__main__":
    main()
