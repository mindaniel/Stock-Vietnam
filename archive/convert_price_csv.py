"""One-time script: convert all data/*.csv price files to parquet, then delete CSVs."""
import os
import glob
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files in {DATA_DIR}")

ok, err = 0, 0
for f in csv_files:
    pq = f[:-4] + ".parquet"
    try:
        df = pd.read_csv(f)
        df.to_parquet(pq, index=False, engine="pyarrow")
        os.remove(f)
        ok += 1
        if ok % 200 == 0:
            print(f"  ... {ok}/{len(csv_files)} converted")
    except Exception as e:
        print(f"  ERR {os.path.basename(f)}: {e}")
        err += 1

print(f"Done: {ok} converted, {err} errors")
remaining = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Remaining CSVs in data/: {len(remaining)}")
parquets = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
print(f"Parquet files in data/: {len(parquets)}")
