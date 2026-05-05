import os
import glob

DATA_DIR = r"D:\Tick_data"
OUTPUT_FILE = os.path.join(DATA_DIR, "BTCUSDT-aggTrades-2024-full.csv")

files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT-aggTrades-2024-*.csv")))
print(f"Found {len(files)} CSV files.")

with open(OUTPUT_FILE, "wb") as out:
    for file in files:
        print(f"  Merging {os.path.basename(file)}...")
        with open(file, "rb") as f:
            while chunk := f.read(256 * 1024 * 1024):  # 256MB chunks
                out.write(chunk)

print(f"Done → {OUTPUT_FILE}")