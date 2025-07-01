import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path.cwd() / "data/"

def concat_data(param_name: str, sdate: str, edate: str) -> pd.DataFrame:
    collected = pd.DataFrame()

    sdate_dt = datetime.strptime(sdate, "%Y-%m-%d")
    edate_dt = datetime.strptime(edate, "%Y-%m-%d")

    start_year = sdate_dt.year
    end_year = edate_dt.year

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            f = DATA_DIR / f"sooList_{year}{month:02d}.json"
            try:
                if f.is_file():
                    df = pd.read_json(f)
                    if param_name in df.columns:
                        collected = pd.concat([collected, df[[param_name]]], ignore_index=True)
            except Exception as e:
                print(f"Error reading {f.name}: {e}")

    return collected