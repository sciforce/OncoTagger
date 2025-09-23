from pathlib import Path
import sys
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_PATH = RAW_DIR / "combined_dataset.xlsx"

COLUMNS_TO_KEEP = [
    "Authors", "Article Title", "Source Title", "Author Keywords",
    "Keywords Plus", "Abstract", "Publication Year", "Reprint Addresses",
    "DOI", "DOI Link", "Book DOI", "WoS Categories"
]

def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".xlsx":
            # requires openpyxl
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            sys.exit("openpyxl is not installed. Run: pip install openpyxl")
        return pd.read_excel(path, engine="openpyxl")
    if suf == ".xls":
        # requires xlrd
        try:
            import xlrd  # noqa: F401
        except ImportError:
            sys.exit("xlrd is not installed. Run: pip install xlrd==2.0.1")
        return pd.read_excel(path, engine="xlrd")
    raise ValueError(f"Unsupported extension: {path.name}")

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in RAW_DIR.iterdir()
                    if p.is_file()
                    and p.name.lower().startswith("savedrecs")
                    and p.suffix.lower() in (".xlsx", ".xls", ".csv")])

    if not files:
        sys.exit(f"In {RAW_DIR} no files savedrecs*.xlsx|xls|csv found")

    combined = []
    for f in files:
        df = read_any(f)

        missing = [c for c in COLUMNS_TO_KEEP if c not in df.columns]
        for c in missing:
            df[c] = pd.NA

        df = df[[c for c in COLUMNS_TO_KEEP]] 
        combined.append(df)

        print(f"[OK] {f.name}: {len(df):,} rows")

    out_df = pd.concat(combined, ignore_index=True)
    try:
        import openpyxl  
    except ImportError:
        sys.exit("openpyxl is not installed. Run: pip install openpyxl")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_excel(OUT_PATH, index=False, engine="openpyxl")
    print(f"\nSaved {len(out_df):,} rows → {OUT_PATH}")

if __name__ == "__main__":
    main()
