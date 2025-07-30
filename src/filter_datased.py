import os
import re
import pandas as pd
from collections import OrderedDict

def uniq_lower(lst):
    return list(OrderedDict.fromkeys(w.lower() for w in lst if isinstance(w, str)))

def compile_pattern(keywords):
    esc = sorted((re.escape(w) for w in keywords), key=len, reverse=True)
    return re.compile(r'\b(?:' + '|'.join(esc) + r')\b', re.IGNORECASE)

def filter_dataset(
    input_file: str = "data/processed/processed_dataset.xlsx",
    output_file: str = "data/filtered/filtered_dataset.xlsx"):
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    base = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.normpath(os.path.join(base, os.pardir, "sources"))
    
    excl_csv = os.path.join(sources_dir, "wos_exclusion_categories.csv")
    df_excl = pd.read_csv(excl_csv, dtype=str)
    exclusion_categories = df_excl.iloc[:,0].dropna().tolist()

    onco_csv = os.path.join(sources_dir, "onco_terms_filter.csv")
    df_onco = pd.read_csv(onco_csv, dtype=str)
    onco_terms = uniq_lower(df_onco.iloc[:,0].tolist())
    onc_pat = compile_pattern(onco_terms)

    ai_csv = os.path.join(sources_dir, "raw_ai_terms_filter.csv")
    df_ai = pd.read_csv(ai_csv, dtype=str)
    ai_terms = uniq_lower(df_ai.iloc[:,0].tolist())
    ai_pat = compile_pattern(ai_terms)

    df = pd.read_excel(input_file, dtype=str).fillna("")

    df = df[~df["WoS Categories"].isin(exclusion_categories)]

    cols = ["Authors", "Article Title", "Source Title", "Author Keywords", "Abstract"]
    df["__combined_text"] = df[cols].agg(" ".join, axis=1).str.lower()

    mask = (
        df["__combined_text"].str.contains(onc_pat) &
        df["__combined_text"].str.contains(ai_pat)
    )
    result = df.loc[mask].drop(columns="__combined_text")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_excel(output_file, index=False)
    print(f"Filtered {len(result)} articles → {output_file}")

if __name__ == "__main__":
    filter_dataset()