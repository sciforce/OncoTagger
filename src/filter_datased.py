import os
import re
import pandas as pd
from collections import OrderedDict

def uniq_lower(lst):
    return list(OrderedDict.fromkeys(w.lower().strip() for w in lst if isinstance(w, str)))

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
    
    # 1. Завантаження словників
    excl_tsv = os.path.join(sources_dir, "wos_exclusion_categories.tsv")
    df_excl = pd.read_csv(excl_tsv, sep='\t', header=None, dtype=str)
    exclusion_categories = set(df_excl.iloc[:,0].dropna().str.lower().str.strip().tolist())

    onco_csv = os.path.join(sources_dir, "onco_terms_filter.csv")
    df_onco = pd.read_csv(onco_csv, dtype=str)
    onc_pat = compile_pattern(uniq_lower(df_onco.iloc[:,0].tolist()))

    ai_csv = os.path.join(sources_dir, "raw_ai_terms_filter.csv")
    df_ai = pd.read_csv(ai_csv, dtype=str)
    ai_pat = compile_pattern(uniq_lower(df_ai.iloc[:,0].tolist()))

    # 2. Зчитування датасету
    df = pd.read_excel(input_file, dtype=str).fillna("")
    initial_count = len(df)
    
    # 3. Фільтрація року
    df["Publication Year"] = df["Publication Year"].astype(str).str.strip()
    mask_2026 = df["Publication Year"].str.contains("2026")
    dropped_year = mask_2026.sum()
    df = df[~mask_2026]

    # 4. Фільтрація WoS категорій
    wos_cat_normalized = df["WoS Categories"].str.lower().str.strip()
    mask_wos_excluded = wos_cat_normalized.isin(exclusion_categories)
    dropped_wos = mask_wos_excluded.sum()
    df = df[~mask_wos_excluded]

    # 5. Підготовка тексту для пошуку
    cols = ["Authors", "Article Title", "Source Title", "Author Keywords", "Abstract"]
    df["__combined_text"] = df[cols].agg(" ".join, axis=1).str.lower()
    
    # 6. Одночасна перевірка ключових слів
    mask_cancer = df["__combined_text"].str.contains(onc_pat)
    mask_ai = df["__combined_text"].str.contains(ai_pat)
    
    # Строга умова: обов'язкова наявність і тих, і інших термінів одночасно
    mask_both = mask_cancer & mask_ai
    
    # Підрахунок статистики відхилень для логів
    dropped_cancer_only = (mask_cancer & ~mask_ai).sum()
    dropped_ai_only = (~mask_cancer & mask_ai).sum()
    dropped_none = (~mask_cancer & ~mask_ai).sum()
    dropped_keywords_total = (~mask_both).sum()

    # Застосування фільтру та видалення тимчасової колонки
    df = df[mask_both].drop(columns="__combined_text")

    # 7. Логування результатів
    print("=" * 50)
    print(f"Initial dataset size: {initial_count} articles")
    print("-" * 50)
    print(f"Deleted {dropped_year} articles with Publication year = 2026")
    print(f"Deleted {dropped_wos} articles with excluded WoS categories")
    print(f"Deleted {dropped_keywords_total} articles missing required keywords, including:")
    print(f"  - Missing AI terms (but has Cancer terms): {dropped_cancer_only}")
    print(f"  - Missing Cancer terms (but has AI terms): {dropped_ai_only}")
    print(f"  - Missing both Cancer and AI terms: {dropped_none}")
    print("-" * 50)
    print(f"Final dataset size: {len(df)} articles")
    print(f"Total deleted: {initial_count - len(df)} articles")
    print("=" * 50)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    print(f"\n[OK] Filtered data saved to -> {output_file}")

if __name__ == "__main__":
    filter_dataset()