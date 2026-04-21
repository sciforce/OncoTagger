from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_WORKBOOK = PROJECT_ROOT / "data" / "results" / "filtered_dataset_binary_classification_analysis.xlsx"
POPULATION_CSV = PROJECT_ROOT / "sources" / "total-population-by-country-2025 (1) (1).csv"
OUTPUT_XLSX = PROJECT_ROOT / "data" / "results" / "article to population ratio.xlsx"
OUTPUT_JSON = PROJECT_ROOT / "documentation" / "self-documented docs" / "article_to_population_ratio_summary.json"
OUTPUT_TXT = PROJECT_ROOT / "documentation" / "self-documented docs" / "article_to_population_ratio_summary.txt"


COUNTRY_NAME_OVERRIDES = {
    "Korea, Republic of": "South Korea",
    "Taiwan, Province of China": "Taiwan",
    "Iran, Islamic Republic of": "Iran",
    "Tanzania, United Republic of": "Tanzania",
    "Dominican Rep": "Dominican Republic",
    "Bosnia & Herceg": "Bosnia and Herzegovina",
    "Viet Nam": "Vietnam",
    "Czechia": "Czech Republic",
    "Syrian Arab Republic": "Syria",
}


def load_country_counts() -> pd.DataFrame:
    df = pd.read_excel(ANALYSIS_WORKBOOK, sheet_name="Reprint Country Overall")
    df = df.rename(columns={"Reprint-address country": "Country", "Count": "Articles count", "Share_of_all_articles": "Share_of_all_articles"})
    df = df[~df["Country"].astype(str).str.contains("country unavailable", case=False, na=False)].copy()
    return df


def load_population() -> pd.DataFrame:
    df = pd.read_csv(POPULATION_CSV)
    return df[["country", "pop2025"]].rename(columns={"country": "Population country match", "pop2025": "Population in 2025"})


def build_ratio_table() -> pd.DataFrame:
    counts = load_country_counts()
    population = load_population()

    counts["Population country match"] = counts["Country"].replace(COUNTRY_NAME_OVERRIDES)
    merged = counts.merge(population, on="Population country match", how="left")

    merged["Articles per 1M of population"] = merged["Articles count"] / (merged["Population in 2025"] / 1_000_000)
    merged["Share of full corpus (%)"] = merged["Share_of_all_articles"] * 100

    merged = merged.sort_values(["Articles per 1M of population", "Articles count"], ascending=[False, False]).reset_index(drop=True)
    merged["Per-capita rank"] = merged.index + 1
    return merged


def build_summary(df: pd.DataFrame) -> dict:
    threshold_100 = df[df["Articles count"] >= 100].copy()
    threshold_20 = df[df["Articles count"] >= 20].copy()

    def top_rows(frame: pd.DataFrame, n: int = 10) -> list[dict]:
        cols = ["Country", "Articles count", "Population in 2025", "Articles per 1M of population"]
        out = []
        for _, row in frame.head(n)[cols].iterrows():
            out.append(
                {
                    "Country": row["Country"],
                    "Articles count": int(row["Articles count"]),
                    "Population in 2025": int(row["Population in 2025"]),
                    "Articles per 1M of population": round(float(row["Articles per 1M of population"]), 4),
                }
            )
        return out

    return {
        "input_countries": int(len(df)),
        "matched_population_rows": int(df["Population in 2025"].notna().sum()),
        "missing_population_rows": int(df["Population in 2025"].isna().sum()),
        "top10_per_capita_all": top_rows(df, 10),
        "top10_per_capita_min20_articles": top_rows(threshold_20, 10),
        "top10_per_capita_min100_articles": top_rows(threshold_100, 10),
    }


def write_outputs(df: pd.DataFrame, summary: dict) -> None:
    missing = df[df["Population in 2025"].isna()].copy()
    min20 = df[df["Articles count"] >= 20].copy()
    min100 = df[df["Articles count"] >= 100].copy()

    ordered_cols = [
        "Per-capita rank",
        "Country",
        "Population country match",
        "Articles per 1M of population",
        "Articles count",
        "Population in 2025",
        "Share of full corpus (%)",
    ]

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df[ordered_cols].to_excel(writer, sheet_name="All countries", index=False)
        min20[ordered_cols].to_excel(writer, sheet_name="PerCapita N20", index=False)
        min100[ordered_cols].to_excel(writer, sheet_name="PerCapita N100", index=False)
        missing[ordered_cols].to_excel(writer, sheet_name="Missing population", index=False)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "Article to Population Ratio Summary",
        "=================================",
        "",
        f"Input countries: {summary['input_countries']}",
        f"Matched population rows: {summary['matched_population_rows']}",
        f"Missing population rows: {summary['missing_population_rows']}",
        "",
        "Top 10 per-capita countries, all matched rows:",
    ]
    for row in summary["top10_per_capita_all"]:
        lines.append(
            f"- {row['Country']}: {row['Articles per 1M of population']:.4f} per 1M "
            f"({row['Articles count']} articles; population {row['Population in 2025']})"
        )
    lines.extend(["", "Top 10 per-capita countries, threshold >=20 articles:"])
    for row in summary["top10_per_capita_min20_articles"]:
        lines.append(
            f"- {row['Country']}: {row['Articles per 1M of population']:.4f} per 1M "
            f"({row['Articles count']} articles; population {row['Population in 2025']})"
        )
    lines.extend(["", "Top 10 per-capita countries, threshold >=100 articles:"])
    for row in summary["top10_per_capita_min100_articles"]:
        lines.append(
            f"- {row['Country']}: {row['Articles per 1M of population']:.4f} per 1M "
            f"({row['Articles count']} articles; population {row['Population in 2025']})"
        )

    OUTPUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = build_ratio_table()
    summary = build_summary(df)
    write_outputs(df, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {OUTPUT_XLSX}")
    print(f"[OK] Wrote {OUTPUT_JSON}")
    print(f"[OK] Wrote {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
