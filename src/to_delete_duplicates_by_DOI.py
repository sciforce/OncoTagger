from pathlib import Path
import re
import unicodedata

import pandas as pd


DOI_COLUMN = "DOI"
TITLE_COLUMN = "Article Title"
YEAR_COLUMN = "Publication Year"
EXCLUDED_PUBLICATION_YEARS = ("2026",)

HELPER_COLUMNS = ["_dedup_doi", "_dedup_title", "_dedup_year"]


def _clean_scalar(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "<na>"}:
        return ""
    return text


def normalize_doi(value) -> str:
    """Return a comparable DOI value, or an empty string when DOI is absent."""
    text = _clean_scalar(value)
    if not text:
        return ""

    text = re.sub(r"\s+", "", text)
    text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", text, flags=re.IGNORECASE)

    doi_match = re.search(r"(10\.\d{4,9}/\S+)", text, flags=re.IGNORECASE)
    if doi_match:
        text = doi_match.group(1)

    return text.strip(".,;:)]}").casefold()


def normalize_title(value) -> str:
    """Normalize titles conservatively for duplicate detection."""
    text = _clean_scalar(value)
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text.casefold())
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_publication_year(value) -> str:
    text = _clean_scalar(value)
    if not text:
        return ""

    match = re.search(r"\b((?:19|20)\d{2})\b", text)
    if match:
        return match.group(1)

    try:
        year = int(float(text))
    except ValueError:
        return ""

    if 1900 <= year <= 2099:
        return str(year)
    return ""


def _mark_title_year_duplicates(df: pd.DataFrame) -> pd.Series:
    """
    Mark duplicate title/year rows after DOI deduplication.

    The title/year fallback is deliberately conservative:
    - it requires both title and year;
    - it prefers keeping a record that has a DOI;
    - it does not collapse rows when the same title/year has conflicting DOIs.
    """
    duplicate_mask = pd.Series(False, index=df.index)
    has_title_year = df["_dedup_title"].ne("") & df["_dedup_year"].ne("")

    for _, group in df.loc[has_title_year].groupby(["_dedup_title", "_dedup_year"], sort=False):
        if len(group) <= 1:
            continue

        unique_dois = set(group.loc[group["_dedup_doi"].ne(""), "_dedup_doi"])
        if len(unique_dois) > 1:
            continue

        rows_with_doi = group.index[group["_dedup_doi"].ne("")]
        keep_index = rows_with_doi[0] if len(rows_with_doi) else group.index[0]
        duplicate_mask.loc[group.index.difference([keep_index])] = True

    return duplicate_mask


def deduplicate_records(
    df: pd.DataFrame,
    exclude_publication_years: tuple[str, ...] = EXCLUDED_PUBLICATION_YEARS,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if DOI_COLUMN not in df.columns:
        raise ValueError(f"Column '{DOI_COLUMN}' not found in input file.")

    work = df.copy()
    work["_dedup_doi"] = work[DOI_COLUMN].map(normalize_doi)

    if TITLE_COLUMN in work.columns and YEAR_COLUMN in work.columns:
        work["_dedup_title"] = work[TITLE_COLUMN].map(normalize_title)
        work["_dedup_year"] = work[YEAR_COLUMN].map(normalize_publication_year)
    else:
        work["_dedup_title"] = ""
        work["_dedup_year"] = ""

    has_doi = work["_dedup_doi"].ne("")
    doi_duplicate_mask = has_doi & work.duplicated(subset="_dedup_doi", keep="first")
    after_doi = work.loc[~doi_duplicate_mask].copy()

    title_year_duplicate_mask = _mark_title_year_duplicates(after_doi)
    after_dedup = after_doi.loc[~title_year_duplicate_mask].copy()

    excluded_years = {normalize_publication_year(year) for year in exclude_publication_years}
    excluded_years.discard("")
    year_exclusion_mask = after_dedup["_dedup_year"].isin(excluded_years)
    clean = after_dedup.loc[~year_exclusion_mask].copy()

    excluded_years_label = ", ".join(sorted(excluded_years)) if excluded_years else "none"
    stats = {
        "input_rows": len(df),
        "dropped_by_doi": int(doi_duplicate_mask.sum()),
        "dropped_by_title_year": int(title_year_duplicate_mask.sum()),
        "excluded_publication_years": excluded_years_label,
        "dropped_by_excluded_publication_year": int(year_exclusion_mask.sum()),
        "output_rows": len(clean),
        "rows_without_doi_in_output": int(clean["_dedup_doi"].eq("").sum()),
    }

    return clean.drop(columns=HELPER_COLUMNS), stats


def remove_duplicates_by_doi(input_file: str = "data/raw/combined_dataset.xlsx",
                             output_file: str = "data/processed/processed_dataset.xlsx"):
    """
    Reads combined_dataset.xlsx, removes duplicates, and writes processed_dataset.xlsx.

    DOI deduplication ignores empty DOI values, so different no-DOI articles are
    not collapsed into one row. A conservative title/year fallback catches
    duplicates among no-DOI records or DOI/no-DOI copies of the same article.
    Records from excluded publication years default to 2026 and are reported
    separately in the terminal summary.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_excel(input_path)
    df_clean, stats = deduplicate_records(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_excel(output_path, index=False)

    print(
        "Duplicates removed. "
        f"Input: {stats['input_rows']:,}; "
        f"dropped by DOI: {stats['dropped_by_doi']:,}; "
        f"dropped by title/year: {stats['dropped_by_title_year']:,}; "
        f"dropped by publication year {stats['excluded_publication_years']}: "
        f"{stats['dropped_by_excluded_publication_year']:,}; "
        f"output: {stats['output_rows']:,}; "
        f"rows without DOI kept: {stats['rows_without_doi_in_output']:,}. "
        f"Clean file written to: {output_path}"
    )


if __name__ == "__main__":
    remove_duplicates_by_doi()
