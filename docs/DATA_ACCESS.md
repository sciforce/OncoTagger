# Web of Science data access

## Public-release boundary

The study used bibliographic records exported from the Web of Science Core Collection (WoSCC). WoSCC exports are licensed content and are not redistributed in this repository. The restriction applies to raw exports and to row-level derivatives retaining article titles, abstracts, authors, keywords, addresses, DOIs, source titles, or WoS categories.

Open-access status of an underlying article does not by itself authorize redistribution of the corresponding WoSCC export record. Users are responsible for complying with their institutional agreement and Clarivate terms.

## Exact search query

The exact manuscript-locked query is stored verbatim in [`wos_search_query.txt`](wos_search_query.txt). It targets publication years 2019-2025, document type `Article`, English language, and open-access records.

## Export procedure

1. Sign in to Web of Science using an account with authorized WoSCC access.
2. Select **Web of Science Core Collection** rather than an all-databases search.
3. Run the query in [`wos_search_query.txt`](wos_search_query.txt).
4. Confirm the publication-year, document-type, language, and open-access limits shown by the query.
5. Export the results in batches supported by the current WoS interface.
6. Export a full record or a custom record containing all required fields listed below.
7. Save each batch as CSV, XLS, or XLSX with a filename beginning with `savedrecs`.
8. Place all batches in `data/raw/` and run `python src/combine_wos_exports.py`.

The WoS interface and export batch limits may change. Record the database edition, export date and time, result count, batch boundaries, and export format for every independent reproduction.

## Required fields

The merge step retains these columns:

- `Authors`
- `Article Title`
- `Source Title`
- `Author Keywords`
- `Keywords Plus`
- `Abstract`
- `Publication Year`
- `Reprint Addresses`
- `DOI`
- `DOI Link`
- `Book DOI`
- `WoS Categories`

If an export batch lacks one of these columns, the merge script creates it as empty. Later stages require at least `Article Title`, `Author Keywords`, `Abstract`, and `Publication Year`; DOI-based deduplication uses `DOI` when available.

## Manuscript snapshot

The manuscript analysis began from 59,994 records returned in the locked export set. After deduplication and exclusion of 128 records indexed with publication year 2026, 59,828 records entered eligibility filtering. The final manually adjudicated corpus contained 20,766 records.

These counts document the manuscript snapshot. A later WoSCC search may return a different result set because database records and indexing can change over time.

## Files that must remain local

Do not commit:

- WoS export batches;
- `combined_dataset.xlsx`;
- processed, included, excluded, or borderline record workbooks;
- article-level annotation workbooks;
- manual-review files containing article metadata or abstracts;
- article-level supplementary subsets.

The repository `.gitignore` blocks the standard runtime locations. Before publishing any new artifact, inspect both its columns and its cell contents.
