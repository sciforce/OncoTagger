# OncoTagger

End-to-end, rule-based pipeline for rebuilding an analytic corpus of Web of Science articles about artificial intelligence in oncology. The repository merges WoS exports, removes DOI duplicates, filters records for oncology and AI relevance, annotates cancer types, AI model families, study tasks, performance metrics, and then produces analysis tables for manual review, figures, and audit.

The research goal is to make the article-selection and annotation workflow reproducible for a review of AI applications in oncology. The primary input is a set of WoS `savedrecs*` exports. The final outputs are filtered article workbooks, binary annotation workbooks, metric categories, and aggregated counter tables under `data/results/`.

> Raw WoS exports are not part of the repository. Place your own exports in `data/raw/` and rebuild the derived files from the scripts and dictionaries committed here.

## Pipeline

Run scripts from the repository root in this order:

1. **WoS export merge**
   `src/combine_wos_exports.py` reads `data/raw/savedrecs*.csv|xls|xlsx` and writes `data/raw/combined_dataset.xlsx`.

2. **DOI deduplication**
   `src/to_delete_duplicates_by_DOI.py` reads the combined workbook, drops duplicate `DOI` values, and writes `data/processed/processed_dataset.xlsx`.

3. **Eligibility filter**
   `src/filter_dataset.py` scores oncology and AI relevance, excludes 2026 records by default, applies severe-negative gates, and writes include/manual-review/excluded/audit workbooks.

4. **Cancer typing**
   `src/main_binary.py` scans title, abstract, and author keywords with hard and soft cancer vocabularies, then creates one-hot cancer columns plus detection metadata.

5. **AI model annotation**
   `src/main_binary.py` detects AI model families from curated keyword columns and records where the AI signal was found.

6. **Task detection**
   `src/main_binary.py` detects study task labels, keeps a single `primary_task` for metric interpretation, and stores all matched tasks in `all_tasks`.

7. **Metric extraction**
   `src/main_binary.py` extracts performance metrics from abstracts, normalizes numeric values, bins them with metric-specific thresholds, and adds trace columns for context and raw values.

8. **Aggregation / counters / outputs**
   `src/counter.py` reads the binary annotation workbook and writes an analysis workbook with frequencies, year trends, cross-tabs, country summaries, source-title summaries, no-metrics summaries, and AI class mapping tables.

## Repository Structure

```text
data/
  raw/
    combined_dataset.xlsx
  processed/
    processed_dataset.xlsx
  filtered/
    filtered_dataset.xlsx
    filtered_dataset_manual_review.xlsx
    filtered_dataset_excluded.xlsx
    filtered_dataset_audit_all_decisions.xlsx
  results/
    filtered_dataset_binary_classification.xlsx
    filtered_dataset_binary_classification_analysis.xlsx
docs/
  samples/
    sample_savedrecs.xlsx
    sample_combined_dataset.xlsx
    sample_filtered_dataset_binary_classification.xlsx
sources/
  controlled vocabularies, thresholds, task priorities, mappings
src/
  combine_wos_exports.py
  to_delete_duplicates_by_DOI.py
  filter_dataset.py
  main_binary.py
  counter.py
```

Some `data/` files are generated artifacts or local analysis files. Rebuild them from your own WoS snapshot when reproducing a run.

## Input Data

Place WoS export batches in `data/raw/`. `combine_wos_exports.py` only picks files whose names start with `savedrecs` and whose extensions are `.csv`, `.xls`, or `.xlsx`.

Expected WoS columns retained by the merge step:

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

If a retained column is missing in one export, the merge script creates it as empty. Later scripts require at least `Article Title`, `Author Keywords`, `Abstract`, and `Publication Year`; DOI deduplication requires `DOI`.

## Controlled Vocabularies

The pipeline is driven by files in `sources/`:

- `cancer_keywords.csv` - hard cancer type keywords used for cancer one-hot columns.
- `cancer_keywords_soft.csv` - fallback organ/site proxies used only when no hard cancer match is found.
- `onco_terms_filter_strong.csv`, `onco_terms_filter_moderate.csv`, `onco_terms_filter_weak.csv`, `onco_terms_filter_remove.csv` - oncology eligibility terms grouped by evidence strength.
- `onco_terms_filter.csv` - legacy single-file oncology term source used as fallback.
- `ai_terms_filter_strong.csv`, `ai_terms_filter_moderate.csv`, `ai_terms_filter_weak.csv`, `ai_terms_filter_remove.csv` - AI eligibility terms grouped by evidence strength.
- `raw_ai_terms_filter.csv` - legacy single-file AI term source used as fallback.
- `ai_keywords.csv` - AI model and model-family keywords for binary annotation.
- `ai_family_map.csv` - maps AI subfamily columns to broader AI classes.
- `task_keywords.csv` - task vocabularies for classification, segmentation, prognosis, synthesis, genomic, integration, NLP, and auxiliary tasks.
- `task_priority.csv` - priority order used to select `primary_task`.
- `task_metric_priority.csv` - task-specific metric ladders used for composite and weighted performance categories.
- `metric_synonyms.csv` - metric names and textual synonyms used by the abstract parser.
- `thresholds.csv` - metric-specific cutoffs for `Very High`, `High`, `Medium`, `Low`, and `Very Low`.
- `category_scores.csv` - numeric scores for performance categories used by weighted aggregation.
- `country_synonyms.csv` - country aliases used when parsing `Reprint Addresses`.
- `wos_exclusion_categories.tsv` - WoS category trace layer for non-oncology or ambiguous records.

## Main Scripts

### `src/combine_wos_exports.py`

Input: `data/raw/savedrecs*.csv|xls|xlsx`

Output: `data/raw/combined_dataset.xlsx`

Merges WoS batches, keeps the core WoS columns listed above, and fills missing retained columns with empty values.

### `src/to_delete_duplicates_by_DOI.py`

Input: `data/raw/combined_dataset.xlsx`

Output: `data/processed/processed_dataset.xlsx`

Drops duplicate rows by exact `DOI`, keeping the first occurrence.

### `src/filter_dataset.py`

Input: `data/processed/processed_dataset.xlsx`

Outputs:

- `data/filtered/filtered_dataset.xlsx`
- `data/filtered/filtered_dataset_manual_review.xlsx`
- `data/filtered/filtered_dataset_excluded.xlsx`
- `data/filtered/filtered_dataset_audit_all_decisions.xlsx`

Adds oncology and AI scores, hit traces, flags, WoS exclusion hits, final `decision`, and `decision_reason`. The default run excludes publication year `2026`.

### `src/main_binary.py`

Input: `data/filtered/filtered_dataset.xlsx`

Output: `data/results/filtered_dataset_binary_classification.xlsx`

Adds binary cancer, AI model, and task labels; `primary_task`; `all_tasks`; metric category columns; metric trace columns; `composite_metric`; `composite_source`; `weighted_score`; and `weighted_category`.

### `src/counter.py`

Input: `data/results/filtered_dataset_binary_classification.xlsx`

Output: `data/results/filtered_dataset_binary_classification_analysis.xlsx`

Builds analysis sheets for cancer frequencies, AI model frequencies, AI class frequencies, task frequencies, task-by-year tables, cancer/model/class-by-year tables, metric-by-year tables, metric-by-task tables, cross-tabs, country summaries, source-title summaries, no-metrics summaries, and top-10 temporal trends.

## How Classification Works

### Cancer Typing

Cancer detection scans `Article Title`, `Abstract`, and `Author Keywords` in that order. Hard cancer keywords are preferred. The first field with a hard match wins and stops the scan. If no hard match is found anywhere, the first soft-only match is used as a fallback. The script keeps one-hot cancer columns plus:

- `cancer_detected_in`
- `cancer_match_level`
- `cancer_hard_detected_in`
- `cancer_soft_detected_in`

### Various Cancers Fallback

The binary classifier can mark several cancer columns for one article. `counter.py` counts selected cancer columns and writes:

- `number_of_cancer_types`
- `how_many_cancer_studied`

If more than one cancer type is detected, `how_many_cancer_studied` becomes `various cancers`. If exactly one is detected, it records `just one cancer - <cancer type>`. If none is detected, it records `not specified`.

### AI Family / AI Class Mapping

`main_binary.py` creates one-hot AI model or subfamily columns from `ai_keywords.csv`. `counter.py` then loads `ai_family_map.csv` and creates broader AI class columns by taking the maximum value across mapped subfamilies. The analysis workbook includes both an `AI Class Map` sheet and an `AI Class Breakdown` sheet.

### Task Priority

Task detection uses `task_keywords.csv` and `task_priority.csv`. The current priority order is:

1. `segmentation`
2. `classification`
3. `prognosis`
4. `synthesis`
5. `genomic`
6. `integration`
7. `nlp`
8. `auxiliary`

The first matched task by priority becomes `primary_task`. All matched tasks are preserved in `all_tasks`.

### Composite Metric

For each article, `main_binary.py` selects the first usable metric from the `primary_task` ladder in `task_metric_priority.csv`. That category is written to `composite_metric`, and the source metric name is written to `composite_source`.

Metric extraction uses sentence-level candidates, ignores relative-change language such as improvement-by or reduction-by phrasing, and ranks context as:

`external_validation > test > validation > holdout > cross_validation_summary > train > unknown`

### Weighted Category

For the same task-specific metric ladder, all usable detected metrics are converted to numeric scores via `category_scores.csv`. Metrics earlier in the task ladder receive higher weights. The weighted mean is written to `weighted_score`, and the nearest category label is written to `weighted_category`.

## Output Files

Typical generated files:

- `data/raw/combined_dataset.xlsx` - merged WoS exports.
- `data/processed/processed_dataset.xlsx` - DOI-deduplicated records.
- `data/filtered/filtered_dataset.xlsx` - included records after eligibility filtering.
- `data/filtered/filtered_dataset_manual_review.xlsx` - borderline records selected for manual review.
- `data/filtered/filtered_dataset_excluded.xlsx` - excluded records with scores and reasons.
- `data/filtered/filtered_dataset_audit_all_decisions.xlsx` - full filter audit table.
- `data/results/filtered_dataset_binary_classification.xlsx` - article-level annotation and metric workbook.
- `data/results/filtered_dataset_binary_classification_analysis.xlsx` - aggregated counters, trends, and cross-tabs.

The repository also contains sample workbooks in `docs/samples/` for orientation.

## Validation / Audit Layers

The workflow includes several audit layers:

- Filter-level `manual_review` output for ambiguous but potentially eligible records.
- Full audit workbook with include, manual-review, and exclude decisions.
- Hit-trace columns for oncology and AI evidence by field and bucket.
- `decision_reason` explaining the final eligibility decision.
- Cancer hard/soft source metadata.
- Task source metadata through `task_source_field`.
- Metric trace columns: context, raw value, sentence, source type, and suspicious extraction flag.
- `no_metrics_reported` and no-metrics analysis sheets.
- Dictionary enrichment via editable files in `sources/`.

Manual validation sets can be kept in `data/raw/` or `data/filtered/`, but they are not required for a basic pipeline run.

## Reproducibility

Recommended environment:

- Python 3.11 or newer
- spaCy `en_core_web_sm`
- Dependencies from `requirements.txt`

Setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run order:

```bash
python src/combine_wos_exports.py
python src/to_delete_duplicates_by_DOI.py
python src/filter_dataset.py
python src/main_binary.py
python src/counter.py
```

Given a fixed WoS export snapshot and fixed `sources/` dictionaries, the pipeline is deterministic. WoS itself can change over time, so record the export date/time and repository commit when publishing derived results.

## Known Limitations

- Site-unspecified cancers may remain difficult to assign to a precise organ class.
- Metastatic-site language can be ambiguous when the primary tumor site and metastatic site are both mentioned.
- Country parsing depends on `Reprint Addresses`, address formatting, and `country_synonyms.csv`; it is useful for summaries but not a full affiliation parser.
- Metric extraction is abstract-only and may miss values reported only in full text, tables, supplements, or figures.
- DOI deduplication currently uses exact `DOI` values; upstream DOI normalization should be added if exports contain inconsistent casing, URLs, or prefixes.
- Rule-based keywords are transparent and auditable, but they require periodic enrichment when new terminology appears.

## Troubleshooting

- `No files savedrecs*.xlsx|xls|csv found`: put WoS batches in `data/raw/` and keep the `savedrecs` filename prefix.
- Excel engine errors: install `openpyxl`; install `xlrd>=2.0.1` only for legacy `.xls` files.
- `en_core_web_sm` missing: run `python -m spacy download en_core_web_sm`.
- Unexpected filtering decisions: inspect `data/filtered/filtered_dataset_audit_all_decisions.xlsx`, especially score, hit, flag, and `decision_reason` columns.
- Country aliases missing: add them to `sources/country_synonyms.csv`.

## License

Code is released under the MIT License. See `LICENSE.txt`.
