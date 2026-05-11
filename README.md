# OncoTagger

End-to-end, rule-based pipeline for rebuilding an analytic corpus of Web of Science articles about artificial intelligence in oncology. The repository supports the manuscript **"A reproducible bibliographic landscape of AI in oncology"**.

## What this repository does

OncoTagger is a deterministic, abstract-level evidence-surveillance pipeline. It starts from user-supplied Web of Science Core Collection (WoSCC) export batches in `data/raw/`; raw WoSCC exports are not redistributed because they are licensed Clarivate content. The core outputs are a filtered workbook, a binary annotation workbook, an aggregate analysis workbook, manual validation artifacts, and manuscript-facing derived release artifacts such as population-normalized country outputs and candidate translational-signal subsets. Reproducibility depends on rerunning the documented script order from a fixed WoSCC snapshot and fixed dictionaries under `sources/`.

## Pipeline

### Run order

Run scripts from the repository root in this order:

1. **WoS export merge**
   `src/combine_wos_exports.py` reads `data/raw/savedrecs*.{csv,xls,xlsx}` and writes `data/raw/combined_dataset.xlsx`. This means files in `data/raw/` with names beginning with `savedrecs` and one of the supported extensions.

2. **Deduplication and year exclusion**
   `src/to_delete_duplicates_by_DOI.py` reads the combined workbook, normalizes non-empty `DOI` values, applies a conservative title/year fallback, excludes publication year `2026`, reports each removal count, and writes `data/processed/processed_dataset.xlsx`.

3. **Eligibility filter**
   `src/filter_dataset.py` scores oncology and AI relevance, keeps a publication-year exclusion safeguard, applies severe-negative gates, and writes:
   - `data/filtered/filtered_dataset.xlsx` - automatically included records.
   - `data/filtered/filtered_dataset_manual_review.xlsx` - borderline records requiring manual review.
   - `data/filtered/filtered_dataset_excluded.xlsx` - automatically excluded records.
   - `data/filtered/filtered_dataset_audit_all_decisions.xlsx` - full audit table with scores, hits, flags, final decision, and `decision_reason`.

4. **Cancer typing**
   `src/main_binary.py` scans title, abstract, and author keywords with hard and soft cancer vocabularies, then creates one-hot cancer columns plus detection metadata.

5. **AI model annotation**
   `src/main_binary.py` detects AI model families from curated keyword columns and records where the AI signal was found.

6. **Task detection**
   `src/main_binary.py` detects study task labels. The script writes a single task label to the `primary_task` output column for downstream metric interpretation and preserves all matched task labels in the `all_tasks` output column.

7. **Metric extraction**
   `src/main_binary.py` extracts performance metrics from abstracts, normalizes numeric values, bins them with metric-specific thresholds, and adds trace columns for context and raw values.

8. **Aggregation, counters, and output tables**
   `src/counter.py` reads the binary annotation workbook and writes an analysis workbook with frequencies, year trends, cross-tabs, country summaries, source-title summaries, no-metrics summaries, and AI class mapping tables.

9. **Article-facing helper outputs**
   - `src/build_article_to_population_ratio.py`
     - Input: `data/results/filtered_dataset_binary_classification_analysis.xlsx`
     - Input: `sources/total-population-by-country-2025.csv`
     - Output: `data/results/article to population ratio.xlsx`
     - Purpose: derives population-normalized corresponding-author country outputs.
   - `src/build_translational_subset.py`
     - Input: `data/results/filtered_dataset_binary_classification.xlsx`
     - Output: manuscript-facing translational subset workbook and summary files.
     - Purpose: derives a cautious candidate translational-signal subset using external-validation context and implementation-oriented title signals.

## Repository Structure

```text
data/
  manual validation/
    manual reference sets, audit CSVs, and validation summaries
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
    article to population ratio.xlsx
  supplementary material/
    synchronized supplementary workbooks and release summaries
docs/
  samples/
    sample_savedrecs.xlsx
    sample_combined_dataset.xlsx
    sample_filtered_dataset_binary_classification.xlsx
sources/
  controlled vocabularies, thresholds, task priorities, mappings, population source tables
src/
  build_article_to_population_ratio.py
  build_translational_subset.py
  combine_wos_exports.py
  to_delete_duplicates_by_DOI.py
  filter_dataset.py
  main_binary.py
  counter.py
```

Several files under `data/` are generated outputs from the pipeline rather than manually edited source files. To reproduce them, place your own WoSCC export batches in `data/raw/` and run the scripts in the documented [Run order](#run-order). The raw WoSCC exports are not redistributed because they are licensed Clarivate content.

Input and output roles:

- User-supplied input: WoSCC export batches under `data/raw/`.
- Controlled source files: dictionaries, thresholds, priorities, and mappings under `sources/`.
- Generated outputs: `data/raw/combined_dataset.xlsx`, `data/processed/processed_dataset.xlsx`, `data/filtered/*.xlsx`, and `data/results/*.xlsx`.
- Manuscript-facing derived release artifacts: supplementary packages and submission-stage derived files, when present, under `data/supplementary material/` or the local submission staging area.

## Input Data

Place WoS export batches in `data/raw/`. The merge script looks for WoS export files whose filenames start with `savedrecs` and end in `.csv`, `.xls`, or `.xlsx`, for example `savedrecs1.xlsx`, `savedrecs(2).csv`, or `savedrecs_batch_03.xlsx`. In shell-style notation this is `data/raw/savedrecs*.{csv,xls,xlsx}`. The asterisk is a wildcard pattern, not a footnote marker.

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
- `total-population-by-country-2025.csv` - population denominators for the article-to-population helper output.
- `wos_exclusion_categories.tsv` - WoS category trace layer for non-oncology or ambiguous records.

## Main Scripts

### `src/combine_wos_exports.py`

Input: `data/raw/savedrecs*.{csv,xls,xlsx}`. This means files in `data/raw/` with names beginning with `savedrecs` and one of the supported extensions.

Output: `data/raw/combined_dataset.xlsx`

Merges WoS batches, keeps the core WoS columns listed above, and fills missing retained columns with empty values.

### `src/to_delete_duplicates_by_DOI.py`

Input: `data/raw/combined_dataset.xlsx`

Output: `data/processed/processed_dataset.xlsx`

Removes duplicate rows by normalized non-empty `DOI`, keeping the first occurrence. Rows without DOI are not collapsed together. A conservative `Article Title` + `Publication Year` fallback removes likely no-DOI duplicates or DOI/no-DOI copies while preserving rows with conflicting DOI values. Publication year `2026` is excluded at this preprocessing stage and the terminal summary reports how many records were removed for that year.

### `src/filter_dataset.py`

Input: `data/processed/processed_dataset.xlsx`

Outputs:

- `data/filtered/filtered_dataset.xlsx`
- `data/filtered/filtered_dataset_manual_review.xlsx`
- `data/filtered/filtered_dataset_excluded.xlsx`
- `data/filtered/filtered_dataset_audit_all_decisions.xlsx`

Adds oncology and AI scores, hit traces, flags, WoS exclusion hits, final `decision`, and `decision_reason`. The default run keeps a publication-year exclusion safeguard for `2026`, which should normally remove zero records after the preprocessing step above.

### `src/main_binary.py`

Input: `data/filtered/filtered_dataset.xlsx`

Output: `data/results/filtered_dataset_binary_classification.xlsx`

`src/main_binary.py` adds:

- binary cancer-site columns;
- binary AI model/family columns;
- binary task columns;
- `primary_task`;
- `all_tasks`;
- metric category columns;
- metric trace columns;
- `composite_metric`;
- `composite_source`;
- `weighted_score`;
- `weighted_category`.

### `src/counter.py`

Input: `data/results/filtered_dataset_binary_classification.xlsx`

Output: `data/results/filtered_dataset_binary_classification_analysis.xlsx`

Builds analysis sheets grouped as:

- frequency tables:
  - cancer-site frequencies;
  - AI model frequencies;
  - AI class frequencies;
  - task frequencies;
- temporal tables:
  - task-by-year;
  - cancer-by-year;
  - AI model-by-year;
  - AI class-by-year;
  - metric-by-year;
- cross-tabulations:
  - task x cancer;
  - task x model;
  - task x AI class;
  - metric-category cross-tabs against cancer, AI model, and AI class outputs;
- reporting-quality summaries:
  - no-metrics summaries;
  - source-title summaries;
- geography outputs:
  - corresponding-author country summaries;
- trend summaries:
  - top-10 temporal trends.

### `src/build_article_to_population_ratio.py`

Inputs:

- `data/results/filtered_dataset_binary_classification_analysis.xlsx`
- `sources/total-population-by-country-2025.csv`

Output:

- `data/results/article to population ratio.xlsx`

Purpose: derives population-normalized corresponding-author country outputs.

### `src/build_translational_subset.py`

Input:

- `data/results/filtered_dataset_binary_classification.xlsx`

Output:

- manuscript-facing translational subset workbook and summary files.

Purpose: derives a cautious candidate translational-signal subset using external-validation context and implementation-oriented title signals.

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
- `data/processed/processed_dataset.xlsx` - deduplicated records after publication year `2026` exclusion.
- `data/filtered/filtered_dataset.xlsx` - included records after eligibility filtering.
- `data/filtered/filtered_dataset_manual_review.xlsx` - borderline records selected for manual review.
- `data/filtered/filtered_dataset_excluded.xlsx` - excluded records with scores and reasons.
- `data/filtered/filtered_dataset_audit_all_decisions.xlsx` - full filter audit table.
- `data/results/filtered_dataset_binary_classification.xlsx` - article-level annotation and metric workbook.
- `data/results/filtered_dataset_binary_classification_analysis.xlsx` - aggregated counters, trends, and cross-tabs.
- `data/results/article to population ratio.xlsx` - population-normalized corresponding-author country output.
- `data/manual validation/` - manual audit files, ordinal validation tables, and detection-audit summaries.
- `data/supplementary material/` - synchronized article-supporting release artifacts and summary manifests.

The repository also contains sample workbooks in `docs/samples/` for orientation.

## Validation / Audit Layers

The workflow includes several audit layers:

- Filter-level `manual_review` output for ambiguous but potentially eligible records.
- Full audit workbook with include, manual-review, and exclude decisions.
- Hit-trace columns for oncology and AI evidence by field and bucket.
- `decision_reason` explaining the final eligibility decision.
- Cancer hard/soft source metadata.
- Task source metadata through `task_source_field`.
- Metric trace columns:
  - context;
  - raw value;
  - sentence;
  - source type;
  - suspicious extraction flag.
- `no_metrics_reported` column and no-metrics analysis sheets.
- Dictionary enrichment via editable files in `sources/`.
- Repository-facing manual validation artifacts in `data/manual validation/`.
- Repository-facing supplementary release artifacts in `data/supplementary material/`.

Manual validation and supplementary release folders are not required for a basic pipeline run, but they are useful for article support, reproducibility, and auditability.

## Reproducibility

Recommended environment:

- Python 3.11 or newer
- Python packages from `requirements.txt`, including `pandas`, `numpy`, `spacy`, `tqdm`, and `pycountry`
- Excel engine dependency: `openpyxl` for `.xlsx` files
- Optional Excel engine dependency for legacy `.xls` exports: `xlrd>=2.0.1`
- spaCy English language model: `en_core_web_sm`

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
python src/build_article_to_population_ratio.py
# optional article-facing helper
python src/build_translational_subset.py
```

Given a fixed WoS export snapshot and fixed `sources/` dictionaries, the pipeline is deterministic. WoS itself can change over time, so record the export date/time and repository commit when publishing derived results.

## Known Limitations

- Site-unspecified cancers may remain difficult to assign to a precise organ class.
- Metastatic-site language can be ambiguous when the primary tumor site and metastatic site are both mentioned.
- Country parsing depends on `Reprint Addresses`, address formatting, and `country_synonyms.csv`; it is useful for summaries but not a full affiliation parser.
- Metric extraction is abstract-only and may miss values reported only in full text, tables, supplements, or figures.
- Deduplication depends on DOI and title/year metadata quality; rows with conflicting DOI values but the same normalized title/year are preserved for manual review rather than automatically collapsed.
- Rule-based keywords are transparent and auditable, but they require periodic enrichment when new terminology appears.

## Troubleshooting

- `No files matching savedrecs*.{csv,xls,xlsx} found`: put WoS batches in `data/raw/` and keep the `savedrecs` filename prefix.
- Excel engine errors: install `openpyxl` for `.xlsx` files; install `xlrd>=2.0.1` only for legacy `.xls` files.
- Missing spaCy English language model: if `en_core_web_sm` is missing, run `python -m spacy download en_core_web_sm`.
- Unexpected filtering decisions: inspect `data/filtered/filtered_dataset_audit_all_decisions.xlsx`, especially columns containing `score`, `hit`, or `flag`, plus the exact `decision_reason` column. These fields explain why a record was included, excluded, or routed to manual review.
- Country aliases missing: add them to `sources/country_synonyms.csv`.
- README/release mismatch: treat `data/supplementary material/` as a synchronized release layer, not as the sole source of truth for rerunning the core pipeline.

## License

Code is released under the MIT License. See `LICENSE.txt`.
