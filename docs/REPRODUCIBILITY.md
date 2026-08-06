# Reproducing the OncoTagger workflow

## Environment

Use Python 3.11 or newer. Create a clean environment from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The pipeline uses relative paths rooted at the repository. Run every command from the repository root.

## Input preparation

Obtain an authorized WoSCC export by following [`DATA_ACCESS.md`](DATA_ACCESS.md). Place all batches in `data/raw/` using filenames beginning with `savedrecs` and extensions `.csv`, `.xls`, or `.xlsx`.

## Core run order

### 1. Merge WoS exports

```bash
python src/combine_wos_exports.py
```

Output: `data/raw/combined_dataset.xlsx`.

### 2. Deduplicate and enforce the publication window

```bash
python src/to_delete_duplicates_by_DOI.py
```

Output: `data/processed/processed_dataset.xlsx`.

### 3. Apply eligibility filtering

```bash
python src/filter_dataset.py
```

Outputs:

- `data/filtered/filtered_dataset.xlsx`
- `data/filtered/filtered_dataset_manual_review.xlsx`
- `data/filtered/filtered_dataset_excluded.xlsx`
- `data/filtered/filtered_dataset_audit_all_decisions.xlsx`

Borderline records require manual adjudication before a final corpus is assembled. The manuscript retained 43 of 48 borderline records and rejected 5.

### 4. Apply cancer, AI-family, task, and metric annotation

```bash
python src/main_binary.py
```

Output: `data/results/filtered_dataset_binary_classification.xlsx`.

### 5. Build aggregate summaries

```bash
python src/counter.py
```

Output: `data/results/filtered_dataset_binary_classification_analysis.xlsx`.

### 6. Build article-supporting derivatives

```bash
python src/build_article_to_population_ratio.py
python src/build_translational_subset.py
```

These helpers consume locally generated article-level outputs. Their row-level outputs remain local and must not be committed.

### 7. Build figures and aggregate figure tables

```bash
python visualizations/build_visualizations.py
python visualizations/build_supplementary_outputs.py
```

## Path compatibility

The public-repository cleanup does not change the paths expected by the pipeline. Empty runtime directories are retained under `data/`, and each script creates its output directory when needed. `.gitignore` prevents generated licensed or row-level files from being added to Git.

## Verification

Run the bounded task-unassigned sensitivity verification:

```bash
python validation/reproduce_task_unassigned_sensitivity.py
```

For a manuscript-snapshot reproduction, compare the main stage counts with the release manifest and record the exact repository commit, dictionary checksums, WoS export date/time, and source-record count.

## Reproducibility boundary

The public code and controlled dictionaries make the computational workflow inspectable and rerunnable. Exact regeneration of the manuscript corpus additionally requires authorized access to the locked WoSCC export snapshot, which cannot be redistributed.
