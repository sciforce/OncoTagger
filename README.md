# OncoTagger — Reproducible Workflow

End-to-end, scriptable pipeline to rebuild the analytic corpus and summary tables for the manuscript **“Artificial Intelligence in Oncology: A Global Review of Emerging Trends, Promising Models, and Unexplored Areas (2019–2024)”**.

The workflow ingests **Web of Science Core Collection (WoSCC)** exports, merges batches, de-duplicates by DOI, filters to oncology + AI, annotates cancers/models/tasks, parses performance metrics, and produces counts and temporal trend tables.

> **Legal/ethics**
> Raw WoS “savedrecs\*” exports are **not** distributed in this repository (license terms; dynamic index). Put your own exports into `data/raw/` and rebuild. The pipeline then creates a local `combined_dataset.xlsx`; all scripts and dictionaries are provided for full reproducibility from your WoS snapshot.

---

## Repository layout

```text
.
├─ data/
│  ├─ raw/
│  │  ├─ .gitkeep
│  │  └─ combined_dataset.xlsx              # created by combine_wos_exports.py from your savedrecs* batches
│  ├─ processed/
│  │  ├─ .gitkeep
│  │  └─ processed_dataset.xlsx             # after DOI de-duplication
│  ├─ filtered/
│  │  ├─ .gitkeep
│  │  └─ filtered_dataset.xlsx              # after oncology+AI keyword filter
│  └─ results/
│     ├─ .gitkeep
│     ├─ filtered_dataset_binary_classification.csv
│     └─ filtered_dataset_binary_classification_wide.xlsx
├─ docs/
│  └─ samples/                              # optional figure snippets / example outputs
├─ sources/                                 # controlled vocabularies + thresholds
│  ├─ ai_keywords.csv
│  ├─ cancer_keywords.csv
│  ├─ category_scores.csv
│  ├─ country_synonyms.csv
│  ├─ metric_synonyms.csv
│  ├─ onco_terms_filter.csv
│  ├─ raw_ai_terms_filter.csv
│  ├─ task_keywords.csv
│  ├─ task_metric_priority.csv
│  ├─ task_priority.csv
│  ├─ thresholds.csv
│  └─ wos_exclusion_categories.csv
├─ src/
│  ├─ combine_wos_exports.py                # merge savedrecs* (CSV/XLS/XLSX) → data/raw/combined_dataset.xlsx
│  ├─ to_delete_duplicates_by_DOI.py        # drop duplicate DOIs → data/processed/processed_dataset.xlsx
│  ├─ filter_datased.py                     # rule-based oncology+AI filter → data/filtered/filtered_dataset.xlsx
│  ├─ main_binary.py                        # one-hot labels (cancers/models/tasks) + metric parsing + summaries
│  └─ new_counter.py                        # aggregate counts and temporal tables → data/results/*
├─ .gitignore
└─ README.md                                # this file
```

---

## Prerequisites

* **Python** ≥ 3.11
* Install the minimal stack:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install pandas numpy openpyxl xlrd tqdm pycountry
```

> `openpyxl` handles .xlsx; `xlrd>=2.0.1` is needed only if you have legacy `.xls` exports.

---

## Data source (your snapshot)

* **Index:** Web of Science Core Collection (WoSCC)
* **Window:** 2019-01-01 … 2024-12-31
* **Language:** English **Doc type:** Article **Open Access:** yes
* **Snapshot timestamp (paper run):** **05 Jan 2025 17:29 EET**
* Rationale for a single index (schema stability, reproducibility) is discussed in the manuscript.

---

## What you need to supply

1. Export your WoSCC results in batches (WoS limit ≈ 1,000 per file) as **CSV/XLS/XLSX**.
2. Place **all** downloaded files into:

```
data/raw/
```

Filenames may follow the default WoS pattern (`savedrecs(1).xlsx`, `savedrecs2.csv`, …). We do **not** commit these raw files.

---

## Step-by-step pipeline

> Run from the repository root. Each script writes into `data/...` as shown.

### 0) Combine WoS exports → `data/raw/combined_dataset.xlsx`

```bash
python src/combine_wos_exports.py
```

**Input:** any `data/raw/*.csv|*.xls|*.xlsx`
**Output:** `data/raw/combined_dataset.xlsx` (keeps core columns: Authors, Article Title, Source Title, Author Keywords, Keywords Plus, Abstract, Publication Year, Reprint Addresses, DOI, DOI Link, Book DOI, WoS Categories).

### 1) De-duplicate by DOI

```bash
python src/to_delete_duplicates_by_DOI.py
```

**Input:** `data/raw/combined_dataset.xlsx`
**Output:** `data/processed/processed_dataset.xlsx`
(robust case-/prefix-insensitive DOI matching)

### 2) Eligibility filter (oncology + AI, year window, OA/article)

```bash
python src/filter_datased.py
```

**Input:** `data/processed/processed_dataset.xlsx`
**Output:** `data/filtered/filtered_dataset.xlsx`
(uses `sources/onco_terms_filter.csv` and `sources/raw_ai_terms_filter.csv`)

### 3) Annotation & metric parsing

```bash
python src/main_binary.py
```

**Input:** `data/filtered/filtered_dataset.xlsx`
**Outputs (to `data/results/`):**

* `filtered_dataset_binary_classification.xlsx` / `.csv` (original fields + one-hot **cancers/models/tasks** + per-study **Composite**/**Weighted** accuracy categories)

**Under the hood (brief):**

* **Tasks:** primary task via curated keywords with fixed priority
  `classification → segmentation → prognosis → synthesis → integration → NLP → genomic → auxiliary`
* **Metrics:** regex around `sources/metric_synonyms.csv`; numbers normalised to 0–1; binned by **metric-specific thresholds** (`sources/thresholds.csv`)
* **Summaries:**

  * **Composite** = first task-appropriate metric on the priority ladder
  * **Weighted** = all detected metrics averaged with descending weights (mapping in `sources/category_scores.csv`)
  * Task-specific metric ladders in `sources/task_metric_priority.csv` (e.g., Segmentation: Dice→IoU→HD95; Prognosis: C-index→R²→MAE→RMSE; Classification: ROC-AUC→PR-AUC→F1→precision→recall)

### 4) Counters & trends (tables for figures)

```bash
python src/new_counter.py
```

**Input:** `data/results/filtered_dataset_binary_classification.xlsx`
**Outputs (to `data/results/`):** Excel/CSV sheets for:

* Year × task counts; Year × accuracy (Weighted & Composite)
* Year × ROC-AUC categories
* Cross-tabs (task × cancer, task × model, etc.)
* Top-10 cancer/model temporal trends

> Visualisations in the paper were created in **Flourish** and **Datawrapper** from these tables.

---

## Reproducibility notes

* **Snapshot drift.** WoSCC changes over time (new records, retractions); rerunning later may slightly change totals.
* **Determinism.** Given a fixed snapshot and the provided dictionaries in `sources/`, outputs are deterministic.
* Please record your **WoS timestamp** and the **Git commit** of this repo when publishing derived results.

---

## Methods cheat-sheet

* **Cancer types:** 33 organ-site classes from curated synonyms; multi-site papers labelled *various cancers*.
* **AI models:** \~33 families (CNN, random forest, logistic regression, gradient boosting, transformers, GNNs, GANs, U-Net, …).
* **Metrics:** ROC-/PR-AUC, F1, precision/recall, specificity, Dice, IoU, HD95, MCC, Cohen’s κ, C-index, etc.; proxy percentages kept only for sensitivity checks.
* **Validation (paper):** optional 400-abstract manual audit with weighted Cohen’s κ and bootstrap CIs.

---

## Data availability

Processed, analysis-ready tables live under `data/results/`.
Raw WoSCC exports are **not shared**; the locally created `data/raw/combined_dataset.xlsx` is sufficient to reproduce all tables.

---

## Code availability & licence

All code is under `src/` and released under the **MIT License**.
A long-term archive of this repository will be posted on Zenodo (DOI **TBA**).

---

## Troubleshooting

* **“No files found”:** ensure your exports are in `data/raw/` (CSV/XLS/XLSX).
* **Excel engine error:** install `openpyxl` (and `xlrd>=2.0.1` for legacy `.xls`).
* **Country parsing quirks:** add aliases to `sources/country_synonyms.csv`.

---

## Contact

**Bohdan Khilchevskyi, MD** — Medical Oncologist, Healthcare Data Analyst (SciForce, Ukraine)
[bohdan.khilchevskyi@sciforce.tech](mailto:bohdan.khilchevskyi@sciforce.tech)

**Denys Kaduk, MD** — Healthcare Data Analyst, Tech Lead (SciForce, Ukraine)
[denys.kaduk@sciforce.tech](mailto:denys.kaduk@sciforce.tech)

