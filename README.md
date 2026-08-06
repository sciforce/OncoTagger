# OncoTagger

OncoTagger is a deterministic, rule-based pipeline for abstract-level surveillance of artificial-intelligence research in oncology. This repository accompanies the manuscript **"OncoTagger: a reproducible abstract-level landscape of open-access AI-oncology articles in Web of Science"**.

## Data availability and licensing

This repository does **not** distribute Web of Science Core Collection (WoSCC) exports or row-level derivatives containing article titles, abstracts, authors, keywords, addresses, DOIs, or other licensed record metadata. Users must obtain their own authorized WoSCC export through an institutional or individual subscription.

The exact search query, required fields, export instructions, and local file layout are documented in [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md). The empty directories under `data/` are local runtime locations only; their contents are excluded by `.gitignore`.

Code and author-created dictionaries are released under the MIT License. That license does not grant rights to Clarivate content or third-party article text.

## Repository contents

```text
src/                     Core filtering, annotation, aggregation, and helper scripts
sources/                 Manuscript-locked dictionaries, thresholds, and mappings
docs/                    Data-access, validation, and reproduction documentation
validation/
  aggregate/             Aggregate validation and audit results
  redacted_labels/       Arbitrary validation IDs and derived labels only
visualizations/          Figure-generation code and aggregate figure source tables
data/                    Empty local runtime directories; generated contents are ignored
```

Internal manuscript files, reviewer documents, QA logs, temporary programs, and private validation inputs are not part of the public release.

## Quick start

Requirements:

- Python 3.11 or newer
- packages in `requirements.txt`
- spaCy model `en_core_web_sm`
- authorized WoSCC exports matching the documented query and fields

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Place export batches named `savedrecs*.csv`, `savedrecs*.xls`, or `savedrecs*.xlsx` in `data/raw/`, then run from the repository root:

```bash
python src/combine_wos_exports.py
python src/to_delete_duplicates_by_DOI.py
python src/filter_dataset.py
python src/main_binary.py
python src/counter.py
python src/build_article_to_population_ratio.py
python src/build_translational_subset.py
```

The established paths remain unchanged:

- `data/raw/` - licensed user-supplied exports and the merged workbook;
- `data/processed/` - deduplicated records after year restriction;
- `data/filtered/` - included, excluded, borderline, and audit outputs;
- `data/results/` - article-level annotations and aggregate analysis outputs.

These files are generated locally and must not be committed.

## Reproducibility documentation

- [`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md) - WoSCC query, export fields, licensing boundary, and input preparation.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) - environment, run order, expected outputs, and verification.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) - public aggregate results and redacted labels.
- [`docs/TASK_DICTIONARY.md`](docs/TASK_DICTIONARY.md) - manuscript-locked task dictionary and known coverage gap.
- [`validation/README.md`](validation/README.md) - validation file inventory and interpretation.
- [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json) - manuscript denominators, validation sample sizes, and controlled-input checksums.

## Task-dictionary release policy

The public `sources/task_keywords.csv` is the exact manuscript-locked dictionary used for the reported corpus. It was not tuned on the same 300 task-unassigned records used to identify the coverage gap, because development and evaluation on the identical records would produce a resubstitution estimate rather than independent validation.

Manual consensus assigned a primary task to 250 of those 300 records. The repository therefore:

- explicitly retains and documents the known limitation;
- publishes redacted consensus labels without WoSCC record metadata;
- publishes the bounded sensitivity analysis;
- provides a public script that reproduces the sensitivity calculations from redacted labels and aggregate inputs.

Run the public verification with:

```bash
python validation/reproduce_task_unassigned_sensitivity.py
```

## Validation scope

The public validation layer includes prediction-stratum-weighted metric-detection estimates and bootstrap confidence intervals, ordinal and taxonomy validation summaries, the task-unassigned sensitivity analysis, manual-audit summaries, and aggregate characterization of the 36 false exclusions. No article-level WoSCC text is included.

The reported labels and distributions remain pipeline-derived descriptive estimates. OncoTagger is not a validated article-level classifier, a systematic-review replacement, or a field-wide recall estimate.

## License

Repository code and author-created configuration files are released under the MIT License. See `LICENSE.txt`.
