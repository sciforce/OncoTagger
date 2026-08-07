# Public validation package

This directory contains only aggregate analysis outputs and redacted labels. It does not contain WoSCC article titles, abstracts, authors, keywords, addresses, DOIs, source titles, or WoS categories.

## `aggregate/`

The aggregate files cover:

- validation estimates and 95% confidence intervals;
- prediction-stratum-weighted and balanced-audit metric-detection results;
- task-unassigned sensitivity results;
- taxonomy agreement and confusion summaries;
- ordinal reported-performance validation;
- translational-signal, excluded-record, and reference-set audit summaries;
- aggregate characterization of the 36 false exclusions and the 22-record imaging subgroup.

## `redacted_labels/`

- `taxonomy_validation_labels.csv` uses arbitrary validation IDs and derived reviewer, consensus, and pipeline labels.
- `task_unassigned_300_redacted.csv` contains arbitrary validation IDs, publication year, consensus task, and two derived binary flags required for the bounded sensitivity calculation.

The private mapping from arbitrary validation IDs to licensed WoSCC records is not distributed.

## Reproduction script

`reproduce_task_unassigned_sensitivity.py` recalculates the corrected task-unassigned scenario from the redacted 300-record labels and aggregate baseline summaries, then checks the published values.

```bash
python validation/reproduce_task_unassigned_sensitivity.py
```

This is a purpose-specific manuscript analysis script. Internal QA scripts, temporary programs, and reviewer-document tooling are not included in the repository.

`SOURCE_MANIFEST.json` records LF-normalized SHA-256 checksums so verification is stable across Windows and Linux checkouts.
