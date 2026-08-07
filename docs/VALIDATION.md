# Validation and audit outputs

The public validation package contains aggregate results and redacted labels only. It excludes article titles, abstracts, authors, keywords, addresses, DOIs, source titles, and WoS categories.

## Validation streams

- Primary 400-record ordinal validation of weighted and composite reported-performance categories.
- Secondary 200-record metric-detection audit, sampled equally from pipeline metric-reported and no-metric-reported strata.
- Prediction-stratum-weighted corpus-level metric-detection estimates with stratified bootstrap 95% confidence intervals.
- Two-reviewer taxonomy validation for primary task, all task labels, cancer-site labels, and AI-family labels.
- Complete manual review of the 300 pipeline task-unassigned records.
- Random audit of 250 automatically excluded WoSCC export records.
- Candidate translational-signal and near-miss audits.
- Curated reference-set coverage check.

## Weighted and balanced-audit estimates

The primary corpus-level metric-detection estimates are prediction-stratum weighted because the secondary audit deliberately sampled 100 pipeline-positive and 100 pipeline-negative records from corpus strata of 12,538 and 8,228 records. The unweighted estimates remain available and are labelled as balanced-audit results rather than corpus-level estimates.

The corresponding public files are:

- `aggregate/metric_detection_weighted_bootstrap_ci.csv`
- `aggregate/metric_detection_prediction_stratum_weighted_estimates.csv`
- `aggregate/metric_detection_balanced_audit_metrics.csv`
- `aggregate/metric_detection_confusion_matrix.csv`

The row-level error table is not public because it contains licensed article metadata and abstracts.

## Task-unassigned sensitivity analysis

Manual consensus assigned a primary task to 250 of 300 pipeline-unassigned records. The analysis corrects only this complete observed stratum and leaves every other pipeline label unchanged.

Public inputs and outputs:

- `redacted_labels/task_unassigned_300_redacted.csv`
- `aggregate/task_unassigned_sensitivity.csv`
- `reproduce_task_unassigned_sensitivity.py`

The redacted input uses arbitrary validation IDs and derived flags. It contains no record text or bibliographic identifiers. The analysis is bounded: it does not correct disagreements among already assigned task labels and does not establish article-level task accuracy.

## Excluded-record characterization

The 36 false exclusions are described only through aggregate distributions by year, task, data domain, study type, cancer-site group, and corresponding-author country. The imaging subgroup is summarized by technique, modality, failure path, and overlapping rule patterns. No article-level source records are included.

These distributions indicate plausible directions of undercoverage but are not stratum-specific recall estimates or corrected corpus totals.

## Redacted taxonomy labels

`redacted_labels/taxonomy_validation_labels.csv` contains arbitrary validation IDs, publication year, reviewer labels, consensus labels, and pipeline-derived labels. The mapping from validation IDs to WoSCC records is private and is not distributed.

## Confidence intervals

Proportions use Wilson 95% confidence intervals. Kappa, Jaccard, and F1 statistics use record-level bootstrap 95% confidence intervals. Prediction-stratum-weighted metric-detection estimates use a record-level bootstrap sampled independently within each prediction stratum.
