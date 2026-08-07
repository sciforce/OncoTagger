# Visualization pipeline

This folder contains the manuscript-linked figure/table generation scripts and
their aggregate output tables. The published output files contain no WoSCC
article text or bibliographic identifiers.

## Inputs

- `RELEASE_MANIFEST.json`
- `validation/aggregate/validation_metrics_with_ci.csv`
- `validation/aggregate/metric_detection_weighted_bootstrap_ci.csv`
- `data/results/filtered_dataset_binary_classification_analysis.xlsx`
- `data/results/article to population ratio.xlsx`
- `documentation/self-documented docs/translational_subset_summary.json`

The first three inputs are public aggregate release artifacts. The remaining
workbooks and translational summary are generated locally from an authorized
WoSCC export and are intentionally excluded from Git because they derive from
licensed row-level records.

## Outputs

Run:

```powershell
python visualizations/build_visualizations.py
python visualizations/build_supplementary_outputs.py
```

The scripts write:

- publication-ready draft PNG figures to `visualizations/outputs/figures/`
- figure/table CSV data to `visualizations/outputs/data/`
- a display-item triage file, `main_display_plan.csv`, distinguishing main
  manuscript items from supplementary material
- supplementary sheet indexes, dictionary manifests, AI/cancer trend tables,
  and reported-performance summary tables to `visualizations/outputs/data/`

## Main Display Triage

npj Digital Medicine allows up to six display items for an Article. The current
triage keeps the main manuscript to four figures and one table:

1. PRISMA-inspired corpus flow diagram.
2. Temporal growth, task redistribution, and metric-reporting change.
3. Top corpus descriptors dashboard.
4. Geography and translational-signal overview.
5. Study design, denominators, and validation summary.

Dense cross-tabs, full rankings, validation confusion matrices, and article-level
translational examples should stay in Supplementary Information.

## Supplementary Outputs

`build_supplementary_outputs.py` generates the machine-readable companion files
for the Supplementary Information draft:

- `supplementary_analysis_workbook_sheet_index.csv`
- `supplementary_dictionary_manifest.csv`
- `supplementary_ai_model_trends.csv`
- `supplementary_ai_class_trends.csv`
- `supplementary_cancer_site_trends.csv`
- `supplementary_weighted_performance_by_ai_model.csv`
- `supplementary_weighted_performance_by_ai_class.csv`
- `supplementary_weighted_performance_by_cancer_site.csv`
- `supplementary_ai_model_performance_early_vs_late.csv`
- `supplementary_key_insights.json`

The reported-performance tables summarize abstract-level reported metric
categories. They are not independent model benchmarks, risk-of-bias
assessments, or evidence of clinical effectiveness.
