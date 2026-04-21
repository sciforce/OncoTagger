# Manual Validation Artifacts

This folder contains the manual-audit and validation artifacts that support the
reproducibility claims of the current oncology AI literature-mapping pipeline.

Contents

- `100 random metric-reported_audited.csv`
  Manual audit of 100 records sampled from pipeline metric-reported outputs.
- `100 random no metric-reported_audited.csv`
  Manual audit of 100 records sampled from pipeline no-metric-reported outputs.
- `manual_dataset_binary_classification.xlsx`
  Manual reference set used for primary ordinal validation.
- `manual_dataset_binary_classification_analysis.xlsx`
  Derived analysis workbook for the manual reference set.
- `primary_validation_400_analysis.json`
  Summary metrics for the 400-record ordinal validation.
- `primary_validation_400_analysis_tables.xlsx`
  Tables supporting the 400-record ordinal validation.
- `secondary_detection_audit_analysis.json`
  Summary metrics for the 200-record metric-presence audit.
- `secondary_detection_audit_analysis_tables.xlsx`
  Tables supporting the 200-record metric-presence audit.

Interpretation

- The 400-record primary validation targets ordinal agreement between manual
  reference labels and algorithmic outputs.
- The 200-record secondary audit targets only metric presence versus
  no-metric presence.
- These files support corpus-level validation claims and should not be
  overinterpreted as full article-level gold-standard annotation for every
  pipeline output.

