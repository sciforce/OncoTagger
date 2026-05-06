# OncoTagger Taxonomy Validation Public Package

This folder contains the minimal public validation materials needed to verify the task, cancer-site, and AI-family taxonomy-validation results reported in the manuscript.

## What Is Included

- `taxonomy_validation_row_level_labels_redacted.csv`  
  Redacted row-level validation labels for 1,000 validation records. This file contains validation IDs, sample names, publication years, reviewer labels, final consensus labels, and pipeline labels. It does not include article titles, abstracts, DOI values, reviewer notes, local paths, adjudication working files, or raw Web of Science exports.

- `taxonomy_validation_summary_tables.xlsx`  
  Workbook containing manuscript-ready validation tables: reviewer completion, inter-reviewer agreement, primary-sample pipeline-versus-consensus metrics, primary-task confusion matrix, per-class/per-label metrics, and challenge-sample summaries.

- `taxonomy_validation_final_metrics_public.json`  
  Machine-readable summary of validation design, consensus status, key primary-sample metrics, and challenge-sample summaries. Local file paths and internal working metadata are excluded.

- `tables_csv/`  
  CSV exports of the validation summary tables from the workbook.

- `checksums_sha256.txt`  
  SHA-256 checksums for the public package files.

## What Is Deliberately Excluded

The public package excludes raw WoSCC exports, reviewer workbooks with article titles and abstracts, adjudication workbooks with article titles and abstracts, reviewer notes, internal guides, local manifests, and manuscript drafts. These materials are not needed to verify the reported aggregate validation metrics and may contain licensed bibliographic text, local working paths, or internal preparation material.

## Validation Design

The primary taxonomy-validation dataset was a 400-record proportional year-stratified random sample from the final 20,766-record corpus. It is the only sample intended for unbiased corpus-level annotation-performance estimates.

Additional samples were analyzed separately:

- `task_unassigned_300`: complete census of records with blank `primary_task` and blank `all_tasks`; used for task-layer false-negative/error analysis.
- `cancer_no_detected_100`: challenge sample from records without pipeline-detected cancer-site labels; used for cancer-site false-negative characterization.
- `ai_family_no_detected_100`: challenge sample from records without pipeline-detected AI-family labels; used for AI-family false-negative characterization.
- `optional_positive_audit_100`: targeted positive-label audit; not an unbiased corpus-level performance sample.

Two medically trained reviewers independently annotated article title, abstract, author keywords, and Keywords Plus while blinded to pipeline outputs. Full texts were not reviewed. Disagreements were resolved by oncology-domain adjudication. The final consensus labels were used as the manual reference standard.

## Key Final Metrics

In the proportional `primary_400` sample:

- Primary-task exact agreement: 68.0%
- Primary-task Cohen's kappa: 0.508
- All-task multi-label micro-F1: 0.723
- Cancer-site multi-label micro-F1: 0.907
- AI-family multi-label micro-F1: 0.886
- Any-cancer-site detection sensitivity: 100.0%
- Any-AI-family detection sensitivity: 95.1%
- Any-AI-family detection specificity: 86.6%

Challenge-sample results should not be interpreted as corpus-level sensitivity or specificity:

- In `task_unassigned_300`, 250 of 300 records had a manually identifiable primary task and 50 remained unassigned from abstract-level metadata.
- In `cancer_no_detected_100`, 8 of 100 records had a manually detectable specific cancer site.
- In `ai_family_no_detected_100`, 5 of 100 records had a manually detectable AI-family label.

## Appropriate Interpretation

These files support the use of OncoTagger for corpus-level evidence surveillance and aggregate descriptor analysis. They do not establish perfect article-level classification, clinical effectiveness, model superiority, or clinical deployment readiness.
