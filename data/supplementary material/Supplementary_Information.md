Supplementary Information
=========================

Manuscript
----------
A reproducible bibliographic landscape of AI in oncology

Purpose
-------
This Supplementary Information file documents the search strategy, taxonomy inputs, aggregate workbook structure, and additional temporal and reported-performance analyses underlying the main manuscript. Large machine-readable tables should be uploaded as Supplementary Data files rather than embedded as long static tables in the PDF.

Supplementary Note 1. Web of Science search strategy
----------------------------------------------------
The synchronized corpus was derived from Web of Science Core Collection exports using the following query, restricted to journal articles, English language, open access status, and publication years 2019-2025.

(((((ALL=("cancer-free survival"
OR "tumor-free survival"
OR "recurrence-free survival"
OR "RFS"
OR "relapse-free survival"
OR "metastasis-free survival"
OR "MFS"
OR "metastasis absence survival"
OR "no metastasis survival"
OR cancer*
OR tumor*
OR neoplas*
OR onco*
OR carcinoma*
OR adenocarcinoma*
OR sarcoma*
OR malignan*
OR tnbc
OR lumpectomy
OR mastectomy
OR familial adenomatous polyposis
OR castration-resistant
OR glioma
OR astrocytoma
OR glioblastoma
OR brainstem tumor
OR medulloblastoma
OR meningioma
OR intrahepatic cholangiocarcinoma
OR melanoma
OR mesothelioma
OR thymoma
OR osteosarcoma
OR chondrosarcoma
OR multiple myeloma
OR leukemia
OR lymphoma))
AND ALL=(artificial intelligent*
OR computational intelligence
OR intelligent learning
OR machine learning
OR image* segmentation
OR supervised learning
OR deep network*
OR deep learning
OR neural network*
OR neural learning
OR neural nets model
OR artificial neural network
OR ai
OR cnn
OR convolutional neural network))
NOT ti=(diabetes
OR atrial fibrillation
OR pulmonary embolism
OR alzheimer*
OR covid-19
OR schizophrenia
OR multiple sclerosis
OR ulcerative colitis))
AND py=(2019-2025))
AND dt=(article)
AND la=(english)
AND OPEN access

Supplementary Note 2. Curated dictionaries and taxonomy files
-------------------------------------------------------------
The reproducible pipeline uses curated CSV/TSV dictionaries for eligibility filtering, cancer-site tagging, AI model-family tagging, task detection, metric extraction, metric thresholding, and country-name harmonization. The machine-readable manifest is provided as `visualizations/outputs/data/supplementary_dictionary_manifest.csv`.

Key dictionary groups:
- Oncology and AI eligibility filters: `onco_terms_filter_*`, `ai_terms_filter_*`, and `wos_exclusion_categories.tsv`.
- Article-level descriptor taxonomies: `cancer_keywords.csv`, `cancer_keywords_soft.csv`, `ai_keywords.csv`, `ai_family_map.csv`, `task_keywords.csv`, and `task_priority.csv`.
- Performance-metric parsing and category assignment: `metric_synonyms.csv`, `thresholds.csv`, `task_metric_priority.csv`, and `category_scores.csv`.
- Country normalization: `country_synonyms.csv` and the 2025 population denominator file.

Supplementary Note 2a. Author-defined ordinal performance-category thresholds
-------------------------------------------------------------------------------
The performance categories used in the manuscript (`very high`, `high`, `medium`, `low`, and `very low`) are author-defined ordinal proxy categories created for standardizing heterogeneous abstract-reported performance metrics across the corpus. They were not adopted from an external clinical-performance standard and should not be interpreted as validated grades of clinical utility, risk of bias, prospective effectiveness, or deployment readiness.

The thresholds were developed by the authors after topic-specific review of common AI-oncology metric reporting patterns and encoded in `sources/thresholds.csv`. Their purpose is to place diverse metrics, such as ROC-AUC, accuracy, Dice, C-index, MAE, RMSE, and false-positive rate, onto a common descriptive scale for corpus-level surveillance. Metrics where higher values indicate better reported performance use greater-than-or-equal thresholds; error metrics where lower values indicate better reported performance use less-than-or-equal thresholds. The readable threshold table below is also provided as `visualizations/outputs/data/supplementary_metric_thresholds_readable.csv`.

Supplementary Table 12. Author-defined ordinal proxy thresholds for reported performance metrics.

| Metric | Direction | Very high | High | Medium | Low | Very low |
| --- | --- | --- | --- | --- | --- | --- |
| precision | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.5 and < 0.7 | < 0.5 |
| sensitivity | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| specificity | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| f1-score | Higher values indicate better reported performance | >= 0.85 | >= 0.75 and < 0.85 | >= 0.6 and < 0.75 | >= 0.5 and < 0.6 | < 0.5 |
| roc-auc | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| pr-auc | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| balanced accuracy | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| mcc | Higher values indicate better reported performance | >= 0.7 | >= 0.5 and < 0.7 | >= 0.3 and < 0.5 | >= 0.1 and < 0.3 | < 0.1 |
| cohen's kappa | Higher values indicate better reported performance | >= 0.7 | >= 0.5 and < 0.7 | >= 0.3 and < 0.5 | >= 0.1 and < 0.3 | < 0.1 |
| npv | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |
| fpr | Lower values indicate better reported performance | <= 0.05 | > 0.05 and <= 0.1 | > 0.1 and <= 0.2 | > 0.2 and <= 0.3 | > 0.3 |
| dice | Higher values indicate better reported performance | >= 0.8 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | >= 0.5 and < 0.6 | < 0.5 |
| iou | Higher values indicate better reported performance | >= 0.7 | >= 0.6 and < 0.7 | >= 0.4 and < 0.6 | >= 0.3 and < 0.4 | < 0.3 |
| hd95 | Lower values indicate better reported performance | <= 2 | > 2 and <= 5 | > 5 and <= 10 | > 10 and <= 20 | > 20 |
| mae | Lower values indicate better reported performance | <= 2 | > 2 and <= 5 | > 5 and <= 10 | > 10 and <= 15 | > 15 |
| rmse | Lower values indicate better reported performance | <= 2 | > 2 and <= 5 | > 5 and <= 10 | > 10 and <= 15 | > 15 |
| r2 | Higher values indicate better reported performance | >= 0.85 | >= 0.7 and < 0.85 | >= 0.5 and < 0.7 | >= 0.3 and < 0.5 | < 0.3 |
| c-index | Higher values indicate better reported performance | >= 0.8 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | >= 0.55 and < 0.6 | < 0.55 |
| accuracy | Higher values indicate better reported performance | >= 0.9 | >= 0.8 and < 0.9 | >= 0.7 and < 0.8 | >= 0.6 and < 0.7 | < 0.6 |

Supplementary Note 3. Aggregate workbook structure
--------------------------------------------------
The main aggregate workbook is `data/supplementary material/filtered_dataset_binary_classification_analysis.xlsx`. Its sheet index, dimensions, first columns, and suggested roles are listed in `visualizations/outputs/data/supplementary_analysis_workbook_sheet_index.csv`.

The workbook contains:
- Corresponding-author geography: raw country counts, country-by-year, country-by-task, and missingness.
- Metric reporting: composite metric sources, no-metric rates, metric categories by year/task/source/cancer, and metric-specific cross-tabs.
- Descriptor frequencies: primary tasks, cancer sites, AI model families, AI classes, and multi-label count distributions.
- Cross-tabs: task by cancer, task by AI models/classes, weighted/composite categories by cancer and AI.
- Temporal dynamics: tasks by year, cancer sites by year, AI models/classes by year, and top-10 descriptors by year.

Supplementary Note 4. Additional temporal and reported-performance analyses
---------------------------------------------------------------------------
Additional machine-readable summary tables generated from the article-level workbook are provided in `visualizations/outputs/data/`:

- `supplementary_ai_model_trends.csv`
- `supplementary_ai_class_trends.csv`
- `supplementary_cancer_site_trends.csv`
- `supplementary_weighted_performance_by_ai_model.csv`
- `supplementary_weighted_performance_by_ai_class.csv`
- `supplementary_weighted_performance_by_cancer_site.csv`
- `supplementary_ai_model_performance_early_vs_late.csv`
- `supplementary_metric_thresholds_readable.csv`
- `supplementary_key_insights.json`

These tables support additional observations discussed in the manuscript. They show rising relative shares for gradient boosting, penalized regression, ResNet-family models, vision transformers, survival-specific models, CNN backbones, and vision-foundation approaches; relative share declines for SVM and general MLP/ANN/DNN labels; and broader disease coverage beyond the historically largest breast, brain/CNS, and lung cancer categories.

Performance-related supplementary tables should be interpreted cautiously. `weighted_category` and `composite_metric` summarize reported abstract-level metrics using the author-defined ordinal proxy thresholds in Supplementary Table 12. They do not represent independent model benchmarking, clinical utility, prospective effectiveness, deployment readiness, or risk-of-bias assessment. Differences by AI family and cancer site are likely influenced by task mix, endpoint type, reporting conventions, metric choice, and the higher metric-reporting maturity of image-classification and segmentation benchmarks.

Supplementary Note 5. Manual validation of taxonomy annotation
--------------------------------------------------------------
Manual validation of the task, cancer-site, and AI-family annotation layers used blinded reviewer workbooks derived from the final row-level OncoTagger output. Reviewers used only Article Title, Abstract, Author Keywords, and Keywords Plus; full texts and online searches were not used. The primary validation set was a 400-record proportional year-stratified random sample from the final 20,766-record corpus, with 14 records from 2019, 26 from 2020, 40 from 2021, 59 from 2022, 68 from 2023, 77 from 2024, and 116 from 2025. Two medically trained reviewers independently assigned one dominant primary task, all additional visible task labels, detectable cancer-site labels, and detectable AI-family labels while blinded to pipeline outputs. Disagreements were resolved by oncology-domain adjudication.

Inter-reviewer agreement was calculated before adjudication. In the primary validation sample, primary-task exact agreement was 64.8% (Cohen's kappa 0.464), all-task micro-F1 was 0.814, cancer-site micro-F1 was 0.921, and AI-family micro-F1 was 0.917. After consensus adjudication, pipeline-versus-consensus primary-task exact agreement was 68.0% (Cohen's kappa 0.508), all-task micro-F1 was 0.723, cancer-site micro-F1 was 0.907, and AI-family micro-F1 was 0.886.

Additional enriched samples were analyzed separately for error characterization and were not pooled with the proportional sample for corpus-level estimates. In the complete 300-record task-unassigned census, manual consensus identified a primary task in 250 records and left 50 unassigned from abstract-level metadata. In the 100-record cancer-site no-detected challenge sample, 8 records had a manually detectable specific cancer site. In the 100-record AI-family no-detected challenge sample, 5 records had a manually detectable AI-family label.

Recommended Supplementary Data package
--------------------------------------
Supplementary Data 1. Aggregate analysis workbook derived from the synchronized 20,766-article corpus.
- Source file: `data/supplementary material/filtered_dataset_binary_classification_analysis.xlsx`
- Companion index: `visualizations/outputs/data/supplementary_analysis_workbook_sheet_index.csv`

Supplementary Data 2. Curated dictionary and search-strategy package.
- Source files: `sources/*.csv`, `sources/*.tsv`, and the exact Web of Science search-query text file included in the prepared archive.
- Companion manifest: `visualizations/outputs/data/supplementary_dictionary_manifest.csv`
- Prepared archive: `data/supplementary material/Supplementary_Data_2_curated_search_and_dictionaries.zip`

Supplementary Data 3. Population-normalized corresponding-author country output.
- Source file: `data/supplementary material/article to population ratio.xlsx`

Supplementary Data 4. Candidate translational-signal subset workbook.
- Source file: `data/supplementary material/supplementary_translational_subset.xlsx`

Supplementary Data 5. Primary 400-article ordinal validation tables.
- Source file: `data/manual validation/primary_validation_400_analysis_tables.xlsx`

Supplementary Data 6. Secondary 200-article metric-detection audit tables.
- Source file: `data/manual validation/secondary_detection_audit_analysis_tables.xlsx`

Supplementary Data 7. Taxonomy validation analysis tables.
- Source files: `data/manual validation/taxonomy_validation/taxonomy_validation_summary_tables.xlsx`, `data/manual validation/taxonomy_validation/taxonomy_validation_final_metrics_public.json`, `data/manual validation/taxonomy_validation/taxonomy_validation_row_level_labels_redacted.csv`, and `data/manual validation/taxonomy_validation/tables_csv/*.csv`

Supplementary Data 8. Derived supplementary trend and reported-performance summary tables.
- Source files: `visualizations/outputs/data/supplementary_*.csv` and `visualizations/outputs/data/supplementary_key_insights.json`
- Prepared archive: `data/supplementary material/Supplementary_Data_8_derived_supplementary_trend_tables.zip`

Recommended Supplementary Tables
--------------------------------
Supplementary Table 1. Source data for Figure 3 top corpus descriptors.
Supplementary Table 2. Curated dictionary manifest.
Supplementary Table 3. Aggregate analysis workbook sheet index.
Supplementary Table 4. AI model-family temporal trends.
Supplementary Table 5. AI class temporal trends.
Supplementary Table 6. Cancer-site temporal trends.
Supplementary Table 7. Reported weighted-performance categories by AI model family.
Supplementary Table 8. Reported weighted-performance categories by AI class.
Supplementary Table 9. Reported weighted-performance categories by cancer site.
Supplementary Table 10. Early-versus-late reported weighted-performance summary for leading AI model families.
Supplementary Table 11. Manual taxonomy validation completion, agreement, and pipeline-versus-consensus metrics.
Supplementary Table 12. Author-defined ordinal proxy thresholds for reported performance metrics.
