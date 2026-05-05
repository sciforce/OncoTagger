# Supplementary Material Artifacts

This folder contains the article-facing supplementary package for the current
OncoTagger submission snapshot. It is intended to give editors and reviewers
the minimum derived material needed to verify the manuscript's aggregate claims
without redistributing licensed raw Web of Science Core Collection exports.

## Core Files

- `Supplementary_Information.md`
  Single-file Supplementary Information source. It contains the exact search
  strategy, dictionary overview, aggregate workbook index summary,
  author-defined ordinal proxy threshold explanation, supplementary trend notes,
  and manual-validation summary.
- `filtered_dataset_binary_classification_analysis.xlsx`
  Figure-ready aggregate workbook derived from the synchronized 20,766-article
  corpus. This is the main workbook underlying the article figures and
  supplementary trend/cross-tab analyses.
- `article to population ratio.xlsx`
  Population-normalized corresponding-author country output workbook.
- `supplementary_translational_subset.xlsx`
  Rule-based candidate translational-signal subset workbook.
- `current_full_pipeline_summary.json`
  High-level synchronized pipeline counts and distributions.
- `article_to_population_ratio_summary.json`
  Summary statistics for the population-normalized country analysis.
- `translational_subset_summary.json`
  Summary statistics for the translational subset derivation.

## Upload-Ready Archives

- `Supplementary_Data_2_curated_search_and_dictionaries.zip`
  Curated search strategy, eligibility dictionaries, descriptor taxonomies,
  metric dictionaries, metric thresholds, task-specific metric priorities, and
  harmonization files.
- `Supplementary_Data_8_derived_supplementary_trend_tables.zip`
  Derived supplementary temporal and reported-performance summary tables.

## Related Visible Validation Material

Manual validation tables are stored separately under:

- `data/manual validation/`
- `data/manual validation/taxonomy_validation/`

These files include the metric-extraction validation tables, secondary
metric-detection audit tables, and public taxonomy-validation summaries.

## Notes

- Raw WoSCC exports are not included because they are subject to Clarivate
  licensing.
- These supplementary files contain derived annotations, aggregate counts,
  dictionary inputs, audit summaries, and validation summaries rather than raw
  licensed exports.
- The final repository DOI or immutable public release identifier should be
  inserted in the manuscript after the manual public release has been created.
