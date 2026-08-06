# Manuscript-locked task dictionary

## Released dictionary

`sources/task_keywords.csv` is the manuscript-locked task dictionary used to produce the reported 20,766-record corpus annotations. `sources/task_priority.csv` defines the priority used when more than one task is matched.

The released dictionary is intentionally unchanged from the manuscript analysis. This preserves exact reproducibility of the reported pipeline-derived task counts.

## Known coverage gap

The pipeline left 300 corpus records without a task assignment. Complete manual consensus assigned:

| Manual primary task | Records |
|---|---:|
| Classification | 83 |
| NLP | 43 |
| Prognosis | 41 |
| Auxiliary | 41 |
| Genomic | 24 |
| Integration | 9 |
| Synthesis | 6 |
| Segmentation | 3 |
| Remained unassigned | 50 |

The concentration in classification, NLP, prognosis, and auxiliary tasks demonstrates a systematic dictionary-coverage limitation.

The locked rules also contain overlapping vocabulary. For example, a generic higher-priority term such as `recognition` can compete with a more specific NLP phrase. This behavior is retained because changing rule precedence would alter the manuscript-linked annotation layer.

## Why the dictionary was not tuned on these records

The 300-record census is the dataset that revealed the gap. Adding phrases from these records and then reporting performance on the same records would measure development-set or resubstitution performance, not independent validation. The manuscript-locked dictionary therefore remains the reproducibility default.

The public release instead provides:

- redacted consensus labels for all 300 records;
- aggregate baseline and corrected-stratum summaries;
- a script reproducing the bounded sensitivity analysis;
- an explicit limitation in the repository documentation.

## Future dictionary update protocol

A future production update should:

1. freeze this dictionary as the manuscript version;
2. document every added or changed rule;
3. treat the 300 records as development data only;
4. rerun the full corpus with a versioned candidate dictionary;
5. draw a new blinded sample not used for rule development;
6. report post-update performance separately.

Until that protocol is completed, the repository must not describe OncoTagger as a validated article-level task classifier.
