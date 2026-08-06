# Local data workspace

The directories in this folder are intentionally empty in the public repository.

- `raw/` receives licensed user-supplied WoSCC export batches and the merged workbook.
- `processed/` receives the deduplicated and year-restricted workbook.
- `filtered/` receives eligibility-filter outputs and decision traces.
- `results/` receives article-level annotations and aggregate analysis outputs.
- `supplementary material/` may receive locally generated article-supporting outputs.

All contents of these directories are ignored by Git. Do not commit Web of Science exports or row-level derivatives containing article metadata or abstracts.
