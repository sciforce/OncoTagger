import logging
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

try:
    import pycountry  # optional, lightweight, already used in the legacy script
except Exception:  # pragma: no cover
    pycountry = None


DEFAULT_TASK_COLUMNS = [
    'classification',
    'segmentation',
    'prognosis',
    'synthesis',
    'integration',
    'nlp',
    'genomic',
    'auxiliary',
]

DEFAULT_SOURCE_TITLE_CANDIDATES = [
    'Source Title',
    'source_title',
    'source title',
    'SO',
]

DEFAULT_REPRINT_ADDRESS_CANDIDATES = [
    'Reprint Addresses',
    'reprint_addresses',
    'reprint addresses',
]

DEFAULT_YEAR_CANDIDATES = [
    'Publication Year',
    'publication_year',
    'publication year',
    'PY',
]

DEFAULT_NO_METRICS_CANDIDATES = [
    'no_metrics_reported',
    'no metrics reported',
    'No metrics reported',
    'no_metrics',
]

DEFAULT_COMPOSITE_SOURCE_CANDIDATES = [
    'composite_source',
    'Composite Source',
    'composite source',
]

DEFAULT_PRIMARY_TASK_CANDIDATES = [
    'primary_task',
    'Primary Task',
    'primary task',
]

DEFAULT_CANCER_START_AFTER = 'weighted_category'
DEFAULT_CANCER_END_BEFORE = 'cancer_detected_in'
DEFAULT_AI_START_AFTER = 'cancer_detected_in'
DEFAULT_AI_END_BEFORE = 'ai_detected_in'

DEFAULT_AI_FAMILY_MAP_CANDIDATES = [
    'ai_family_map.csv',
    'ai_family_map.tsv',
]

DEFAULT_AI_ONTOLOGY_CANDIDATES = [
    'onco-AI ontology - hierarchy.csv',
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def configure_logging() -> None:
    logging.basicConfig(
        filename='app.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def first_existing_path(*candidates: Path) -> Path | None:
    for path in candidates:
        if path and path.exists():
            return path
    return None


def build_candidate_paths(file_names: list[str], *base_dirs: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for base_dir in base_dirs:
        if not base_dir:
            continue
        for file_name in file_names:
            for candidate in (base_dir / file_name, base_dir / 'sources' / file_name):
                key = str(candidate.resolve()) if candidate.exists() else str(candidate)
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)
    return candidates


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def safe_sheet_name(name: str) -> str:
    cleaned = re.sub(r'[\\/*?:\[\]]', '_', str(name)).strip()
    return cleaned[:31] if cleaned else 'Sheet1'


def clean_text(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    text = re.sub(r'\s+', ' ', text)
    return text or None


class ArticleAnalyzer:
    def __init__(
        self,
        file_path: str | Path,
        min_source_n: int = 5,
        top_n_source_titles: int = 20,
        top_n_cancers: int = 10,
    ):
        configure_logging()

        self.file_path = Path(file_path)
        self.df = pd.read_excel(self.file_path)
        self.min_source_n = int(min_source_n)
        self.top_n_source_titles = int(top_n_source_titles)
        self.top_n_cancers = int(top_n_cancers)

        self.country_synonyms = self._load_country_synonyms()

        self.reprint_address_col = first_existing_column(self.df, DEFAULT_REPRINT_ADDRESS_CANDIDATES)
        self.source_title_col = first_existing_column(self.df, DEFAULT_SOURCE_TITLE_CANDIDATES)
        self.year_col = first_existing_column(self.df, DEFAULT_YEAR_CANDIDATES)
        self.no_metrics_col = first_existing_column(self.df, DEFAULT_NO_METRICS_CANDIDATES)
        self.composite_source_col = first_existing_column(self.df, DEFAULT_COMPOSITE_SOURCE_CANDIDATES)
        self.primary_task_col = first_existing_column(self.df, DEFAULT_PRIMARY_TASK_CANDIDATES)

        self.metric_order = ['very high', 'high', 'medium', 'low', 'very low', 'no metrics reported']

        self.cancer_columns = self._infer_between_markers(
            start_after=DEFAULT_CANCER_START_AFTER,
            end_before=DEFAULT_CANCER_END_BEFORE,
        )
        self.ai_columns = self._infer_between_markers(
            start_after=DEFAULT_AI_START_AFTER,
            end_before=DEFAULT_AI_END_BEFORE,
        )
        self.task_columns = [c for c in DEFAULT_TASK_COLUMNS if c in self.df.columns]

        self.ai_ontology_map_df = self._load_ai_ontology_map()
        self.ai_class_map_df = self._load_ai_class_map()
        self.ai_inferred_columns_raw = list(self.ai_columns)
        self.ai_inferred_extra_columns: list[str] = []
        self.class_to_families: dict[str, list[str]] = {}
        self.class_columns: list[str] = []
        self._resolve_ai_columns_against_taxonomy()
        self._add_ai_class_columns()

        self.df['number_of_cancer_types'] = self.df.get('number_of_cancer_types', 0)
        self.df['how_many_cancer_studied'] = self.df.get('how_many_cancer_studied', 'cancer type is not specified')

        for col in ['composite_metric', 'weighted_category', 'roc-auc', 'accuracy']:
            if col in self.df.columns:
                self.df[col] = self._normalize_metric_bucket_series(self.df[col])

        self._add_analysis_columns()

    # ------------------------------------------------------------------
    # Column inference and normalization
    # ------------------------------------------------------------------
    def _infer_between_markers(self, start_after: str, end_before: str) -> list[str]:
        if start_after not in self.df.columns or end_before not in self.df.columns:
            return []

        cols = list(self.df.columns)
        start_idx = cols.index(start_after) + 1
        end_idx = cols.index(end_before)
        inferred = cols[start_idx:end_idx]

        # Restrict to binary-like columns only.
        clean = []
        for col in inferred:
            series = self.df[col].dropna()
            if series.empty:
                clean.append(col)
                continue
            unique_values = set(series.astype(str).str.strip().unique())
            if unique_values.issubset({'0', '1', '0.0', '1.0', 'True', 'False', 'true', 'false'}):
                clean.append(col)
                continue
            # Many of these columns are numeric already.
            try:
                numeric_unique = set(pd.to_numeric(series, errors='coerce').dropna().astype(int).astype(str).unique())
                if numeric_unique.issubset({'0', '1'}):
                    clean.append(col)
            except Exception:
                continue
        return clean

    def _load_country_synonyms(self) -> dict[str, str]:
        synonym_path = first_existing_path(
            PROJECT_ROOT / 'sources' / 'country_synonyms.csv',
            SCRIPT_DIR / 'sources' / 'country_synonyms.csv',
            SCRIPT_DIR / 'country_synonyms.csv',
            Path('/mnt/data/country_synonyms.csv'),
        )
        if synonym_path is None:
            logging.warning('country_synonyms.csv not found; country normalization will be limited.')
            return {}

        df_syn = pd.read_csv(synonym_path)
        raw_col = 'raw' if 'raw' in df_syn.columns else df_syn.columns[0]
        norm_col = 'normalized' if 'normalized' in df_syn.columns else df_syn.columns[1]

        mapping = {}
        for _, row in df_syn.iterrows():
            raw = clean_text(row.get(raw_col))
            normalized = clean_text(row.get(norm_col))
            if raw and normalized:
                mapping[raw.casefold()] = normalized
        return mapping

    def _normalize_metric_bucket_series(self, series: pd.Series) -> pd.Series:
        return (
            series.fillna('no metrics reported')
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({'unknown': 'no metrics reported', 'nan': 'no metrics reported', 'none': 'no metrics reported'})
        )

    def _load_ai_ontology_map(self) -> pd.DataFrame:
        candidate_paths = build_candidate_paths(
            DEFAULT_AI_ONTOLOGY_CANDIDATES,
            PROJECT_ROOT,
            SCRIPT_DIR,
            self.file_path.parent,
            Path.cwd(),
            Path('/mnt/data'),
        )
        ontology_path = first_existing_path(*candidate_paths)
        if ontology_path is None:
            logging.warning('AI ontology file not found; ontology-level validation will be skipped.')
            return pd.DataFrame(columns=['subfamily_column', 'main_family'])

        try:
            df_ontology = pd.read_csv(ontology_path, sep=None, engine='python')
        except Exception as exc:
            logging.warning('Failed to read AI ontology from %s: %s', ontology_path, exc)
            return pd.DataFrame(columns=['subfamily_column', 'main_family'])

        normalized_columns = {str(c).strip().lower(): c for c in df_ontology.columns}
        level1_col = normalized_columns.get('level 1 class')
        level2_col = normalized_columns.get('level 2 columns')
        if level1_col is None or level2_col is None:
            logging.warning('AI ontology must contain "Level 1 class" and "Level 2 columns" columns.')
            return pd.DataFrame(columns=['subfamily_column', 'main_family'])

        rows = []
        for _, row in df_ontology.iterrows():
            main_family = clean_text(row.get(level1_col))
            raw_level2 = clean_text(row.get(level2_col))
            if not main_family or not raw_level2:
                continue
            for subfamily in [clean_text(x) for x in str(raw_level2).split(';')]:
                if subfamily:
                    rows.append({
                        'subfamily_column': subfamily,
                        'main_family': main_family,
                    })

        out = pd.DataFrame(rows).drop_duplicates()
        return out.reset_index(drop=True)

    def _load_ai_class_map(self) -> pd.DataFrame:
        candidate_paths = build_candidate_paths(
            DEFAULT_AI_FAMILY_MAP_CANDIDATES,
            PROJECT_ROOT,
            SCRIPT_DIR,
            self.file_path.parent,
            Path.cwd(),
            Path('/mnt/data'),
        )
        class_map_path = first_existing_path(*candidate_paths)
        if class_map_path is None:
            raise FileNotFoundError('Could not locate ai_family_map.csv')

        try:
            df_map = pd.read_csv(class_map_path, sep=None, engine='python')
        except Exception as exc:
            raise ValueError(f'Failed to read ai_family_map.csv from {class_map_path}: {exc}') from exc

        normalized_columns = {str(c).strip().lower(): c for c in df_map.columns}
        subfamily_col = normalized_columns.get('subfamily_column')
        main_family_col = normalized_columns.get('main_family')
        subgroup_col = normalized_columns.get('subgroup')

        if subfamily_col is None or main_family_col is None:
            raise ValueError('ai_family_map.csv must contain subfamily_column and main_family columns.')

        out = pd.DataFrame({
            'subfamily_column': df_map[subfamily_col].map(clean_text),
            'main_family': df_map[main_family_col].map(clean_text),
            'subgroup': df_map[subgroup_col].map(clean_text) if subgroup_col else None,
        })
        out = out.dropna(subset=['subfamily_column', 'main_family']).drop_duplicates().reset_index(drop=True)
        if 'subgroup' not in out.columns:
            out['subgroup'] = None
        return out

    def _resolve_ai_columns_against_taxonomy(self) -> None:
        """Resolve true AI family columns against the ontology/map and ignore helper binary columns.

        The dataset may contain additional binary helper columns between the marker columns,
        for example cancer_hard_detected_in or cancer_match_level. Those are not AI family
        columns and must not participate in AI family/class aggregation.
        """
        canonical_families = set()
        if not self.ai_class_map_df.empty:
            canonical_families.update(self.ai_class_map_df['subfamily_column'].dropna().astype(str).tolist())
        if not self.ai_ontology_map_df.empty:
            canonical_families.update(self.ai_ontology_map_df['subfamily_column'].dropna().astype(str).tolist())

        if not canonical_families:
            logging.warning('No canonical AI family names were loaded from ai_family_map.csv / ontology; using raw inferred AI columns.')
            return

        raw_inferred = list(self.ai_columns)
        raw_inferred_set = set(raw_inferred)
        all_dataset_canonical = [c for c in self.df.columns if c in canonical_families]

        extras = sorted(raw_inferred_set - canonical_families)
        missing_from_marker_slice = [c for c in all_dataset_canonical if c not in raw_inferred_set]

        if extras:
            self.ai_inferred_extra_columns = extras
            logging.warning(
                'Ignoring non-canonical binary columns from AI marker slice: %s',
                extras,
            )

        if missing_from_marker_slice:
            logging.warning(
                'Canonical AI family columns found outside the marker slice; they will still be used: %s',
                missing_from_marker_slice,
            )

        self.ai_columns = all_dataset_canonical

    def _validate_ai_class_map(self) -> pd.DataFrame:
        if self.ai_class_map_df.empty:
            raise ValueError('ai_family_map.csv is empty after normalization.')
        if not self.ai_columns:
            raise ValueError('No canonical AI family columns were resolved from the dataset after matching against ai_family_map.csv / ontology; cannot aggregate AI classes.')

        df_map = self.ai_class_map_df.copy()

        duplicates = (
            df_map[df_map['subfamily_column'].duplicated(keep=False)]
            .sort_values(['subfamily_column', 'main_family'])
        )
        if not duplicates.empty:
            dup_pairs = duplicates[['subfamily_column', 'main_family']].to_dict('records')
            raise ValueError(f'ai_family_map.csv contains duplicate subfamily_column entries: {dup_pairs}')

        dataset_family_set = set(self.ai_columns)
        map_family_set = set(df_map['subfamily_column'])

        missing_in_dataset = sorted(map_family_set - dataset_family_set)
        missing_in_map = sorted(dataset_family_set - map_family_set)
        if missing_in_dataset or missing_in_map:
            problems = []
            if missing_in_dataset:
                problems.append(f'families present in ai_family_map.csv but absent from dataset AI columns: {missing_in_dataset}')
            if missing_in_map:
                problems.append(f'dataset AI columns missing from ai_family_map.csv: {missing_in_map}')
            raise ValueError('; '.join(problems))

        if not self.ai_ontology_map_df.empty:
            ontology_family_set = set(self.ai_ontology_map_df['subfamily_column'])
            ontology_class_set = set(self.ai_ontology_map_df['main_family'])

            missing_in_ontology = sorted(map_family_set - ontology_family_set)
            missing_in_map_vs_ontology = sorted(ontology_family_set - map_family_set)
            invalid_classes = sorted(set(df_map['main_family']) - ontology_class_set)
            if missing_in_ontology or missing_in_map_vs_ontology or invalid_classes:
                problems = []
                if missing_in_ontology:
                    problems.append(f'families present in ai_family_map.csv but absent from ontology: {missing_in_ontology}')
                if missing_in_map_vs_ontology:
                    problems.append(f'ontology families missing from ai_family_map.csv: {missing_in_map_vs_ontology}')
                if invalid_classes:
                    problems.append(f'non-canonical main_family values absent from ontology: {invalid_classes}')
                raise ValueError('; '.join(problems))

            expected_lookup = dict(self.ai_ontology_map_df[['subfamily_column', 'main_family']].itertuples(index=False, name=None))
            mismatched_rows = []
            for _, row in df_map.iterrows():
                expected_class = expected_lookup.get(row['subfamily_column'])
                if expected_class != row['main_family']:
                    mismatched_rows.append({
                        'subfamily_column': row['subfamily_column'],
                        'main_family': row['main_family'],
                        'expected_main_family': expected_class,
                    })
            if mismatched_rows:
                raise ValueError(f'ai_family_map.csv contains ontology-inconsistent mappings: {mismatched_rows}')

        subgroup_mismatch = df_map[
            df_map['subgroup'].notna() & (df_map['subgroup'] != '') & (df_map['subgroup'] != df_map['main_family'])
        ]
        if not subgroup_mismatch.empty:
            logging.warning(
                'Some subgroup values differ from main_family. They will be preserved for reporting only: %s',
                subgroup_mismatch[['subfamily_column', 'main_family', 'subgroup']].to_dict('records')
            )

        ontology_order = {}
        if not self.ai_ontology_map_df.empty:
            ontology_order = {name: i for i, name in enumerate(self.ai_ontology_map_df['main_family'].drop_duplicates().tolist())}
            family_order = {name: i for i, name in enumerate(self.ai_ontology_map_df['subfamily_column'].tolist())}
            df_map['__class_order'] = df_map['main_family'].map(ontology_order)
            df_map['__family_order'] = df_map['subfamily_column'].map(family_order)
            df_map = df_map.sort_values(['__class_order', '__family_order', 'subfamily_column'])
            df_map = df_map.drop(columns=['__class_order', '__family_order'])
        else:
            df_map = df_map.sort_values(['main_family', 'subfamily_column'])

        return df_map.reset_index(drop=True)

    def _add_ai_class_columns(self) -> None:
        self.ai_class_map_df = self._validate_ai_class_map()

        for main_family, group in self.ai_class_map_df.groupby('main_family', sort=False):
            subfamilies = group['subfamily_column'].tolist()
            class_series = (
                self.df[subfamilies]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .clip(lower=0, upper=1)
                .max(axis=1)
                .astype(int)
            )
            self.df[main_family] = class_series
            self.class_columns.append(main_family)
            self.class_to_families[main_family] = subfamilies

    def _build_ai_class_breakdown(self) -> pd.DataFrame:
        if not self.class_columns or self.ai_class_map_df.empty:
            return pd.DataFrame(columns=[
                'AI Class', 'AI Family', 'Article Count',
                'Share of All Articles', 'Share Within Class Articles'
            ])

        rows = []
        for _, row in self.ai_class_map_df.iterrows():
            subfamily = row['subfamily_column']
            ai_class = row['main_family']
            if subfamily not in self.df.columns or ai_class not in self.df.columns:
                continue
            sub_count = int(pd.to_numeric(self.df[subfamily], errors='coerce').fillna(0).sum())
            class_count = int(pd.to_numeric(self.df[ai_class], errors='coerce').fillna(0).sum())
            rows.append({
                'AI Class': ai_class,
                'AI Family': subfamily,
                'Article Count': sub_count,
                'Share of All Articles': sub_count / len(self.df) if len(self.df) else 0,
                'Share Within Class Articles': sub_count / class_count if class_count else 0,
            })
        return pd.DataFrame(rows).sort_values(
            ['AI Class', 'Article Count', 'AI Family'],
            ascending=[True, False, True],
        )

    def write_ai_class_mapping(self, writer) -> None:
        if self.ai_class_map_df.empty:
            return
        map_out = self.ai_class_map_df.rename(columns={
            'subfamily_column': 'AI Family',
            'main_family': 'AI Class',
            'subgroup': 'Subgroup',
        })
        self._write_table(writer, map_out, 'AI Class Map')

        breakdown = self._build_ai_class_breakdown()
        if not breakdown.empty:
            self._write_table(writer, breakdown, 'AI Class Breakdown')

    def write_ai_family_mapping(self, writer) -> None:
        self.write_ai_class_mapping(writer)

    def _parse_bool_like(self, value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return 1 if int(value) != 0 else 0
        if isinstance(value, float):
            if np.isnan(value):
                return pd.NA
            return 1 if float(value) != 0 else 0

        text = str(value).strip().lower()
        if text in {'1', 'true', 'yes', 'y', 'reported no metrics', 'no metrics reported'}:
            return 1
        if text in {'0', 'false', 'no', 'n', ''}:
            return 0
        if text in {'nan', 'none', 'null'}:
            return pd.NA
        return pd.NA

    def _normalize_composite_source(self, value) -> str | None:
        text = clean_text(value)
        return text.lower() if text else None

    def _extract_reprint_country_raw(self, address) -> str | None:
        text = clean_text(address)
        if not text:
            return None

        text = re.sub(r'\[[^\]]+\]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        segments = [seg.strip() for seg in re.split(r'\s*;\s*', text) if seg.strip()]
        if not segments:
            return None

        marker_re = re.compile(
            r'\((?:author.*correspond|corresponding author|автор.*корреспонд)',
            re.IGNORECASE
        )

        preferred = None
        for seg in segments:
            if marker_re.search(seg):
                preferred = seg
                break

        if preferred is None:
            preferred = max(segments, key=lambda s: (s.count(','), len(s)))

        preferred = re.sub(r'^.*?\)\s*,\s*', '', preferred)

        parts = [p.strip() for p in preferred.split(',') if p.strip()]
        if len(parts) >= 4:
            first = parts[0]
            second = parts[1].replace('.', '').replace(' ', '')
            if re.fullmatch(r"[A-Z][A-Za-z'\- ]+", first) and re.fullmatch(r'[A-Z]{1,4}', second.upper()):
                parts = parts[2:]

        candidate = parts[-1] if parts else preferred
        candidate = re.sub(r'\b(?:email|e-mail)\b.*$', '', candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(' .,:;')

        if not candidate or len(candidate) <= 2:
            m = re.search(
                r'\b(?:Peoples R China|China|Japan|India|Italy|Portugal|Israel|South Korea|Korea|'
                r'USA|United States|United Kingdom|England|Scotland|Wales|Turkey|Turkiye|Türkiye|'
                r'Germany|France|Spain|Canada|Australia|Brazil|Mexico|Poland|Romania|Norway|'
                r'Thailand|Ethiopia|Finland|Denmark|Greece|Ireland|Singapore|Austria|Iraq|'
                r'Viet Nam|Cyprus|Sudan|Bahrain|Cameroon|Malaysia)\b\.?$',
                preferred,
                re.IGNORECASE
            )
            if m:
                candidate = m.group(0).strip(' .,:;')

        return candidate or None

    def _normalize_country(self, raw_country) -> str | None:
        text = clean_text(raw_country)
        if not text:
            return None

        candidates = [text]
        if ' ' in text:
            candidates.append(text.split()[-1])

        for candidate in candidates:
            key = candidate.casefold()
            if key in self.country_synonyms:
                return self.country_synonyms[key]

            if pycountry is not None:
                try:
                    return pycountry.countries.lookup(candidate).name
                except LookupError:
                    pass

        return text

    def _build_source_title_mapping(self) -> dict[str, str]:
        if not self.source_title_col:
            return {}

        raw_titles = self.df[self.source_title_col].map(clean_text)
        counter_by_key = {}
        for title in raw_titles.dropna():
            key = title.casefold()
            counter_by_key.setdefault(key, Counter())
            counter_by_key[key][title] += 1

        mapping = {}
        for key, counter in counter_by_key.items():
            # Preserve the most common original surface form after whitespace cleanup.
            mapping[key] = counter.most_common(1)[0][0]
        return mapping

    def _safe_primary_task(self, value) -> str:
        text = clean_text(value)
        return text.lower() if text else 'task not specified'

    def _safe_category(self, value, missing_label: str = 'not available') -> str:
        text = clean_text(value)
        return text if text else missing_label

    def _add_analysis_columns(self) -> None:
        if self.reprint_address_col:
            self.df['reprint_country'] = self.df[self.reprint_address_col].map(self._extract_reprint_country_raw).map(self._normalize_country)
        else:
            self.df['reprint_country'] = pd.Series([None] * len(self.df), index=self.df.index)

        self.df['reprint_country_status'] = np.where(
            self.df['reprint_country'].notna(),
            'country available from reprint address',
            'country unavailable from reprint address',
        )

        if self.no_metrics_col:
            self.df['no_metrics_reported_bool'] = self.df[self.no_metrics_col].map(self._parse_bool_like).astype('Int64')
        else:
            self.df['no_metrics_reported_bool'] = pd.Series([pd.NA] * len(self.df), dtype='Int64')

        if self.composite_source_col:
            self.df['composite_source_clean'] = self.df[self.composite_source_col].map(self._normalize_composite_source)
        else:
            self.df['composite_source_clean'] = pd.Series([None] * len(self.df), index=self.df.index)

        if self.source_title_col:
            title_map = self._build_source_title_mapping()
            self.df['source_title_clean'] = self.df[self.source_title_col].map(clean_text)
            self.df['source_title_clean'] = self.df['source_title_clean'].map(
                lambda x: title_map.get(x.casefold(), x) if isinstance(x, str) else None
            )
        else:
            self.df['source_title_clean'] = pd.Series([None] * len(self.df), index=self.df.index)

        if self.primary_task_col:
            self.df['primary_task_clean'] = self.df[self.primary_task_col].map(self._safe_primary_task)
        else:
            self.df['primary_task_clean'] = 'task not specified'

        if self.year_col and self.year_col in self.df.columns:
            self.df['publication_year_clean'] = pd.to_numeric(self.df[self.year_col], errors='coerce').astype('Int64')
        else:
            self.df['publication_year_clean'] = pd.Series([pd.NA] * len(self.df), dtype='Int64')

    # ------------------------------------------------------------------
    # Legacy summaries preserved
    # ------------------------------------------------------------------
    def count_cancer_types(self):
        if not self.cancer_columns:
            return
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc='Counting cancer types...'):
            count = int(pd.to_numeric(row[self.cancer_columns], errors='coerce').fillna(0).sum())
            self.df.at[idx, 'number_of_cancer_types'] = count

            if count > 1:
                self.df.at[idx, 'how_many_cancer_studied'] = 'various cancers'
            elif count == 1:
                cancer = next((c for c in self.cancer_columns if pd.to_numeric(row.get(c), errors='coerce') == 1), None)
                if cancer:
                    self.df.at[idx, 'how_many_cancer_studied'] = f'just one cancer - {cancer}'
            else:
                self.df.at[idx, 'how_many_cancer_studied'] = 'not specified'

    def count_ai_models(self):
        if not self.ai_columns:
            self.df['number_of_ai_models'] = 0
            return
        self.df['number_of_ai_models'] = self.df[self.ai_columns].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    def count_ai_classes(self):
        if not self.class_columns:
            self.df['number_of_ai_classes'] = 0
            return
        self.df['number_of_ai_classes'] = self.df[self.class_columns].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    def count_ai_families(self):
        self.count_ai_classes()

    def count_task_categories(self, writer):
        if not self.task_columns:
            return
        df_tasks = self.df[self.task_columns].apply(pd.to_numeric, errors='coerce').fillna(0).sum().sort_values(ascending=False).reset_index()
        df_tasks.columns = ['Task Category', 'Count']
        df_tasks.to_excel(writer, sheet_name='Task Categories Frequency', index=False)

    def count_frequency(self, columns, sheet_name, col_header, writer):
        if not columns:
            return
        df_freq = self.df[columns].apply(pd.to_numeric, errors='coerce').fillna(0).sum().sort_values(ascending=False).reset_index()
        df_freq.columns = [col_header, 'Count']
        df_freq.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)

    def count_cancer_type_distribution(self, writer):
        df_dist = self.df['how_many_cancer_studied'].value_counts(dropna=False).reset_index()
        df_dist.columns = ['Cancer Study Type', 'Count']
        df_dist.to_excel(writer, sheet_name='Cancer Type Distribution', index=False)

    def count_ordered_metric_totals(self, col_name, sheet_name, writer):
        if col_name not in self.df.columns:
            return
        vc = self.df[col_name].value_counts(dropna=False)
        df_ordered = vc.reindex(self.metric_order).fillna(0).astype(int).reset_index()
        df_ordered.columns = [sheet_name, 'Count']
        df_ordered.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)

    def count_tasks_by_year(self, writer):
        if not self.task_columns:
            return
        df_ty = self.df.groupby('publication_year_clean')[self.task_columns].sum(min_count=1)
        df_ty.index.name = 'Publication Year'
        df_ty.to_excel(writer, sheet_name='Tasks by Year')

    def count_by_years(self, columns, sheet_name, writer):
        if not columns:
            return
        df_year = self.df.groupby('publication_year_clean')[columns].sum(min_count=1)
        df_year.index.name = 'Publication Year'
        df_year.to_excel(writer, sheet_name=safe_sheet_name(sheet_name))

    def count_metric_by_year(self, metric, writer):
        if metric not in self.df.columns:
            return
        df_my = self.df.groupby(['publication_year_clean', metric]).size().unstack(fill_value=0)
        existing_cols = [c for c in self.metric_order if c in df_my.columns]
        if existing_cols:
            df_my = df_my[existing_cols]
        sheet = safe_sheet_name(f'{metric.replace("_", " ").title()} by Year')
        df_my.index.name = 'Publication Year'
        df_my.to_excel(writer, sheet_name=sheet)

    def count_metric_by_task(self, metric, writer):
        if metric not in self.df.columns or not self.task_columns:
            return
        rows = []
        for task in self.task_columns:
            df_t = self.df[self.df[task] == 1]
            vc = df_t[metric].value_counts(dropna=False)
            rows.append(vc.rename(task))
        if not rows:
            return
        df_mt = pd.concat(rows, axis=1).fillna(0).astype(int)
        existing_idx = [idx for idx in self.metric_order if idx in df_mt.index]
        if existing_idx:
            df_mt = df_mt.reindex(existing_idx).fillna(0).astype(int)
        sheet = safe_sheet_name(f'{metric.replace("_", " ").title()} by Task')
        df_mt.index.name = metric.replace('_', ' ').title()
        df_mt.to_excel(writer, sheet_name=sheet)

    def crosstab_tasks_vs(self, bins, writer, sheet_name):
        if not self.task_columns or not bins:
            return
        ct = pd.DataFrame(index=self.task_columns, columns=bins)
        for task in self.task_columns:
            df_t = self.df[self.df[task] == 1]
            ct.loc[task] = df_t[bins].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
        ct = ct.loc[ct.index.notnull()].fillna(0).astype(int)
        ct.index.name = 'Task Category'
        ct.to_excel(writer, sheet_name=safe_sheet_name(sheet_name))

    def crosstab_metric_vs(self, metric, bins, writer, sheet_name):
        if metric not in self.df.columns or not bins:
            return
        cats = [c for c in self.metric_order if c in self.df[metric].dropna().unique()]
        ct = pd.DataFrame(index=cats, columns=bins)
        for category in cats:
            df_c = self.df[self.df[metric] == category]
            ct.loc[category] = df_c[bins].apply(pd.to_numeric, errors='coerce').fillna(0).sum()
        ct = ct.loc[ct.index.notnull()].fillna(0).astype(int)
        ct.index.name = metric.replace('_', ' ').title()
        ct.to_excel(writer, sheet_name=safe_sheet_name(sheet_name))

    # ------------------------------------------------------------------
    # New cautious analytic outputs
    # ------------------------------------------------------------------
    def _write_table(self, writer, df_out: pd.DataFrame, sheet_name: str) -> None:
        df_out.to_excel(writer, sheet_name=safe_sheet_name(sheet_name), index=False)

    def _value_count_with_share(self, series: pd.Series, label: str, missing_label: str = 'not available') -> pd.DataFrame:
        clean = series.map(lambda x: self._safe_category(x, missing_label=missing_label))
        vc = clean.value_counts(dropna=False)
        out = vc.rename_axis(label).reset_index(name='Count')
        out['Share_of_all_articles'] = out['Count'] / len(self.df)
        return out

    def _crosstab_count(self, row_series: pd.Series, col_series: pd.Series, row_name: str) -> pd.DataFrame:
        ct = pd.crosstab(row_series.fillna('not available'), col_series.fillna('not available'))
        ct = ct.reset_index().rename(columns={ct.columns[0]: row_name})
        return ct

    def _share_by_group(self, group_col: str, bool_col: str, group_label: str) -> pd.DataFrame:
        subset = self.df[[group_col, bool_col]].dropna(subset=[group_col])
        grouped = subset.groupby(group_col)[bool_col].agg(['size', 'sum'])
        grouped = grouped.rename(columns={'size': 'N_articles', 'sum': 'N_positive'})
        grouped['Share'] = grouped['N_positive'] / grouped['N_articles']
        grouped = grouped.reset_index().rename(columns={group_col: group_label})
        return grouped.sort_values(['Share', 'N_articles', group_label], ascending=[False, False, True])

    def count_reprint_country_outputs(self, writer):
        if self.reprint_address_col is None:
            logging.warning('Reprint address column not found; country summaries skipped.')
            return

        overall = self._value_count_with_share(
            self.df['reprint_country'],
            label='Reprint-address country',
            missing_label='country unavailable from reprint address',
        )
        self._write_table(writer, overall, 'Reprint Country Overall')

        missing = (
            self.df['reprint_country_status']
            .value_counts(dropna=False)
            .rename_axis('Reprint-address status')
            .reset_index(name='Count')
        )
        missing['Share_of_all_articles'] = missing['Count'] / len(self.df)
        self._write_table(writer, missing, 'Reprint Country Missing')

        by_year = pd.crosstab(
            self.df['publication_year_clean'],
            self.df['reprint_country'].fillna('country unavailable from reprint address')
        ).reset_index().rename(columns={'publication_year_clean': 'Publication Year'})
        self._write_table(writer, by_year, 'Reprint Country by Year')

        by_task = pd.crosstab(
            self.df['primary_task_clean'],
            self.df['reprint_country'].fillna('country unavailable from reprint address')
        ).reset_index().rename(columns={'primary_task_clean': 'Primary Task'})
        self._write_table(writer, by_task, 'Reprint Country by Task')

        missing_by_year = self._share_by_group('publication_year_clean', 'no_metrics_dummy_tmp', 'Publication Year') if False else None
        # Explicit missingness trend for reprint-address country.
        tmp = self.df.copy()
        tmp['reprint_country_missing_bool'] = tmp['reprint_country'].isna().astype(int)
        miss_trend = tmp.groupby('publication_year_clean')['reprint_country_missing_bool'].agg(['size', 'sum']).reset_index()
        miss_trend.columns = ['Publication Year', 'N_articles', 'N_missing_reprint_country']
        miss_trend['Missing_share'] = miss_trend['N_missing_reprint_country'] / miss_trend['N_articles']
        self._write_table(writer, miss_trend, 'Reprint Country Miss Year')

    def count_composite_source_outputs(self, writer):
        if self.composite_source_col is None:
            logging.warning('Composite source column not found; composite source summaries skipped.')
            return

        overall = self._value_count_with_share(
            self.df['composite_source_clean'],
            label='Composite source',
            missing_label='not available',
        )
        self._write_table(writer, overall, 'Composite Src Overall')

        by_year = pd.crosstab(
            self.df['publication_year_clean'],
            self.df['composite_source_clean'].fillna('not available')
        ).reset_index().rename(columns={'publication_year_clean': 'Publication Year'})
        self._write_table(writer, by_year, 'Composite Src by Year')

        by_task = pd.crosstab(
            self.df['primary_task_clean'],
            self.df['composite_source_clean'].fillna('not available')
        ).reset_index().rename(columns={'primary_task_clean': 'Primary Task'})
        self._write_table(writer, by_task, 'Composite Src by Task')

        if self.source_title_col:
            source_subset = self.df[self.df['source_title_clean'].notna()].copy()
            if not source_subset.empty:
                top_sources = (
                    source_subset['source_title_clean']
                    .value_counts()
                    .head(self.top_n_source_titles)
                    .index
                    .tolist()
                )
                ct_source = pd.crosstab(
                    source_subset.loc[source_subset['source_title_clean'].isin(top_sources), 'source_title_clean'],
                    source_subset.loc[source_subset['source_title_clean'].isin(top_sources), 'composite_source_clean'].fillna('not available')
                ).reset_index().rename(columns={'source_title_clean': 'Source Title'})
                self._write_table(writer, ct_source, 'Composite Src by Source')

        if self.cancer_columns:
            top_cancers = (
                self.df[self.cancer_columns]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .sum()
                .sort_values(ascending=False)
                .head(self.top_n_cancers)
                .index
                .tolist()
            )
            rows = []
            for cancer in top_cancers:
                df_c = self.df[self.df[cancer] == 1]
                vc = df_c['composite_source_clean'].fillna('not available').value_counts()
                for source_name, count in vc.items():
                    rows.append({'Cancer Type': cancer, 'Composite Source': source_name, 'Count': int(count)})
            if rows:
                out = pd.DataFrame(rows).sort_values(['Cancer Type', 'Count', 'Composite Source'], ascending=[True, False, True])
                self._write_table(writer, out, 'Composite Src by Cancer')

    def count_no_metrics_outputs(self, writer):
        if self.no_metrics_col is None:
            logging.warning('No-metrics column not found; no-metrics summaries skipped.')
            return

        valid = self.df[self.df['no_metrics_reported_bool'].notna()].copy()
        if valid.empty:
            logging.warning('No-metrics column present but could not be parsed reliably.')
            return

        overall = (
            valid['no_metrics_reported_bool']
            .map({1: 'no metrics reported', 0: 'at least one metric reported'})
            .value_counts(dropna=False)
            .rename_axis('Reporting status')
            .reset_index(name='Count')
        )
        overall['Share_of_parsed_articles'] = overall['Count'] / len(valid)
        self._write_table(writer, overall, 'No Metrics Overall')

        by_year = valid.groupby('publication_year_clean')['no_metrics_reported_bool'].agg(['size', 'sum']).reset_index()
        by_year.columns = ['Publication Year', 'N_articles', 'N_no_metrics_reported']
        by_year['Share_no_metrics_reported'] = by_year['N_no_metrics_reported'] / by_year['N_articles']
        self._write_table(writer, by_year, 'No Metrics by Year')

        by_task = valid.groupby('primary_task_clean')['no_metrics_reported_bool'].agg(['size', 'sum']).reset_index()
        by_task.columns = ['Primary Task', 'N_articles', 'N_no_metrics_reported']
        by_task['Share_no_metrics_reported'] = by_task['N_no_metrics_reported'] / by_task['N_articles']
        by_task = by_task.sort_values(['Share_no_metrics_reported', 'N_articles', 'Primary Task'], ascending=[False, False, True])
        self._write_table(writer, by_task, 'No Metrics by Task')

        if self.source_title_col:
            source_df = valid[valid['source_title_clean'].notna()].copy()
            if not source_df.empty:
                venue = source_df.groupby('source_title_clean')['no_metrics_reported_bool'].agg(['size', 'sum']).reset_index()
                venue.columns = ['Source Title', 'N_articles', 'N_no_metrics_reported']
                venue['Share_no_metrics_reported'] = venue['N_no_metrics_reported'] / venue['N_articles']
                venue = venue.sort_values(['Share_no_metrics_reported', 'N_articles', 'Source Title'], ascending=[False, False, True])
                self._write_table(writer, venue, 'No Metrics by Source')

                eligible = venue[venue['N_articles'] >= self.min_source_n].copy()
                if not eligible.empty:
                    high = eligible.sort_values(
                        ['Share_no_metrics_reported', 'N_articles', 'Source Title'],
                        ascending=[False, False, True]
                    )
                    low = eligible.sort_values(
                        ['Share_no_metrics_reported', 'N_articles', 'Source Title'],
                        ascending=[True, False, True]
                    )
                    self._write_table(writer, high, f'NoMetrics Src Hi N{self.min_source_n}')
                    self._write_table(writer, low, f'NoMetrics Src Lo N{self.min_source_n}')

        if self.cancer_columns:
            top_cancers = (
                self.df[self.cancer_columns]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .sum()
                .sort_values(ascending=False)
                .head(self.top_n_cancers)
                .index
                .tolist()
            )
            rows = []
            for cancer in top_cancers:
                df_c = valid[valid[cancer] == 1]
                if df_c.empty:
                    continue
                n_articles = len(df_c)
                n_no_metrics = int(df_c['no_metrics_reported_bool'].sum())
                rows.append({
                    'Cancer Type': cancer,
                    'N_articles': n_articles,
                    'N_no_metrics_reported': n_no_metrics,
                    'Share_no_metrics_reported': n_no_metrics / n_articles,
                })
            if rows:
                out = pd.DataFrame(rows).sort_values(['Share_no_metrics_reported', 'N_articles', 'Cancer Type'], ascending=[False, False, True])
                self._write_table(writer, out, 'No Metrics by Cancer')

    def count_source_title_outputs(self, writer):
        if not self.source_title_col:
            logging.warning('Source Title column not found; source-title summaries skipped.')
            return

        valid = self.df[self.df['source_title_clean'].notna()].copy()
        if valid.empty:
            return

        overall = valid['source_title_clean'].value_counts().head(self.top_n_source_titles).reset_index()
        overall.columns = ['Source Title', 'Count']
        overall['Share_of_articles_with_source_title'] = overall['Count'] / len(valid)
        self._write_table(writer, overall, 'Source Titles Overall')

        task_rows = []
        for task, df_t in valid.groupby('primary_task_clean'):
            top = df_t['source_title_clean'].value_counts().head(self.top_n_source_titles)
            for rank, (title, count) in enumerate(top.items(), start=1):
                task_rows.append({'Primary Task': task, 'Rank_within_task': rank, 'Source Title': title, 'Count': int(count)})
        if task_rows:
            task_out = pd.DataFrame(task_rows)
            self._write_table(writer, task_out, 'Source Titles by Task')

        if self.cancer_columns:
            cancer_rows = []
            top_cancers = (
                self.df[self.cancer_columns]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .sum()
                .sort_values(ascending=False)
                .head(self.top_n_cancers)
                .index
                .tolist()
            )
            for cancer in top_cancers:
                df_c = valid[valid[cancer] == 1]
                top = df_c['source_title_clean'].value_counts().head(self.top_n_source_titles)
                for rank, (title, count) in enumerate(top.items(), start=1):
                    cancer_rows.append({'Cancer Type': cancer, 'Rank_within_cancer': rank, 'Source Title': title, 'Count': int(count)})
            if cancer_rows:
                cancer_out = pd.DataFrame(cancer_rows)
                self._write_table(writer, cancer_out, 'Source Titles by Cancer')

        top_sources = valid['source_title_clean'].value_counts().head(self.top_n_source_titles).index.tolist()
        year_top = valid[valid['source_title_clean'].isin(top_sources)].copy()
        year_top = pd.crosstab(year_top['publication_year_clean'], year_top['source_title_clean']).reset_index()
        year_top = year_top.rename(columns={'publication_year_clean': 'Publication Year'})
        self._write_table(writer, year_top, 'SourceTitle Year Top')

    def count_simple_metadata_outputs(self, writer):
        for meta_col in ['cancer_detected_in', 'ai_detected_in', 'composite_source', 'all_tasks']:
            if meta_col in self.df.columns:
                meta_counts = self.df[meta_col].fillna('Unknown').value_counts().reset_index()
                meta_counts.columns = [meta_col, 'Count']
                self._write_table(writer, meta_counts, f'Meta_{meta_col}')

    def count_numeric_distribution(self, col_name: str, label: str, writer) -> None:
        if col_name not in self.df.columns:
            return
        series = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0).astype(int)
        out = series.value_counts(dropna=False).sort_index().reset_index()
        out.columns = [label, 'Count']
        self._write_table(writer, out, label)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def run_analysis(self):
        prep_steps = [
            ('Count cancer types', self.count_cancer_types),
            ('Count AI models', self.count_ai_models),
            ('Count AI classes', self.count_ai_classes),
        ]

        with tqdm(total=len(prep_steps), desc='Preparation', unit='step', dynamic_ncols=True) as prep_bar:
            for step_name, step_func in prep_steps:
                prep_bar.set_postfix_str(step_name)
                step_func()
                prep_bar.update(1)

        output_file = self.file_path.with_name(f'{self.file_path.stem}_analysis.xlsx')

        top10_cancers = (
            self.df[self.cancer_columns]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .sum()
            .nlargest(min(10, len(self.cancer_columns)))
            .index
            .tolist()
        ) if self.cancer_columns else []

        top10_models = (
            self.df[self.ai_columns]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .sum()
            .nlargest(min(10, len(self.ai_columns)))
            .index
            .tolist()
        ) if self.ai_columns else []

        top10_classes = (
            self.df[self.class_columns]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
            .sum()
            .nlargest(min(10, len(self.class_columns)))
            .index
            .tolist()
        ) if self.class_columns else []

        writer_steps = [
            ('Reprint country outputs', lambda w: self.count_reprint_country_outputs(w)),
            ('Composite source outputs', lambda w: self.count_composite_source_outputs(w)),
            ('No-metrics outputs', lambda w: self.count_no_metrics_outputs(w)),
            ('Source title outputs', lambda w: self.count_source_title_outputs(w)),
            ('Task categories', lambda w: self.count_task_categories(w)),
            ('Cancer frequency', lambda w: self.count_frequency(self.cancer_columns, 'Cancer Types Frequency', 'Cancer Type', w)),
            ('AI model frequency', lambda w: self.count_frequency(self.ai_columns, 'AI Models Frequency', 'AI Model', w)),
            ('AI class frequency', lambda w: self.count_frequency(self.class_columns, 'AI Classes Frequency', 'AI Class', w)),
            ('AI class mapping', lambda w: self.write_ai_class_mapping(w)),
            ('Number of AI models', lambda w: self.count_numeric_distribution('number_of_ai_models', 'Number of AI Models', w)),
            ('Number of AI classes', lambda w: self.count_numeric_distribution('number_of_ai_classes', 'Number of AI Classes', w)),
            ('Cancer type distribution', lambda w: self.count_cancer_type_distribution(w)),
            ('Composite total', lambda w: self.count_ordered_metric_totals('composite_metric', 'Composite Total', w)),
            ('Weighted total', lambda w: self.count_ordered_metric_totals('weighted_category', 'Weighted Total', w)),
            ('Composite by year', lambda w: self.count_metric_by_year('composite_metric', w)),
            ('Composite by task', lambda w: self.count_metric_by_task('composite_metric', w)),
            ('Weighted by year', lambda w: self.count_metric_by_year('weighted_category', w)),
            ('Weighted by task', lambda w: self.count_metric_by_task('weighted_category', w)),
            ('ROC-AUC by year', lambda w: self.count_metric_by_year('roc-auc', w)),
            ('ROC-AUC by task', lambda w: self.count_metric_by_task('roc-auc', w)),
            ('Task x Cancer', lambda w: self.crosstab_tasks_vs(self.cancer_columns, w, 'Task x Cancer')),
            ('Task x AI Models', lambda w: self.crosstab_tasks_vs(self.ai_columns, w, 'Task x AI Models')),
            ('Task x AI Classes', lambda w: self.crosstab_tasks_vs(self.class_columns, w, 'Task x AI Classes')),
            ('Composite x Cancer', lambda w: self.crosstab_metric_vs('composite_metric', self.cancer_columns, w, 'Composite x Cancer')),
            ('Composite x AI', lambda w: self.crosstab_metric_vs('composite_metric', self.ai_columns, w, 'Composite x AI')),
            ('Composite x AI Classes', lambda w: self.crosstab_metric_vs('composite_metric', self.class_columns, w, 'Composite x AI Classes')),
            ('Weighted x Cancer', lambda w: self.crosstab_metric_vs('weighted_category', self.cancer_columns, w, 'Weighted x Cancer')),
            ('Weighted x AI', lambda w: self.crosstab_metric_vs('weighted_category', self.ai_columns, w, 'Weighted x AI')),
            ('Weighted x AI Classes', lambda w: self.crosstab_metric_vs('weighted_category', self.class_columns, w, 'Weighted x AI Classes')),
            ('ROC-AUC x Cancer', lambda w: self.crosstab_metric_vs('roc-auc', self.cancer_columns, w, 'ROC-AUC x Cancer')),
            ('ROC-AUC x AI', lambda w: self.crosstab_metric_vs('roc-auc', self.ai_columns, w, 'ROC-AUC x AI')),
            ('ROC-AUC x AI Classes', lambda w: self.crosstab_metric_vs('roc-auc', self.class_columns, w, 'ROC-AUC x AI Classes')),
            ('Simple metadata outputs', lambda w: self.count_simple_metadata_outputs(w)),
            ('Tasks by year', lambda w: self.count_tasks_by_year(w)),
            ('Cancer types by year', lambda w: self.count_by_years(self.cancer_columns, 'Cancer Types by Year', w)),
            ('AI models by year', lambda w: self.count_by_years(self.ai_columns, 'AI Models by Year', w)),
            ('AI classes by year', lambda w: self.count_by_years(self.class_columns, 'AI Classes by Year', w)),
        ]

        if 'accuracy' in self.df.columns:
            writer_steps.extend([
                ('Accuracy x Cancer', lambda w: self.crosstab_metric_vs('accuracy', self.cancer_columns, w, 'Accuracy x Cancer')),
                ('Accuracy x AI', lambda w: self.crosstab_metric_vs('accuracy', self.ai_columns, w, 'Accuracy x AI')),
                ('Accuracy x AI Classes', lambda w: self.crosstab_metric_vs('accuracy', self.class_columns, w, 'Accuracy x AI Classes')),
            ])

        if top10_cancers:
            writer_steps.append(('Top-10 cancers by year', lambda w: self.count_by_years(top10_cancers, 'Top-10 Cancers by Year', w)))
        if top10_models:
            writer_steps.append(('Top-10 AI models by year', lambda w: self.count_by_years(top10_models, 'Top-10 AI Models by Year', w)))
        if top10_classes:
            writer_steps.append(('Top-10 AI classes by year', lambda w: self.count_by_years(top10_classes, 'Top-10 AI Classes by Year', w)))

        with pd.ExcelWriter(output_file) as writer:
            with tqdm(total=len(writer_steps), desc='Writing analysis workbook', unit='sheet', dynamic_ncols=True) as write_bar:
                for step_name, step_func in writer_steps:
                    write_bar.set_postfix_str(step_name)
                    step_func(writer)
                    write_bar.update(1)

        logging.info('Analysis complete. File saved: %s', output_file)
        print(f'Analysis complete. File saved: {output_file}')
        return output_file


if __name__ == '__main__':
    default_input = first_existing_path(
        PROJECT_ROOT / 'data' / 'results' / 'filtered_dataset_binary_classification.xlsx',
        PROJECT_ROOT / 'filtered_dataset_binary_classification.xlsx',
        SCRIPT_DIR / 'filtered_dataset_binary_classification.xlsx',
        Path.cwd() / 'filtered_dataset_binary_classification.xlsx',
        Path.cwd() / 'data' / 'results' / 'filtered_dataset_binary_classification.xlsx',
        Path('/mnt/data/filtered_dataset_binary_classification.xlsx'),
    )
    if default_input is None:
        raise FileNotFoundError('Could not locate filtered_dataset_binary_classification.xlsx')

    analyzer = ArticleAnalyzer(default_input)
    analyzer.run_analysis()
