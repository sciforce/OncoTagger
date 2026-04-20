import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import re
import logging
import os
from pathlib import Path

script_dir   = Path(__file__).parent.resolve()        # …/wos_oncoarticleclassifier/src
project_root = script_dir.parent                      # …/wos_oncoarticleclassifier
sources_dir  = project_root / 'sources'
filtered_path = project_root / 'data' / 'filtered' / 'filtered_dataset.xlsx'
results_dir  = project_root / 'data' / 'results'
results_dir.mkdir(exist_ok=True, parents=True)

class CancerClassifier:
    
    def __init__(self):
        # Load keywords for cancer types and AI models
        # main location
        script_dir = Path(__file__).parent.resolve()    # …/wos_oncoarticleclassifier/src
        project_root = script_dir.parent               # …/wos_oncoarticleclassifier
        sources_dir = project_root / 'sources'         # …/wos_oncoarticleclassifier/sources

        self.cancer_keywords = self._read_keyword_table(sources_dir / 'cancer_keywords.csv', min_expected_columns=5)
        self.task_keywords   = self._read_keyword_table(sources_dir / 'task_keywords.csv', min_expected_columns=3)
        self.ai_keywords     = self._read_keyword_table(sources_dir / 'ai_keywords.csv', min_expected_columns=5)
        soft_keywords_path = sources_dir / 'cancer_keywords_soft.csv'
        self.cancer_keywords_soft = self._load_optional_keyword_file(
            soft_keywords_path,
            reference_columns=self.cancer_keywords.columns
        )

        df_ms = pd.read_csv(sources_dir / 'metric_synonyms.csv')
        df_ms['metric'] = df_ms['metric'].astype(str).str.strip().str.lower()
        df_ms['synonym'] = df_ms['synonym'].astype(str).str.strip().str.lower()
        self.metric_synonyms = df_ms.groupby('metric')['synonym'].apply(list).to_dict()
        self.metric_name_order = list(self.metric_synonyms.keys())

        # Build compiled metric patterns once
        self.metric_patterns = {}
        for metric, syns in self.metric_synonyms.items():
            cleaned_syns = []
            for syn in syns:
                syn_norm = str(syn).strip().lower()
                if syn_norm:
                    cleaned_syns.append(syn_norm)
                    if re.fullmatch(r'[a-z]+', syn_norm) and syn_norm.endswith('y'):
                        cleaned_syns.append(f"{syn_norm[:-1]}ies")

            escaped = [re.escape(s) for s in sorted(set(cleaned_syns), key=len, reverse=True)]
            if escaped:
                self.metric_patterns[metric] = re.compile(
                    r'(?<!\w)(?:' + '|'.join(escaped) + r')(?!\w)',
                    re.IGNORECASE
                )
            else:
                self.metric_patterns[metric] = None
    
        # Generic numeric pattern used near metric mentions
        # Supports:
        # - standard decimals: 0.91
        # - leading-dot decimals: .95
        # - signed small deltas: -0.003 (filtered later)
        # Accepts sentence-final punctuation without matching inside longer decimals.
        self.metric_numeric_pattern = re.compile(
            r'(?<![\w.])([+-]?(?:\d{1,3}(?:\.\d+)?|\.\d+))(\s*%)?(?=(?:\s|$|[,;:\)\]]|\.(?!\d)))'
        )

        # Metrics that should normally be in [0, 1] unless explicitly written as %
        self.bounded_metrics = {
            'accuracy',
            'precision',
            'recall',
            'sensitivity',
            'specificity',
            'f1-score',
            'npv',
            'fpr',
            'roc-auc',
            'pr-auc',
            'balanced accuracy',
            'mcc',
            "cohen's kappa",
            'dice',
            'iou',
            'r2',
            'c-index'
        }

        # Evaluation-context ranking
        self.metric_context_rank = {
            'unknown': 0,
            'train': 1,
            'cross_validation_summary': 2,
            'holdout': 3,
            'validation': 4,
            'test': 5,
            'external_validation': 6
        }

        self.metric_context_patterns = [
            (
                'external_validation',
                re.compile(
                    r'\b('
                    r'external(?:ly)?\s+validat(?:ed|ion)|'
                    r'external\s+test(?:ing)?|'
                    r'independent\s+(?:external\s+)?(?:cohort|test|validation)|'
                    r'external\s+cohort|'
                    r'multi-?center\s+external'
                    r')\b',
                    re.IGNORECASE
                )
            ),
            (
                'test',
                re.compile(
                    r'\b('
                    r'test(?:ing)?\s+(?:set|cohort|data|dataset|split)|'
                    r'test\s+group|'
                    r'tested\s+on|'
                    r'for\s+test|'
                    r'on\s+test|'
                    r'in\s+test'
                    r')\b',
                    re.IGNORECASE
                )
            ),
            (
                'validation',
                re.compile(
                    r'\b('
                    r'validation\s+(?:set|cohort|data|dataset|split)|'
                    r'validation\s+group|'
                    r'validated\s+on|'
                    r'internal\s+validation|'
                    r'for\s+validation|'
                    r'on\s+validation|'
                    r'in\s+validation'
                    r')\b',
                    re.IGNORECASE
                )
            ),
            (
                'holdout',
                re.compile(r'\b(hold[- ]?out|held[- ]?out)\b', re.IGNORECASE)
            ),
            (
                'cross_validation_summary',
                re.compile(
                    r'\b('
                    r'(?:\d+[- ]?fold|five[- ]fold|ten[- ]fold)\s+cross[- ]validation|'
                    r'cross[- ]validation|'
                    r'cross validation|'
                    r'cv\s+results|'
                    r'mean\s+cv|'
                    r'average\s+cv'
                    r')\b',
                    re.IGNORECASE
                )
            ),
            (
                'train',
                re.compile(
                    r'\b('
                    r'train(?:ing)?\s+(?:set|cohort|data|dataset|split)|'
                    r'training\s+group'
                    r')\b',
                    re.IGNORECASE
                )
            ),
        ]

        # Relative-change language that should not be interpreted as absolute performance
        self.relative_change_patterns = [
            re.compile(
                r'\b('
                r'improv(?:e|ed|ement|ing)|improvement|'
                r'increas(?:e|ed|es|ing)|increase|'
                r'decreas(?:e|ed|es|ing)|decrease|'
                r'reduc(?:e|ed|es|tion|ing)|reduction|'
                r'outperform(?:ed|ing)?|'
                r'higher|lower|'
                r'gain(?:ed|ing)?|'
                r'boost(?:ed|ing)?|'
                r'drop(?:ped|ping)?|drop'
                r')\b.{0,80}\bby\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\b('
                r'increas(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|'
                r'reduc(?:e|ed|es|tion|ing)|improv(?:e|ed|ement|ing)'
                r')\b.{0,80}\bfrom\b.{0,80}\bto\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\b('
                r'increase\s+in|decrease\s+in|drop\s+in|reduction\s+in|'
                r'improvement\s+in|improvement\s+of|reduction\s+of'
                r')\b.{0,80}\b('
                r'auc|accuracy|precision|recall|sensitivity|specificity|'
                r'f1(?:-score)?|dice|iou|jaccard|c-index|mcc|balanced accuracy|hd95'
                r')\b',
                re.IGNORECASE
            ),
            re.compile(r'\berror reduction\b', re.IGNORECASE),
            re.compile(r'\brelative improvement\b', re.IGNORECASE),
            re.compile(r'\bcompared (?:with|to)\b', re.IGNORECASE),
            re.compile(r'\bvs\.?\s+(?:baseline|standard|conventional|suboptimal|manual)\b', re.IGNORECASE),
            re.compile(r'\bbetter than (?:baseline|standard|conventional)\b', re.IGNORECASE),
            re.compile(r'\bloss(?:es)?\b.{0,40}\bof\b', re.IGNORECASE),
            re.compile(
                r'\b(?:delta|difference|Δ)\b.{0,80}\b('
                r'auc|accuracy|precision|recall|sensitivity|specificity|'
                r'f1(?:-score)?|dice|iou|jaccard|c-index|mcc|balanced accuracy'
                r')\b',
                re.IGNORECASE
            ),
        ]

        # Hard-disable proxy fallback in main analysis
        self.enable_proxy_metric = False

        df_tp = pd.read_csv(sources_dir / 'task_priority.csv')
        self.task_priority = df_tp.sort_values('priority')['task'].tolist()

        df_tmp = pd.read_csv(sources_dir / 'task_metric_priority.csv')
        self.task_metric_priority = {
            task: grp.sort_values('order')['metric'].tolist()
            for task, grp in df_tmp.groupby('task')
        }

        df_cs = pd.read_csv(sources_dir / 'category_scores.csv')
        self.category_scores = dict(zip(df_cs['category'], df_cs['score']))

        df_th = pd.read_csv(sources_dir / 'thresholds.csv')
        self.thresholds = {
            metric: [
                (row['cutoff'], row['label'], row['comparison'])
                for _, row in grp.iterrows()
            ]
            for metric, grp in df_th.groupby('metric')
        }
        # Synonyms of headers in task_keywords.csv
        col_map = {
            'Classification / Detection': 'classification',
            'Segmentation': 'segmentation',
            'Prognosis (survival, recurrence, risk)': 'prognosis',
            'Synthesis / Image Enhancement': 'synthesis',
            'Integration / Recommendation (CDSS, multimodal)': 'integration',
            'NLP (text classification, information extraction)': 'nlp',
            'Genomic Models': 'genomic',
            'Auxiliary Algorithmic Classes': 'auxiliary'
        }
        self.task_keywords.rename(columns=col_map, inplace=True)
        # 1. Prioritise tasks (from most important to least important)
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)

        # Precompute normalized keyword maps once
        self.cancer_keywords_hard_map = self._build_keyword_map(self.cancer_keywords)
        self.cancer_keywords_soft_map = self._build_keyword_map(self.cancer_keywords_soft)
        self.ai_keywords_map = self._build_keyword_map(self.ai_keywords)
        self.task_keywords_map = self._build_keyword_map(self.task_keywords)

        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent
        self.filtered_path = project_root / 'data' / 'filtered' / 'filtered_dataset.xlsx'

        tqdm.pandas()

        logging.basicConfig(
            filename='app.log',
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    # Function to remove apostrophes in text (e.g., "barrett's" -> "barretts")
    def preprocess_text_smart(self, text):
        # Remove apostrophes only inside words, leaving whole terms
        return re.sub(r"(\w)'(\w)", r"\1\2", text)
    
    def _read_keyword_table(self, path: Path, min_expected_columns: int = 2) -> pd.DataFrame:
        last_exc = None

        candidates = [
            dict(sep=None, engine='python'),
            dict(sep=','),
            dict(sep=';'),
            dict(sep='\t'),
        ]

        for kwargs in candidates:
            try:
                df = pd.read_csv(path, **kwargs)
                df.columns = [str(c).strip() for c in df.columns]

                # reject obviously broken one-column parse like:
                # "Anal cancer;Breast cancer;Lung cancer;..."
                if len(df.columns) == 1:
                    only_col = str(df.columns[0])
                    if ';' in only_col or '\t' in only_col:
                        continue

                if len(df.columns) >= min_expected_columns:
                    return df
            except Exception as exc:
                last_exc = exc

        raise ValueError(
            f"Failed to parse keyword file correctly: {path}. "
            f"Detected columns: unreadable or only one merged header. Last error: {last_exc}"
        )
    
    def _load_optional_keyword_file(self, filepath: Path, reference_columns) -> pd.DataFrame:
        """
        Load optional keyword CSV.
        If file does not exist, create an empty DataFrame with reference columns.
        If some columns are missing, add them as empty.
        Extra columns are dropped.
        """
        if filepath.exists():
            df = self._read_keyword_table(filepath, min_expected_columns=1)
        else:
            df = pd.DataFrame(columns=list(reference_columns))

        for col in reference_columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype='object')

        df = df[list(reference_columns)]
        return df


    def _build_keyword_map(self, keywords_df: pd.DataFrame) -> dict:
        """
        Convert keyword DataFrame into:
        {
            'Breast cancer': {'breast cancer', 'mammary carcinoma', ...},
            ...
        }
        using the same normalization function as matcher comparison.
        """
        keyword_map = {}

        for col in keywords_df.columns:
            normalized_keywords = set()

            for raw_keyword in keywords_df[col].dropna().astype(str):
                normalized = self._normalize_keyword_for_matching(raw_keyword)
                if normalized:
                    normalized_keywords.add(normalized)

            keyword_map[col] = normalized_keywords

        return keyword_map


    def _binary_from_keyword_map(self, matched_keywords: set, keyword_map: dict) -> dict:
        """
        Convert matched canonical keywords into one-hot category dict.
        """
        result = {category: 0 for category in keyword_map.keys()}

        if not matched_keywords:
            return result

        for category, kw_set in keyword_map.items():
            if kw_set and (kw_set & matched_keywords):
                result[category] = 1

        return result


    def _any_positive_match(self, binary_result: dict) -> bool:
        return any(v == 1 for v in binary_result.values())
    def _normalize_keyword_for_matching(self, text: str) -> str:
        s = self._safe_text(text).strip().lower()
        if not s:
            return ''

        s = self.preprocess_text_smart(s)
        doc = self.nlp(s)

        lemmas = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = token.lemma_.strip().lower()
            if lemma:
                lemmas.append(lemma)

        return ' '.join(lemmas)


    def _build_keyword_patterns(self, keyword: str):
        """
        Build FULL patterns for the whole keyword, not only the first 1-2 tokens.
        Supports:
        - exact multi-token form
        - optional punctuation / hyphen between tokens
        - collapsed form without separators
        """
        canonical = self._normalize_keyword_for_matching(keyword)
        if not canonical:
            return canonical, []

        parts = canonical.split()
        if not parts:
            return canonical, []

        patterns = []

        # Exact full-token sequence
        patterns.append([{'LOWER': part} for part in parts])

        # Full-token sequence with optional punctuation between every pair
        punct_pattern = []
        for i, part in enumerate(parts):
            punct_pattern.append({'LOWER': part})
            if i < len(parts) - 1:
                punct_pattern.append({'IS_PUNCT': True, 'OP': '?'})
        patterns.append(punct_pattern)

        # Collapsed single-token form: e.g. "pap smear" -> "papsmear"
        if len(parts) > 1:
            patterns.append([{'LOWER': ''.join(parts)}])

        return canonical, patterns

    def add_keywords_to_matcher(self, keywords):
        for keyword_type in keywords.columns:
            keywords_list = keywords[keyword_type].dropna().astype(str)

            for raw_keyword in keywords_list:
                canonical, patterns = self._build_keyword_patterns(raw_keyword)
                if not canonical or not patterns:
                    continue

                # Store category + canonical keyword in match id
                match_id = f"{keyword_type}::{canonical}"
                self.matcher.add(match_id, patterns)


    def match_keywords(self, text):
        logging.info(f"Text: {text}")
        doc = self.nlp(text)
        matches = self.matcher(doc)
        logging.info(f"Matches: {matches}")

        matched_keywords = set()

        for match_id, start, end in matches:
            match_name = self.nlp.vocab.strings[match_id]

            if "::" in match_name:
                _, canonical_keyword = match_name.split("::", 1)
            else:
                canonical_keyword = match_name

            matched_keywords.add(canonical_keyword)

        logging.info(f"Matched canonical keywords: {matched_keywords}")
        return matched_keywords

    def process_matched_text(self, text):
        combined_text = self.preprocess_text_smart(self._safe_text(text).lower())
        doc = self.nlp(combined_text)

        normalized_tokens = []
        for token in doc:
            if token.is_space:
                continue
            if token.is_punct:
                normalized_tokens.append(token.text)
            else:
                normalized_tokens.append(token.lemma_.lower())

        normalized_text = ' '.join(normalized_tokens)
        logging.info(f"Normalized text for keyword matching: {normalized_text}")

        matched_keywords = self.match_keywords(normalized_text)
        return matched_keywords
    
    def _safe_text(self, value) -> str:
        if pd.isna(value):
            return ''
        return str(value)

    def _normalize_numeric_typography(self, text: str) -> str:
        """
        Normalizes OCR/typography-fragmented numeric forms for metric parsing only.
        The original sentence remains unchanged in trace columns.

        Fixes:
        - spaced decimals: '0 .88' -> '0.88', '96. 7%' -> '96.7%'
        - leading-dot decimals at token boundaries: '.95' -> '0.95'
        """
        s = self._safe_text(text)
        if not s:
            return ''

        s = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', s)
        s = re.sub(r'(^|[\s\(\[\{=,:;+\-])\.(\d+)', r'\g<1>0.\2', s)

        return s

    def _mask_matched_spans(self, text: str, patterns: list) -> str:
        s = self._safe_text(text)
        if not s:
            return ''

        masked = s
        for pattern in patterns:
            masked = pattern.sub(lambda m: ' ' * (m.end() - m.start()), masked)
        return masked

    def _mask_confidence_interval_spans(self, text: str) -> str:
        """
        Remove CI spans from candidate space while preserving character offsets.
        The main metric value before CI is left intact.
        """
        s = self._safe_text(text)
        if not s:
            return ''

        ci_patterns = [
            re.compile(r'\(\s*(?:95|90|99)\s*%\s*(?:ci|c\.i\.|confidence interval)[^)]*\)', re.IGNORECASE),
            re.compile(
                r'\b(?:95|90|99)\s*%\s*(?:ci|c\.i\.|confidence interval)\s*[:=]?\s*'
                r'\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%?\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(?:ci|c\.i\.|confidence interval)\s*[:=]?\s*'
                r'\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%?\b',
                re.IGNORECASE
            ),
            re.compile(r'\(\s*\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*%?\s*\)'),
        ]
        return self._mask_matched_spans(s, ci_patterns)

    def _is_range_component_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Precision-first policy:
        suppress pure ranges such as 0.76-0.81 or 97 to 98%.
        """
        s = self._safe_text(sentence).lower()
        start, end = num_match.span()
        pre = s[max(0, start - 16):start]
        post = s[end:min(len(s), end + 16)]

        if re.search(r'^\s*[-–]\s*\d', post):
            return True
        if re.search(r'^\s+to\s+\d', post):
            return True
        if re.search(r'\d\s*[-–]\s*$', pre):
            return True
        if re.search(r'\d\s+to\s*$', pre):
            return True

        local = s[max(0, start - 24):min(len(s), end + 24)]
        if re.search(r'\b(range|ranged|ranging|between)\b', local):
            if re.search(r'[-–]\s*\d', local) or re.search(r'\bto\s+\d', local):
                return True

        return False

    def _is_structural_number_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore structural numbers:
        - ResNet-50 / DenseNet-121 / EfficientNet-B0 style tokens
        - 1-year / 3-year / 5-year labels
        - class labels in 2 vs. 3 vs. 4
        - Top-1 / Rank-1 ordinals
        """
        sent = self._safe_text(sentence)
        sent_lower = sent.lower()
        raw = self._safe_text(num_match.group(1))
        pct = (num_match.group(2) or '').strip()
        start, end = num_match.span()

        pre20 = sent_lower[max(0, start - 20):start]
        post20 = sent_lower[end:min(len(sent_lower), end + 20)]
        local = sent_lower[max(0, start - 25):min(len(sent_lower), end + 25)]

        if re.search(
            r'(?:resnet|densenet|efficientnet|alexnet|mobilenet|inception|convnext|vgg|bert|u-?net|tabnet|retina(?:\s|-)?unet|vit|transformer)\s*[-–]?\s*$',
            pre20
        ):
            return True

        if re.search(r'\b\d+\s*[- ]?year\b', local):
            return True
        if re.search(r'\byear\s*[-–]?\s*$', pre20) or re.search(r'^\s*[-–]?\s*year\b', post20):
            return True

        if re.search(r'(?:top|rank)\s*[-–]?\s*$', pre20):
            return True

        if not pct and re.fullmatch(r'\d+', raw):
            try:
                value_int = int(raw)
            except ValueError:
                value_int = None

            if value_int is not None and value_int <= 10:
                if re.search(r'\b\d+\s*vs\.?\s*\d+(?:\s*vs\.?\s*\d+)+\b', sent_lower):
                    return True

        return False

    def _should_skip_numeric_candidate(self, metric: str, sentence: str, num_match: re.Match) -> bool:
        return (
            self._is_topk_ordinal_candidate(sentence, num_match)
            or self._is_structural_number_candidate(sentence, num_match)
            or self._is_range_component_candidate(sentence, num_match)
            or self._is_colon_ratio_candidate(sentence, num_match)
            or self._is_slash_ratio_candidate(sentence, num_match)
            or self._is_duration_or_runtime_candidate(sentence, num_match)
            or self._is_uncertainty_width_candidate(sentence, num_match)
            or self._is_confidence_interval_candidate(sentence, num_match)
            or self._is_auxiliary_nonperformance_number(metric, sentence, num_match)
        )

    def _between_text_contains_other_metric(self, metric: str, between_text: str) -> bool:
        bt = self._safe_text(between_text)
        if not bt.strip():
            return False

        for other_metric, pattern in self.metric_patterns.items():
            if other_metric == metric or pattern is None:
                continue
            if pattern.search(bt):
                return True

        return False

    def _number_has_competing_metric_anchor(
        self,
        metric: str,
        sentence: str,
        metric_match: re.Match,
        num_match: re.Match
    ) -> bool:
        """
        Reject candidates where the number is more explicitly attached to another metric
        than to the current one.
        """
        current_distance = self._distance_between_spans(
            metric_match.start(), metric_match.end(), num_match.start(), num_match.end()
        )

        for other_metric, pattern in self.metric_patterns.items():
            if other_metric == metric or pattern is None:
                continue

            for other_match in pattern.finditer(sentence):
                other_distance = self._distance_between_spans(
                    other_match.start(), other_match.end(), num_match.start(), num_match.end()
                )
                if other_distance > current_distance:
                    continue

                other_between = self._extract_between_text(sentence, other_match, num_match)
                if self._between_text_contains_other_metric(other_metric, other_between):
                    continue

                explicit = (
                    re.search(r'\b(of|was|were|reached|achieved|attained|yielded|reported|obtained|above|over|at least|exceed(?:ing|ed|s)?)\b', other_between.lower())
                    or '=' in other_between
                    or ':' in other_between
                    or (
                        len(other_between.strip()) <= 3
                        and re.fullmatch(r'[\s\(\)\[\]:=\-]*', other_between)
                    )
                )

                if explicit and other_distance + 2 < current_distance:
                    return True

        return False

    def _metric_numeric_link_is_explicit(
        self,
        metric: str,
        sentence: str,
        metric_match: re.Match,
        num_match: re.Match,
        between_text: str
    ) -> bool:
        """
        For most metrics, keep legacy permissive behavior.
        For accuracy, require an explicit local link to avoid generic bleed.
        Also reject candidate links that cross another metric name.
        """
        metric_key = self._safe_text(metric).strip().lower()
        bt = self._safe_text(between_text).lower()
        local_window = sentence[max(0, metric_match.start() - 20):min(len(sentence), num_match.end() + 20)].lower()

        if self._between_text_contains_other_metric(metric, bt):
            return False
        if self._number_has_competing_metric_anchor(metric, sentence, metric_match, num_match):
            return False

        if metric_key == 'accuracy':
            if re.search(r'\b(?:overall|classification|prediction|diagnostic)\s+accuracy\b', local_window):
                return True

            if re.search(
                r'\b(of|was|were|reached|achieved|attained|yielded|reported|obtained|above|over|at least|exceed(?:ing|ed|s)?)\b',
                bt
            ):
                return True

            if '=' in bt or ':' in bt:
                return True

            if len(bt.strip()) <= 3 and re.fullmatch(r'[\s\(\)\[\]:=\-]*', bt):
                return True

            return False

        return True

    def _build_metric_candidate(
        self,
        metric: str,
        sentence_parse: str,
        sentence_text: str,
        metric_match: re.Match,
        num_match: re.Match,
        sentence_index: int,
        source_type: str
    ):
        if self._metric_mention_is_threshold(sentence_parse, metric_match):
            return None, False

        if self._should_skip_numeric_candidate(metric, sentence_parse, num_match):
            return None, False

        between_text = self._extract_between_text(sentence_parse, metric_match, num_match)
        if not str(source_type).startswith('enumeration_'):
            if not self._metric_numeric_link_is_explicit(metric, sentence_parse, metric_match, num_match, between_text):
                return None, False

        raw_value = num_match.group(1)
        pct_text = num_match.group(2) or ''

        try:
            normalized_value, implied_pct = self._normalize_metric_value(metric, raw_value, pct_text, between_text)
        except ValueError:
            return None, False

        if not self._metric_value_allowed(metric, normalized_value, pct_text):
            return None, False

        link_left = min(metric_match.start(), num_match.start())
        link_right = max(metric_match.end(), num_match.end())
        link_text = sentence_parse[link_left:link_right]
        local_window = sentence_parse[max(0, link_left - 30):min(len(sentence_parse), link_right + 30)]

        if self._is_relative_change_candidate(local_window, link_text):
            return None, True

        context_name = self._detect_local_metric_context(sentence_parse, num_match, metric_match)
        context_rank = self.metric_context_rank.get(context_name, 0)
        quality_score = self._candidate_quality_score(metric, sentence_parse, metric_match, num_match)

        final_source_type = source_type
        if implied_pct:
            final_source_type = f"{final_source_type}_implied_pct"
        modality = self._detect_candidate_modality(sentence_parse, metric_match, num_match)
        if modality:
            final_source_type = f"{final_source_type}_{modality}"

        candidate = {
            'metric': metric,
            'value': normalized_value,
            'raw_value': f"{raw_value}{pct_text.strip()}",
            'sentence': sentence_text,
            'context': context_name,
            'context_rank': context_rank,
            'source_type': final_source_type,
            'distance': self._distance_between_spans(metric_match.start(), metric_match.end(), num_match.start(), num_match.end()),
            'quality_score': quality_score,
            'sentence_index': sentence_index
        }
        return candidate, False

    def _match_categories_in_text(self, text, keywords_df, keyword_map=None) -> dict:
        binary_result = {key_word: 0 for key_word in keywords_df.columns}
        clean_text = self._safe_text(text)

        if not clean_text.strip():
            return binary_result

        matched_keywords = self.process_matched_text(clean_text)

        if keyword_map is None:
            keyword_map = self._build_keyword_map(keywords_df)

        return self._binary_from_keyword_map(matched_keywords, keyword_map)

    def _ordered_unique(self, items, ordered_reference):
        seen = set()
        ordered = []

        reference_set = set(items)
        for item in ordered_reference:
            if item in reference_set and item not in seen:
                ordered.append(item)
                seen.add(item)

        for item in items:
            if item not in seen:
                ordered.append(item)
                seen.add(item)

        return ordered

    def classify_task_with_metadata(self, row) -> pd.Series:
        """
        Keeps the single-label primary_task logic for downstream metric interpretation.
        Also adds metadata without breaking one-hot task columns.
        """
        result = {task: 0 for task in self.task_priority}
        all_matches = []
        primary_task = None
        task_source_field = None

        fields = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields:
            text = self._safe_text(row.get(field, ''))
            matched = self.process_matched_text(text)

            field_matches = []
            for task in self.task_priority:
                kw_set = self.task_keywords_map.get(task, set())
                if kw_set and (kw_set & matched):
                    field_matches.append(task)

            if field_matches:
                all_matches.extend(field_matches)
                if primary_task is None:
                    primary_task = field_matches[0]
                    task_source_field = field

        if primary_task:
            result[primary_task] = 1

        ordered_matches = self._ordered_unique(all_matches, self.task_priority)

        return pd.Series({
            **result,
            'primary_task': primary_task,
            'task_source_field': task_source_field,
            'all_tasks': '; '.join(ordered_matches) if ordered_matches else None
        })

    def categorize_ai_models(self, row) -> pd.Series:
        """
        AI-model extraction is recall-oriented:
        collect matches across title + abstract + keywords,
        instead of stopping at the first field.
        """
        binary_result = {key_word: 0 for key_word in self.ai_keywords.columns}
        matched_fields = []

        fields = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields:
            field_matches = self._match_categories_in_text(
                row.get(field, ''),
                self.ai_keywords,
                keyword_map=self.ai_keywords_map
            )
            if any(field_matches.values()):
                matched_fields.append(field)
                for key, val in field_matches.items():
                    if val == 1:
                        binary_result[key] = 1

        return pd.Series({
            **binary_result,
            'ai_detected_in': '; '.join(matched_fields) if matched_fields else None
        })

    def categorize_binary(self, row, keywords_df):
        """
        Cancer extraction with two keyword levels:
        - hard keywords: primary anchors
        - soft keywords: fallback organ-specific proxies

        Priority logic:
        1) scan fields in order: Title -> Abstract -> Author Keywords
        2) if a hard match is found in any field, use it immediately and stop
        3) if no hard match exists anywhere, use the first soft-only match
        4) keep legacy one-hot output columns
        5) add metadata columns describing hard/soft source
        """
        binary_result = {key_word: 0 for key_word in self.cancer_keywords.columns}

        cancer_detected_in = None
        cancer_match_level = None
        cancer_hard_detected_in = None
        cancer_soft_detected_in = None

        first_soft_result = None
        first_soft_field = None

        fields_priority = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields_priority:
            field_text = self._safe_text(row.get(field, ''))
            logging.info(f"Checking cancer field '{field}': {field_text}")

            if not field_text.strip():
                continue

            matched_keywords = self.process_matched_text(field_text)

            hard_result = self._binary_from_keyword_map(
                matched_keywords,
                self.cancer_keywords_hard_map
            )
            soft_result = self._binary_from_keyword_map(
                matched_keywords,
                self.cancer_keywords_soft_map
            )

            hard_hit = self._any_positive_match(hard_result)
            soft_hit = self._any_positive_match(soft_result)

            # Hard match wins immediately
            if hard_hit:
                binary_result.update(hard_result)
                cancer_detected_in = field
                cancer_match_level = 'hard'
                cancer_hard_detected_in = field
                break

            # Store first soft-only match as fallback
            if soft_hit and first_soft_result is None:
                first_soft_result = soft_result
                first_soft_field = field

        # No hard anywhere -> use earliest soft fallback
        if cancer_match_level is None and first_soft_result is not None:
            binary_result.update(first_soft_result)
            cancer_detected_in = first_soft_field
            cancer_match_level = 'soft'
            cancer_soft_detected_in = first_soft_field

        return pd.Series({
            **binary_result,
            'cancer_detected_in': cancer_detected_in,
            'cancer_match_level': cancer_match_level,
            'cancer_hard_detected_in': cancer_hard_detected_in,
            'cancer_soft_detected_in': cancer_soft_detected_in
        })

    def categorize_task(self, row):
        """
        Backward-compatible wrapper.
        Returns only legacy one-hot task columns.
        """
        full_result = self.classify_task_with_metadata(row)
        return full_result[self.task_priority]
    
    def _split_into_sentences(self, text: str) -> list:
        clean_text = self._safe_text(text)
        if not clean_text.strip():
            return []

        doc = self.nlp(clean_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text and sent.text.strip()]
        if sentences:
            return sentences

        return [chunk.strip() for chunk in re.split(r'(?<=[\.\?!;])\s+', clean_text) if chunk.strip()]

    def _distance_between_spans(self, start_a: int, end_a: int, start_b: int, end_b: int) -> int:
        if end_a <= start_b:
            return start_b - end_a
        if end_b <= start_a:
            return start_a - end_b
        return 0

    def _normalize_context_label(self, label: str) -> str:
        s = self._safe_text(label).strip().lower()
        s = re.sub(r'\s+', ' ', s)

        if 'external' in s:
            return 'external_validation'
        if s in {'testing', 'test'}:
            return 'test'
        if s in {'training', 'train'}:
            return 'train'
        if 'hold' in s:
            return 'holdout'
        if 'cross' in s or s == 'cv':
            return 'cross_validation_summary'
        if 'validation' in s:
            return 'validation'
        return 'unknown'

    def _find_ordered_context_pair(self, sentence: str):
        """
        Detect phrases like:
        - for training and testing data
        - training and testing data
        - for test and validation sets
        Returns a tuple like ('train', 'test') or None.
        """
        text = self._safe_text(sentence)

        context_alt = (
            r'external validation|external test|'
            r'training|train|testing|test|validation'
        )

        patterns = [
            rf'\bfor\s+(?P<c1>{context_alt})\s+(?:and|/)\s+(?P<c2>{context_alt})\s+(?:data|dataset|datasets|set|sets|cohort|cohorts|split|splits)?\b',
            rf'\b(?P<c1>{context_alt})\s+(?:and|/)\s+(?P<c2>{context_alt})\s+(?:data|dataset|datasets|set|sets|cohort|cohorts|split|splits)\b',
        ]

        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                c1 = self._normalize_context_label(m.group('c1'))
                c2 = self._normalize_context_label(m.group('c2'))
                if c1 != 'unknown' and c2 != 'unknown' and c1 != c2:
                    return (c1, c2)

        return None



    def _is_topk_ordinal_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore the ordinal token in:
        - top-1 accuracy
        - top 5 accuracy
        - rank-1
        """
        start, end = num_match.span()
        pre = sentence[max(0, start - 10):start].lower()
        return bool(re.search(r'(?:top|rank)\s*[-–]?\s*$', pre, flags=re.IGNORECASE))


    def _is_colon_ratio_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore numbers from ratios like 80:20.
        """
        start, end = num_match.span()
        pre = sentence[max(0, start - 1):start]
        post = sentence[end:min(len(sentence), end + 1)]
        return pre == ':' or post == ':'


    def _is_slash_ratio_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """Ignore numbers from ratios like 70/30."""
        start, end = num_match.span()
        pre = sentence[max(0, start - 1):start]
        post = sentence[end:min(len(sentence), end + 1)]
        return pre == '/' or post == '/'


    def _is_duration_or_runtime_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore numbers followed by time/runtime units:
        - 4 seconds
        - 10 min
        """
        if (num_match.group(2) or '').strip() == '%':
            return False

        _, end = num_match.span()
        post = sentence[end:min(len(sentence), end + 18)].lower()

        return bool(re.match(
            r'^\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|ms|milliseconds?)\b',
            post,
            flags=re.IGNORECASE
        ))


    def _is_uncertainty_width_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore uncertainty widths in:
        - 0.922 +/- 0.01  (ignore 0.01)
        - 0.922±0.01      (ignore 0.01)
        - 87 +/- 2.2%     (ignore 2.2%)
        """
        sent = self._safe_text(sentence)
        start, _ = num_match.span()
        pre = sent[max(0, start - 8):start].lower()
        return bool(re.search(r'(?:±|\+/?\s*-)', pre))


    def _metric_mention_is_threshold(self, sentence: str, metric_match: re.Match) -> bool:
        """
        If the metric mention is used as a threshold definition (not performance), skip this mention.
        Example: 'IoU threshold of 0.2 ...'
        """
        window = sentence[max(0, metric_match.start() - 18):min(len(sentence), metric_match.end() + 18)].lower()
        return bool(re.search(r'\b(threshold|cut[- ]?off|cutoff)\b', window))


    def _is_auxiliary_nonperformance_number(self, metric: str, sentence: str, num_match: re.Match) -> bool:
        """
        Adjacency-based blockers for common non-performance numbers.
        Designed to avoid blocking true performance numbers just because unrelated
        counts or study descriptors occur elsewhere in the sentence.
        """
        s = sentence.lower()
        raw_sentence = self._safe_text(sentence)
        start, end = num_match.span()
        pre20 = s[max(0, start - 20):start]
        pre35 = s[max(0, start - 35):start]
        pre45 = s[max(0, start - 45):start]
        raw_pre20 = raw_sentence[max(0, start - 20):start]
        post25 = s[end:min(len(s), end + 25)]
        post40 = s[end:min(len(s), end + 40)]

        if re.search(r'\bp\s*[<=>]\s*$', raw_pre20):
            return True
        if 'p-value' in pre20 or 'p value' in pre20:
            return True

        if re.search(r'(threshold|cut[- ]?off|cutoff)\s*(?:of|=|:)?\s*$', pre20):
            return True
        if re.search(r'^\s*(threshold|cut[- ]?off|cutoff)\b', post25):
            return True

        if re.search(r'(reduc|compress|retain).{0,18}(feature|features|dimension|dimensions|dimensionality).{0,12}(?:to|down to)?\s*(?:just|only)?\s*$', pre45):
            return True
        if re.search(r'(reduc(?:ed|es|tion)|compress(?:ed|ion)|retained|retention)\s*(?:to|down to)?\s*$', pre35):
            return True
        if re.search(r'^\s*%?\s*(?:of\s+the\s+)?original\s+(?:set|feature|features|dimension|dimensions)\b', post40):
            return True

        if re.search(r'\bbrier\b\s*(?:score|loss)?\s*\(?\s*$', pre20):
            return True

        if re.search(r'\b(loss(?:es)?|error|difference|delta|Δ)\b\s*(?:of|=|:)?\s*$', pre20):
            return True

        if re.search(r'\b(?:n|patients?|samples?|cases?|subjects?|records?|images?)\s*(?:=|:)?\s*$', pre20):
            return True
        if re.search(r'^\s*(?:patients?|samples?|cases?|subjects?|records?|images?)\b', post25):
            return True

        if re.search(r'\b(?:fold|folds)\b\s*$', pre20):
            return True

        return False


    def _candidate_quality_score(self, metric: str, sentence: str, metric_match: re.Match, num_match: re.Match) -> int:
        """
        Heuristic tie-breaker when several values are plausible.
        Higher score = better candidate.

        This scorer is intentionally conservative:
        - small positive boosts for tight metric↔value linkage
        - negative penalties for common distractor numbers
        (loss, delta/improvement, dimensions/features, Brier, p-values, thresholds)
        """
        metric_name = self._safe_text(metric).strip().lower()

        num_start, num_end = num_match.span()
        met_start, met_end = metric_match.span()

        left = max(0, min(num_start, met_start) - 45)
        right = min(len(sentence), max(num_end, met_end) + 45)
        local_window = sentence[left:right].lower()

        number_window_left = max(0, num_start - 28)
        number_window_right = min(len(sentence), num_end + 28)
        number_window = sentence[number_window_left:number_window_right].lower()

        metric_window_left = max(0, met_start - 18)
        metric_window_right = min(len(sentence), met_end + 18)
        metric_window = sentence[metric_window_left:metric_window_right].lower()

        between = ''
        if num_start >= met_end:
            between = sentence[met_end:num_start].lower()
        elif met_start >= num_end:
            between = sentence[num_end:met_start].lower()

        score = 0

        # ---------------------------------------------------------
        # Positive linkage signals
        # ---------------------------------------------------------
        if re.search(r'\b(of|was|were|reached|achieved|attained|yielded|reported|obtained)\b', between):
            score += 1
        if '=' in between or ':' in between:
            score += 1

        if met_start >= num_end:
            gap = sentence[num_end:met_start].lower()
            if len(gap) <= 3 and re.fullmatch(r'[\s\(\)\[\]:=\-]*', gap):
                score += 2

        if re.search(r'\b(over|above|at least|no less than|not less than|or higher|or more)\b', local_window):
            score += 1
        if '>' in between or '≥' in between:
            score += 1

        if metric_name == 'accuracy':
            if re.search(r'\bclassification\b', local_window):
                score += 3
            if re.search(r'\bclassif(?:y|ication)\b', local_window):
                score += 2
            if re.search(r'\btop\s*[-–]?\s*1\b', local_window):
                score += 2
            if re.search(r'\btop\s*[-–]?\s*5\b', local_window):
                score -= 1

        if re.search(
            r'\b(best(?:-performing)?|best model|best results|ensemble model|final model|optimal model|top-performing|highest-performing|outstanding results)\b',
            local_window
        ):
            score += 4

        # ---------------------------------------------------------
        # Negative penalties for distractor-number contexts
        # These do not replace blockers; they help tie-breaking.
        # ---------------------------------------------------------

        # 1) Relative change / improvement / reduction / delta language
        # e.g. "improved by 20%", "reduction by 15%", "delta AUC 0.03"
        if re.search(
            r'\b('
            r'improv(?:e|ed|ement)|increase[sd]?|decrease[sd]?|reduc(?:e|ed|tion)|gain(?:ed)?|drop(?:ped)?|'
            r'delta|difference|margin|changed?|boost(?:ed)?'
            r')\b',
            number_window
        ):
            score -= 4

        if re.search(r'\bby\b', between) and re.search(
            r'\b(improv(?:e|ed|ement)|increase[sd]?|decrease[sd]?|reduc(?:e|ed|tion)|gain(?:ed)?|delta)\b',
            local_window
        ):
            score -= 3

        # 2) Loss / error / degradation language
        # Important: do not penalize canonical error metrics like MAE/RMSE.
        if metric_name not in {'mae', 'rmse'}:
            if re.search(r'\b(loss|losses|error|errors|degradation|penalty)\b', number_window):
                score -= 5

        # 3) Feature/dimension/subset/retained-feature counts or proportions
        if re.search(
            r'\b('
            r'feature|features|dimension|dimensions|dimensionality|variable|variables|'
            r'subset|selected|retained|remaining|original set|feature set'
            r')\b',
            number_window
        ):
            score -= 5

        # 4) Brier / calibration-loss style numbers are distractors for most metrics
        if re.search(r'\bbrier\b', number_window):
            score -= 5

        # 5) p-values / significance notation
        # Especially important to stop precision/P alias and nearby statistical numbers.
        if re.search(
            r'(\bp\s*[- ]?value\b|\bp\s*[=<>]\s*|\bp\s*≤\s*|\bp\s*≥\s*)',
            number_window
        ):
            score -= 6

        # 6) Threshold / cutoff / operating point numbers
        if re.search(
            r'\b(threshold|cut[\s-]?off|operating point|decision threshold|confidence threshold)\b',
            number_window
        ):
            score -= 4

        # 7) Confidence interval / uncertainty leftovers
        # Blockers should catch most of these; this is a fallback penalty.
        if re.search(r'\b(ci|c\.i\.|confidence interval|95% ci|sd|se|stderr|std\.? dev)\b', number_window):
            score -= 3
        if re.search(r'(\+/-|±)', number_window):
            score -= 3

        # 8) Structural/count distractors: folds, classes, lesions, ROIs, structures
        if re.search(
            r'\b(fold|folds|cv|cross-validation|class|classes|roi|rois|lesion|lesions|structure|structures)\b',
            number_window
        ):
            score -= 2

        # 9) Very small rescue bonus for clean parenthesized or assignment-style local links
        if re.search(r'^[\s\(\[]*$', between) or re.search(r'[\(\[]\s*$', between):
            score += 1

        return score


    def _detect_local_metric_context(self, sentence: str, num_match: re.Match, metric_match: re.Match) -> str:
        """
        Determine context for a specific numeric candidate, not for the whole sentence.
        Prefer the context mention nearest to the numeric span.
        """
        explicit_context = self._explicit_context_for_number(sentence, num_match)
        if explicit_context and explicit_context != 'unknown':
            return explicit_context

        num_start, num_end = num_match.span()
        met_start, met_end = metric_match.span()

        left = max(0, min(num_start, met_start) - 45)
        right = min(len(sentence), max(num_end, met_end) + 45)
        local_window = sentence[left:right]

        best_context = None
        best_key = None

        for context_name, pattern in self.metric_context_patterns:
            for match in pattern.finditer(local_window):
                ctx_start = left + match.start()
                ctx_end = left + match.end()
                dist = self._distance_between_spans(ctx_start, ctx_end, num_start, num_end)
                key = (dist, -self.metric_context_rank.get(context_name, 0))
                if best_key is None or key < best_key:
                    best_key = key
                    best_context = context_name

        if best_context is not None:
            return best_context

        return self._detect_metric_context(sentence)

    def _detect_metric_context(self, sentence: str) -> str:
        """
        Sentence-level fallback only.
        If several contexts are mentioned, prefer the highest-ranked one.
        """
        sentence_text = self._safe_text(sentence)

        matched_contexts = []
        for context_name, pattern in self.metric_context_patterns:
            if pattern.search(sentence_text):
                matched_contexts.append(context_name)

        if matched_contexts:
            return max(matched_contexts, key=lambda c: self.metric_context_rank.get(c, 0))

        return 'unknown'

    def _is_relative_change_candidate(self, local_window: str, link_text: str) -> bool:
        text_to_check = f"{self._safe_text(local_window)} || {self._safe_text(link_text)}"
        return any(pattern.search(text_to_check) for pattern in self.relative_change_patterns)

    def _is_confidence_interval_candidate(self, sentence: str, num_match: re.Match) -> bool:
        """
        Ignore CI numbers such as:
        95% CI 0.81-0.92
        but do not suppress the main metric value before the CI.
        """
        sentence_lower = self._safe_text(sentence).lower()
        start, end = num_match.span()
        num_text = num_match.group(0).lower()

        pre = sentence_lower[max(0, start - 20):start]
        post = sentence_lower[end:min(len(sentence_lower), end + 20)]

        if '%' in num_text and re.search(r'^\s*(ci\b|confidence interval)', post):
            return True

        if re.search(r'(ci\b|confidence interval)\s*[:=]?\s*$', pre):
            return True

        if re.search(r'^\s*[-–]\s*\d', post) and re.search(r'(ci\b|confidence interval)', pre):
            return True

        ci_vicinity_pre = sentence_lower[max(0, start - 40):start]
        if re.search(r'(ci\b|confidence interval)', ci_vicinity_pre) and re.search(r'[-–]\s*$', pre):
            return True

        return False

    def _explicit_context_for_number(self, sentence: str, num_match: re.Match) -> str:
        """
        Extract context from patterns like:
        - 0.91 for validation
        - 0.93 for test
        - test: 0.93
        """
        s = sentence.lower()
        start, end = num_match.span()
        post = s[end:min(len(s), end + 30)]
        m = re.search(
            r'^\s*(?:%?\s*)?(?:for|on|in)\s+(external validation|external test|validation|test|training|train)\b',
            post
        )
        if m:
            return self._normalize_context_label(m.group(1))

        pre = s[max(0, start - 40):start]
        m2 = re.search(
            r'(external validation|external test|validation|test|training|train)\s*[:=]\s*(?:[a-z][a-z\- ]{0,20})?$',
            pre
        )
        if m2:
            return self._normalize_context_label(m2.group(1))

        return None

    def _extract_enumerative_mappings(self, sentence: str, numeric_matches: list) -> list:
        """
        Returns list of tuples: (metric, metric_match, num_match, source_type)

        Supports:
        - metric1, metric2, metric3 were/of x, y, z
        - slash-delimited value lists when they are unambiguous
        - legacy simple 2-metric adjacent pairs
        """
        s = sentence.lower()
        if len(numeric_matches) < 2:
            return []

        metric_hits = []
        for metric, pat in self.metric_patterns.items():
            if pat is None:
                continue
            for match in pat.finditer(sentence):
                metric_hits.append((match.start(), match.end(), metric, match))
        metric_hits.sort()

        if len(metric_hits) < 2:
            return []

        first_num_start = numeric_matches[0].start()

        pre_hits = [(st, metric, match) for st, _, metric, match in metric_hits if st < first_num_start]
        pre_order = []
        pre_match = {}
        for _, metric, match in pre_hits:
            if metric not in pre_match:
                pre_match[metric] = match
                pre_order.append(metric)

        if len(pre_order) >= 2:
            linker_zone = s[pre_match[pre_order[-1]].end():first_num_start]
            has_list_linker = bool(
                re.search(r'\b(of|was|were|are|achieved|reached|attained|yielded|reported|obtained)\b', linker_zone)
                or '=' in linker_zone or ':' in linker_zone
            )

            if 'respectively' in s:
                end_pos = s.find('respectively')
                nums = [nm for nm in numeric_matches if nm.start() < end_pos]
                if has_list_linker and len(nums) == len(pre_order):
                    return [
                        (metric, pre_match[metric], nm, 'enumeration_respectively')
                        for metric, nm in zip(pre_order, nums)
                    ]
            else:
                nums = list(numeric_matches)
                if has_list_linker and len(nums) == len(pre_order):
                    return [
                        (metric, pre_match[metric], nm, 'enumeration_list')
                        for metric, nm in zip(pre_order, nums)
                    ]

        if len(metric_hits) >= 2:
            for i in range(len(metric_hits) - 1):
                _, a_end, m1, mm1 = metric_hits[i]
                b_start, b_end, m2, mm2 = metric_hits[i + 1]
                connector = s[a_end:b_start]
                if not re.search(r'\b(and|/)\b', connector):
                    continue
                if b_start - a_end > 18:
                    continue

                nums = [nm for nm in numeric_matches if nm.start() >= b_end and nm.start() - b_end <= 140]
                if len(nums) < 2:
                    continue
                between_nums = s[nums[0].end():nums[1].start()]
                if not (',' in between_nums or re.search(r'\b(and|/)\b', between_nums)):
                    continue

                return [
                    (m1, mm1, nums[0], 'enumeration_pair'),
                    (m2, mm2, nums[1], 'enumeration_pair')
                ]

        return []

    def _normalize_metric_value(self, metric: str, raw_value: str, pct_text: str, between_text: str) -> tuple:
        """
        Returns (value_float, implied_pct_flag)
        """
        value = float(raw_value)

        if pct_text and '%' in pct_text:
            return (value / 100.0, False)

        metric_key = self._safe_text(metric).strip().lower()

        if metric_key in self.bounded_metrics and 1.0 < value <= 100.0:
            bt = (between_text or '').lower()
            if (
                re.search(r'\b(of|was|were|reached|achieved|attained|yielded|reported|obtained)\b', bt)
                or any(ch in bt for ch in ['=', ':', '(', '['])
                or bt.strip() == ''
            ):
                return (value / 100.0, True)

        return (value, False)

    def _metric_value_allowed(self, metric: str, value: float, pct_text: str) -> bool:
        """
        Prevent impossible interpretations for bounded metrics.
        Example:
        AUC 95  -> reject unless safely normalized earlier
        AUC 95% -> accept and normalize to 0.95
        """
        metric_key = self._safe_text(metric).strip().lower()

        if metric_key in self.bounded_metrics:
            if value < 0:
                return False
            if not (pct_text and '%' in pct_text) and value > 1.0:
                return False

        return True

    def _extract_between_text(self, sentence: str, metric_match: re.Match, num_match: re.Match) -> str:
        if num_match.start() >= metric_match.end():
            return sentence[metric_match.end():num_match.start()]
        return sentence[num_match.end():metric_match.start()]

    def _detect_candidate_modality(self, sentence: str, metric_match: re.Match, num_match: re.Match) -> str:
        link_span_left = min(metric_match.start(), num_match.start())
        link_span_right = max(metric_match.end(), num_match.end())
        local_window = sentence[max(0, link_span_left - 25):min(len(sentence), link_span_right + 25)].lower()

        if re.search(r'\b(over|above|more than|greater than|or higher|or more|exceed(?:ing|ed|s)?)\b', local_window) or '>' in local_window:
            return 'lower_bound'
        if re.search(r'\b(at least|no less than|not less than)\b', local_window) or '≥' in local_window:
            return 'at_least'

        return None

    def _collect_metric_candidates(self, text: str):
        candidates_by_metric = {metric: [] for metric in self.metric_name_order}
        ignored_relative_change = False

        for sentence_index, sentence in enumerate(self._split_into_sentences(text)):
            sentence_text = self._safe_text(sentence)
            if not sentence_text.strip():
                continue

            sentence_parse = self._normalize_numeric_typography(sentence_text)
            sentence_for_numbers = self._mask_confidence_interval_spans(sentence_parse)

            numeric_matches = list(self.metric_numeric_pattern.finditer(sentence_for_numbers))
            if not numeric_matches:
                continue

            used_num_spans_global = set()
            enumeration_numeric_matches = [
                nm for nm in numeric_matches
                if not (
                    self._is_topk_ordinal_candidate(sentence_parse, nm)
                    or self._is_structural_number_candidate(sentence_parse, nm)
                    or self._is_range_component_candidate(sentence_parse, nm)
                    or self._is_colon_ratio_candidate(sentence_parse, nm)
                    or self._is_slash_ratio_candidate(sentence_parse, nm)
                    or self._is_duration_or_runtime_candidate(sentence_parse, nm)
                    or self._is_uncertainty_width_candidate(sentence_parse, nm)
                    or self._is_confidence_interval_candidate(sentence_parse, nm)
                )
            ]
            enum_maps = self._extract_enumerative_mappings(sentence_parse, enumeration_numeric_matches)

            for metric, metric_match, num_match, stype in enum_maps:
                candidate, rel_flag = self._build_metric_candidate(
                    metric=metric,
                    sentence_parse=sentence_parse,
                    sentence_text=sentence_text,
                    metric_match=metric_match,
                    num_match=num_match,
                    sentence_index=sentence_index,
                    source_type=stype
                )
                if rel_flag:
                    ignored_relative_change = True
                if candidate is None:
                    continue

                candidates_by_metric[metric].append(candidate)
                used_num_spans_global.add(num_match.span())

            for metric, metric_pattern in self.metric_patterns.items():
                if metric_pattern is None:
                    continue

                metric_mentions = list(metric_pattern.finditer(sentence_parse))
                if not metric_mentions:
                    continue

                for metric_match in metric_mentions:
                    if self._metric_mention_is_threshold(sentence_parse, metric_match):
                        continue

                    used_num_spans = set(used_num_spans_global)

                    explicit_pairs = []
                    for num_match in numeric_matches:
                        if num_match.span() in used_num_spans:
                            continue
                        if num_match.start() < metric_match.end():
                            continue
                        if num_match.start() - metric_match.end() > 200:
                            continue
                        ctx = self._explicit_context_for_number(sentence_parse, num_match)
                        if ctx and ctx != 'unknown':
                            explicit_pairs.append((ctx, num_match))

                    explicit_added = False

                    if len({ctx for ctx, _ in explicit_pairs}) >= 2:
                        for ctx, num_match in explicit_pairs:
                            candidate, rel_flag = self._build_metric_candidate(
                                metric=metric,
                                sentence_parse=sentence_parse,
                                sentence_text=sentence_text,
                                metric_match=metric_match,
                                num_match=num_match,
                                sentence_index=sentence_index,
                                source_type='paired_contextual'
                            )
                            if rel_flag:
                                ignored_relative_change = True
                            if candidate is None:
                                continue

                            candidate['context'] = ctx
                            candidate['context_rank'] = self.metric_context_rank.get(ctx, 0)
                            candidates_by_metric[metric].append(candidate)
                            used_num_spans.add(num_match.span())
                            explicit_added = True

                    if explicit_added:
                        continue

                    ordered_pair = self._find_ordered_context_pair(sentence_parse)
                    paired_added = False

                    if ordered_pair is not None:
                        pair_numeric_matches = []

                        for num_match in numeric_matches:
                            if num_match.span() in used_num_spans:
                                continue
                            if num_match.start() < metric_match.end():
                                continue
                            if num_match.start() - metric_match.end() > 140:
                                continue

                            candidate, rel_flag = self._build_metric_candidate(
                                metric=metric,
                                sentence_parse=sentence_parse,
                                sentence_text=sentence_text,
                                metric_match=metric_match,
                                num_match=num_match,
                                sentence_index=sentence_index,
                                source_type='paired_contextual'
                            )
                            if rel_flag:
                                ignored_relative_change = True
                            if candidate is None:
                                continue

                            pair_numeric_matches.append((num_match, candidate))

                        if len(pair_numeric_matches) >= 2:
                            selected_pairs = pair_numeric_matches[:2]

                            for idx, (num_match, candidate) in enumerate(selected_pairs):
                                context_name = ordered_pair[idx]
                                candidate['context'] = context_name
                                candidate['context_rank'] = self.metric_context_rank.get(context_name, 0)
                                candidates_by_metric[metric].append(candidate)
                                used_num_spans.add(num_match.span())
                                paired_added = True

                    if paired_added:
                        continue

                    best_local_candidate = None

                    for num_match in numeric_matches:
                        if num_match.span() in used_num_spans or num_match.span() in used_num_spans_global:
                            continue

                        distance = self._distance_between_spans(
                            metric_match.start(), metric_match.end(),
                            num_match.start(), num_match.end()
                        )

                        between_text = self._extract_between_text(sentence_parse, metric_match, num_match)

                        if distance > 45:
                            strong_link = (
                                re.search(r'\b(of|was|were|reached|achieved|attained|yielded|reported|obtained|above|over|at least|exceed(?:ing|ed|s)?)\b', between_text.lower())
                                or '=' in between_text or ':' in between_text
                            )
                            if not strong_link or distance > 140:
                                continue

                        candidate, rel_flag = self._build_metric_candidate(
                            metric=metric,
                            sentence_parse=sentence_parse,
                            sentence_text=sentence_text,
                            metric_match=metric_match,
                            num_match=num_match,
                            sentence_index=sentence_index,
                            source_type='direct'
                        )
                        if rel_flag:
                            ignored_relative_change = True
                        if candidate is None:
                            continue

                        if best_local_candidate is None:
                            best_local_candidate = candidate
                        else:
                            current_key = (
                                candidate['context_rank'],
                                candidate['quality_score'],
                                -candidate['distance']
                            )
                            best_key = (
                                best_local_candidate['context_rank'],
                                best_local_candidate['quality_score'],
                                -best_local_candidate['distance']
                            )
                            if current_key > best_key:
                                best_local_candidate = candidate

                    if best_local_candidate is not None:
                        candidates_by_metric[metric].append(best_local_candidate)

        return candidates_by_metric, ignored_relative_change

    def _select_best_metric_candidate(self, candidates: list):
        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda c: (
                -c['context_rank'],
                -int(str(c.get('source_type', '')).startswith(('paired_contextual', 'enumeration_'))),
                -c.get('quality_score', 0),
                c['distance'],
                c['sentence_index']
            )
        )
        return ranked[0]

    def _build_empty_performance_output(self) -> dict:
        output = {}
        for metric in self.metric_name_order:
            output[metric] = None
            output[f'metric_context_{metric}'] = None
            output[f'metric_raw_value_{metric}'] = None
            output[f'metric_sentence_{metric}'] = None
            output[f'metric_source_type_{metric}'] = 'none'

        # Keep the legacy column available, but disabled in main analysis
        output['proxy_metric'] = None

        output['no_metrics_reported'] = 1
        output['suspicious_extraction_flag'] = 0
        return output

    def _metric_is_usable_for_scoring(self, row: pd.Series, metric: str) -> bool:
        category_value = row.get(metric)

        if not isinstance(category_value, str):
            return False

        category_clean = category_value.strip().lower()
        if not category_clean or category_clean == 'unknown':
            return False

        source_type_col = f'metric_source_type_{metric}'
        source_type = str(row.get(source_type_col, 'direct')).strip().lower()

        if source_type in {'fallback', 'ignored_relative_change', 'none'}:
            return False

        return True

    def extract_auc_by_group(self, text: str) -> dict:
        """
        Deprecated helper retained only for backward compatibility.
        The main parser now uses general sentence-level candidate extraction
        with context ranking.
        """
        return {}

    def assign_category(self, metric: str, value: float) -> str:
        if value is None or pd.isna(value):
            return 'Unknown'

        metric_key = str(metric).strip().lower()

        for cutoff, label, comp in self.thresholds.get(metric_key, []):
            try:
                cutoff_value = float(cutoff)
            except (TypeError, ValueError):
                continue

            if comp == 'le' and value <= cutoff_value:
                return label
            if comp == 'ge' and value >= cutoff_value:
                return label

        return 'Unknown'

    def classify_performance(self, text: str) -> dict:
        """
        Improved metric extraction:
        - no permissive first-percent proxy fallback
        - ignores improved-by / increased-by / reduction-by language
        - ranks contexts:
        external_validation > test > validation > holdout >
        cross_validation_summary > train > unknown
        - preserves backward-compatible metric category columns
        - adds minimal traceability metadata
        """
        output = self._build_empty_performance_output()

        candidates_by_metric, ignored_relative_change = self._collect_metric_candidates(text)
        found_any_metric = False

        for metric in self.metric_name_order:
            best = self._select_best_metric_candidate(candidates_by_metric.get(metric, []))
            if best is None:
                continue

            output[metric] = self.assign_category(metric, best['value'])
            output[f'metric_context_{metric}'] = best['context']
            output[f'metric_raw_value_{metric}'] = best['raw_value']
            output[f'metric_sentence_{metric}'] = best['sentence']
            output[f'metric_source_type_{metric}'] = best['source_type']
            found_any_metric = True

        output['no_metrics_reported'] = 0 if found_any_metric else 1
        output['suspicious_extraction_flag'] = 1 if ignored_relative_change else 0

        # Proxy fallback stays disabled by default
        if not found_any_metric and self.enable_proxy_metric:
            output['proxy_metric'] = 'Unknown'

        return output

    def check_columns(self, df):
        required_columns = ['Article Title', 'Author Keywords', 'Abstract', 'Publication Year']
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")
    
    def build_weights_from_priority(self, task: str) -> dict:
        """
        Automatically builds normalized metric weights
        based on the priority order for a given task.
        """
        prio = self.task_metric_priority.get(task, [])
        n = len(prio)
        if n == 0:
            return {}
        raw = {m: n - idx for idx, m in enumerate(prio)}       # найважливіша: n, остання: 1
        total = sum(raw.values())
        return {m: raw[m] / total for m in raw}

    def compute_composite_and_weighted(self, row: pd.Series) -> pd.Series:
        """
        Returns a Series with columns:
        - composite_metric
        - composite_source
        - weighted_score
        - weighted_category

        Backward compatibility is preserved:
        - primary_task is preferred if present
        - legacy one-hot task columns remain supported
        - fallback / invalid metric sources are excluded from scoring
        """
        task = row.get('primary_task')
        if not isinstance(task, str) or not task.strip():
            task = None
            for t in self.task_priority:
                if row.get(t, 0) == 1:
                    task = t
                    break

        composite = None
        composite_source = None
        weighted_score = None
        weighted_category = None

        if task:
            priority_metrics = self.task_metric_priority.get(task, [])

            # Composite = first usable metric by task-specific priority
            for metric_name in priority_metrics:
                if self._metric_is_usable_for_scoring(row, metric_name):
                    composite = row.get(metric_name)
                    composite_source = metric_name
                    break

            # Weighted score = weighted mean over all usable metrics
            if priority_metrics:
                n_metrics = len(priority_metrics)
                weights = {
                    metric_name: (n_metrics - idx)
                    for idx, metric_name in enumerate(priority_metrics)
                }

                total_weight = 0
                score_accumulator = 0.0

                for metric_name, weight in weights.items():
                    if not self._metric_is_usable_for_scoring(row, metric_name):
                        continue

                    category_value = str(row.get(metric_name)).strip().lower()
                    numeric_score = self.category_scores.get(category_value)

                    if numeric_score is None:
                        continue

                    score_accumulator += numeric_score * weight
                    total_weight += weight

                if total_weight > 0:
                    weighted_score = score_accumulator / total_weight
                    weighted_category = min(
                        self.category_scores.items(),
                        key=lambda kv: abs(kv[1] - weighted_score)
                    )[0]

        return pd.Series({
            'composite_metric': composite,
            'composite_source': composite_source,
            'weighted_score': weighted_score,
            'weighted_category': weighted_category
        })

    def process_excel_file(self):
        try:
            print(f"Loading: {self.filtered_path}")
            df = pd.read_excel(self.filtered_path)

            for col in ['Article Title', 'Abstract', 'Author Keywords']:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)

            self.check_columns(df)

            # Build matcher vocabulary once
            self.add_keywords_to_matcher(self.cancer_keywords)
            self.add_keywords_to_matcher(self.cancer_keywords_soft)
            self.add_keywords_to_matcher(self.ai_keywords)
            self.add_keywords_to_matcher(self.task_keywords)

            # Cancer extraction:
            # keep the existing title-first precision-biased logic
            print("Creating binary classification for cancer types...")
            df_cancer = df.progress_apply(
                lambda row: self.categorize_binary(row, self.cancer_keywords),
                axis=1
            )

            # Task extraction:
            # keep single-label primary_task logic, but add metadata
            print("Creating categorization for task_type of models...")
            df_task = df.progress_apply(self.classify_task_with_metadata, axis=1)

            # AI extraction:
            # more recall-oriented than cancer extraction; scan all fields
            print("Creating binary classification for AI models...")
            df_ai_model = df.progress_apply(self.categorize_ai_models, axis=1)

            # Performance extraction:
            # improved parser with context ranking and relative-change protection
            print("Classifying articles by model accuracy...")
            perf_df = df['Abstract'].progress_apply(
                lambda txt: pd.Series(self.classify_performance(txt))
            )

            # Compute composite + weighted
            print("Computing composite and weighted metrics...")
            df_perf_and_task = pd.concat([perf_df, df_task], axis=1)
            extra = df_perf_and_task.progress_apply(
                self.compute_composite_and_weighted,
                axis=1
            )
            perf_df = pd.concat([perf_df, extra], axis=1)

            # Combine everything
            df_combined = pd.concat([
                df,
                perf_df,
                df_cancer,
                df_ai_model,
                df_task
            ], axis=1)

            output_file = results_dir / f"{self.filtered_path.stem}_binary_classification.xlsx"
            df_combined.to_excel(output_file, index=False)
            print(f"Saved results to: {output_file}")

        except Exception as e:
            print(f"Error processing file: {e}")

if __name__ == '__main__':
    cancer_classifier = CancerClassifier()
    cancer_classifier.process_excel_file()