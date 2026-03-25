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

        self.cancer_keywords = pd.read_csv(sources_dir / 'cancer_keywords.csv')
        self.task_keywords = pd.read_csv(sources_dir / 'task_keywords.csv')
        self.ai_keywords   = pd.read_csv(sources_dir / 'ai_keywords.csv')

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

            escaped = [re.escape(s) for s in sorted(set(cleaned_syns), key=len, reverse=True)]
            if escaped:
                self.metric_patterns[metric] = re.compile(
                    r'(?<!\w)(?:' + '|'.join(escaped) + r')(?!\w)',
                    re.IGNORECASE
                )
            else:
                self.metric_patterns[metric] = None
    
        # Generic numeric pattern used near metric mentions
        self.metric_numeric_pattern = re.compile(
            r'(?<![\w.])(\d{1,3}(?:\.\d+)?)(\s*%)?(?![\w.])'
        )

        # Metrics that should normally be in [0, 1] unless explicitly written as %
        self.bounded_metrics = {
            'accuracy',
            'precision',
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
                    r'tested\s+on'
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
                    r'internal\s+validation'
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
                r'improv(?:e|ed|ement|ing)|'
                r'increase[sd]?|'
                r'decrease[sd]?|'
                r'reduc(?:e|ed|tion)|'
                r'outperform(?:ed|ing)?|'
                r'higher|lower|'
                r'gain(?:ed)?|'
                r'boost(?:ed)?|'
                r'drop(?:ped)?'
                r')\b.{0,20}\bby\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\b('
                r'increase[sd]?|'
                r'decrease[sd]?|'
                r'reduc(?:e|ed|tion)|'
                r'improv(?:e|ed|ement|ing)'
                r')\b.{0,20}\bfrom\b.{0,20}\bto\b',
                re.IGNORECASE
            ),
            re.compile(r'\berror reduction\b', re.IGNORECASE),
            re.compile(r'\brelative improvement\b', re.IGNORECASE),
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

    def add_keywords_to_matcher(self, keywords):
        for keyword_type in keywords.columns:
            logging.info(f"Keyword type: {keyword_type}")
            keywords_list = keywords[keyword_type].dropna()
            for keyword in keywords_list:
                logging.info(f"Keyword: {keyword}")
                keyword = keyword.lower()
                if '-' in keyword:
                    parts = keyword.split('-')
                    pattern1 = [{'LOWER': keyword.replace('-', '')}]  # case without hyphen
                    pattern2 = [{'LOWER': parts[0]}, {'LOWER': parts[1]}]  # case with space
                    pattern3 = [{'LOWER': parts[0]}, {'IS_PUNCT': True}, {'LOWER': parts[1]}]  # case with hyphen or other punctuation
                    self.matcher.add(keyword, [pattern1, pattern2, pattern3])
                elif ' ' in keyword:
                    parts = keyword.split(' ')
                    pattern1 = [{'LOWER': keyword.replace(' ', '')}]  # Original string
                    pattern2 = [{'LOWER': parts[0]}, {'LOWER': parts[1]}]  # case with space
                    self.matcher.add(keyword, [pattern1, pattern2])
                else:
                    pattern1 = [{'LOWER': keyword}]  # Original string
                    self.matcher.add(keyword, [pattern1])


    def match_keywords(self, text):
        logging.info(f"Text: {text}")
        doc = self.nlp(text)
        matches = self.matcher(doc)
        logging.info(f"Matches: {matches}")
        matched_keywords = set()
        for match_id, start, end in matches:
            logging.info(f"Matched keyword: {doc[start:end].text}")
            span = doc[start:end].text
            matched_keywords.add(span.lower())
        logging.info(f"Matched keywords: {matched_keywords}")
        return matched_keywords

    def process_matched_text(self, text):
        combined_text = self.preprocess_text_smart(text.lower())
        doc = self.nlp(combined_text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])
        logging.info(f"Lemmatized text: {lemmatized_text}")
        matched_keywords = self.match_keywords(lemmatized_text)
        return matched_keywords
    
    def _safe_text(self, value) -> str:
        if pd.isna(value):
            return ''
        return str(value)

    def _match_categories_in_text(self, text, keywords_df) -> dict:
        binary_result = {key_word: 0 for key_word in keywords_df.columns}
        clean_text = self._safe_text(text)

        if not clean_text.strip():
            return binary_result

        matched_keywords = self.process_matched_text(clean_text)

        for key_type in keywords_df.columns:
            keywords_list = keywords_df[key_type].dropna().astype(str).str.lower()
            if any(keyword in matched_keywords for keyword in keywords_list):
                binary_result[key_type] = 1

        return binary_result

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
                keywords = self.task_keywords[task].dropna().astype(str).str.lower()
                if any(kw in matched for kw in keywords):
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
            field_matches = self._match_categories_in_text(row.get(field, ''), self.ai_keywords)
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
        Title-first cancer extraction with minimal metadata.
        Backward compatibility:
        - keeps all existing one-hot cancer columns
        - adds only one new column: cancer_detected_in
        """
        binary_result = {key_word: 0 for key_word in keywords_df.columns}
        cancer_detected_in = None

        fields_priority = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields_priority:
            field_text = self._safe_text(row.get(field, ''))
            logging.info(f"Checking '{field}': {field_text}")

            matched_keywords = self.process_matched_text(field_text)
            field_has_match = False

            for key_type in keywords_df.columns:
                keywords_list = keywords_df[key_type].dropna().astype(str).str.lower()
                if any(key_word in matched_keywords for key_word in keywords_list):
                    binary_result[key_type] = 1
                    field_has_match = True

            if field_has_match:
                cancer_detected_in = field
                break

        return pd.Series({
            **binary_result,
            'cancer_detected_in': cancer_detected_in
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

    def _detect_metric_context(self, sentence: str) -> str:
        sentence_text = self._safe_text(sentence)
        for context_name, pattern in self.metric_context_patterns:
            if pattern.search(sentence_text):
                return context_name
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

        # Example: 95% CI ...
        if '%' in num_text and re.search(r'^\s*(ci\b|confidence interval)', post):
            return True

        # Example: CI 0.81-0.92
        if re.search(r'(ci\b|confidence interval)\s*[:=]?\s*$', pre):
            return True

        # Example: CI ... 0.81-0.92
        if re.search(r'^\s*[-–]\s*\d', post) and re.search(r'(ci\b|confidence interval)', pre):
            return True

        return False

    def _normalize_metric_value(self, raw_value: str, pct_text: str) -> float:
        value = float(raw_value)
        if pct_text and '%' in pct_text:
            value = value / 100.0
        return value

    def _metric_value_allowed(self, metric: str, value: float, pct_text: str) -> bool:
        """
        Prevent impossible interpretations for bounded metrics.
        Example:
        AUC 95  -> reject
        AUC 95% -> accept and normalize to 0.95
        """
        if metric in self.bounded_metrics and not (pct_text and '%' in pct_text):
            if value > 1.0:
                return False

        return True

    def _collect_metric_candidates(self, text: str):
        candidates_by_metric = {metric: [] for metric in self.metric_name_order}
        ignored_relative_change = False

        for sentence_index, sentence in enumerate(self._split_into_sentences(text)):
            sentence_text = self._safe_text(sentence)
            if not sentence_text.strip():
                continue

            context_name = self._detect_metric_context(sentence_text)
            context_rank = self.metric_context_rank.get(context_name, 0)

            numeric_matches = list(self.metric_numeric_pattern.finditer(sentence_text))
            if not numeric_matches:
                continue

            for metric, metric_pattern in self.metric_patterns.items():
                if metric_pattern is None:
                    continue

                metric_mentions = list(metric_pattern.finditer(sentence_text))
                if not metric_mentions:
                    continue

                for metric_match in metric_mentions:
                    best_local_candidate = None

                    for num_match in numeric_matches:
                        distance = self._distance_between_spans(
                            metric_match.start(), metric_match.end(),
                            num_match.start(), num_match.end()
                        )

                        if distance > 35:
                            continue

                        raw_value = num_match.group(1)
                        pct_text = num_match.group(2) or ''

                        try:
                            normalized_value = self._normalize_metric_value(raw_value, pct_text)
                        except ValueError:
                            continue

                        if not self._metric_value_allowed(metric, normalized_value, pct_text):
                            continue

                        if self._is_confidence_interval_candidate(sentence_text, num_match):
                            continue

                        left = min(metric_match.start(), num_match.start())
                        right = max(metric_match.end(), num_match.end())

                        link_text = sentence_text[left:right]
                        local_window = sentence_text[max(0, left - 30):min(len(sentence_text), right + 30)]

                        if self._is_relative_change_candidate(local_window, link_text):
                            ignored_relative_change = True
                            continue

                        candidate = {
                            'metric': metric,
                            'value': normalized_value,
                            'raw_value': f"{raw_value}{pct_text.strip()}",
                            'sentence': sentence_text,
                            'context': context_name,
                            'context_rank': context_rank,
                            'source_type': 'direct',
                            'distance': distance,
                            'sentence_index': sentence_index
                        }

                        if best_local_candidate is None:
                            best_local_candidate = candidate
                        elif distance < best_local_candidate['distance']:
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