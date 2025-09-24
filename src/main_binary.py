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
        self.metric_synonyms = df_ms.groupby('metric')['synonym'].apply(list).to_dict()


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

        #0 synonyms of headers in task_keywords csv
        col_map = {
            'Classification / Detection':    'classification',
            'Segmentation':                   'segmentation',
            'Prognosis (survival, recurrence, risk)':     'prognosis',
            'Synthesis / Image Enhancement':  'synthesis',
            'Integration / Recommendation (CDSS, multimodal)': 'integration',
            'NLP (text classification, information extraction)': 'nlp',
            'Genomic Models':                 'genomic',
            'Auxiliary Algorithmic Classes':  'auxiliary'
        }
        self.task_keywords.rename(columns=col_map, inplace=True)

        # 1. Prioritise tasks (from most important to least important)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        script_dir    = Path(__file__).parent.resolve()
        project_root  = script_dir.parent
        self.filtered_path = project_root / 'data' / 'filtered' / 'filtered_dataset.xlsx'

        # Add progress bar with tqdm
        tqdm.pandas()
        # Set logging level
        logging.basicConfig(filename='app.log',  # Set the filename
                            filemode='w',        # Set the file mode ('a' for append, 'w' for overwrite)
                            level=logging.DEBUG, # Set the logging level
                            format='%(asctime)s - %(levelname)s - %(message)s')  # Set the format)

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

    def categorize_binary(self, row, keywords_df):
        binary_result = {key_word: 0 for key_word in keywords_df.columns}
        
        # Field priority for checking
        fields_priority = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields_priority:
            field_text = str(row[field])
            logging.info(f"Checking '{field}': {field_text}")

            # Process text from each field
            matched_keywords = self.process_matched_text(field_text)

            # Iterate through all cancer types and check for keywords
            for key_type in keywords_df.columns:
                keywords_list = keywords_df[key_type].dropna()
                if any(key_word in matched_keywords for key_word in keywords_list):
                    binary_result[key_type] = 1
            # If keywords are found in a prioritized field (but we search for all words in this field)
            if any(binary_result.values()):
                break  # Move to the next row if a match is found in this field

        return pd.Series(binary_result)

    def categorize_task(self, row):
        result = {task: 0 for task in self.task_priority}
        fields = ['Article Title', 'Abstract', 'Author Keywords']
        for field in fields:
            text = str(row[field])
            matched = self.process_matched_text(text)
            for task in self.task_priority:
                keywords = self.task_keywords[task].dropna().str.lower()
                if any(kw in matched for kw in keywords):
                    result[task] = 1
                    return pd.Series(result)
        return pd.Series(result)

    def extract_auc_by_group(self, text: str) -> dict:
        # parsing  phrases patterns such as        "… were 0.81, 0.80, and 0.68 in the training group and 0.91, 0.80, and 0.81 in the test group …"
        # and returns a dictionary { model_name: auc_value } on the test group.
        pattern = re.compile(
            r'the auc of the (?P<models>.*?) were '
            r'(?P<train>[\d\.\s,]+) in the training group and '
            r'(?P<test>[\d\.\s,]+) in the test group',
            re.IGNORECASE
        )
        m = pattern.search(text)
        if not m:
            return {}
        raw_models = m.group('models')
        parts = re.split(r',|and', raw_models)
        models = [p.strip().replace(' model','') for p in parts if p.strip()]
        test_vals = [float(v) for v in m.group('test').split(',')]
        return {
            models[i]: test_vals[i]
            for i in range(min(len(models), len(test_vals)))
        }

    def assign_category(self, metric: str, value: float) -> str:
        for cutoff, label, comp in self.thresholds.get(metric.lower(), []):
            if comp == 'le' and value <= cutoff:
                return label
            if comp == 'ge' and value >= cutoff:
                return label
        return 'Unknown'

    def classify_performance(self, text: str) -> dict:
        """
        Parses all metric mentions by synonyms.
        If none are found, it takes the first percentage and puts it under proxy_metric.
        """
        results = {}

        # 1) Special template for AUC test-group (if available) 
        aucs = self.extract_auc_by_group(text)
        if 'combined' in aucs:
            results['roc-auc'] = self.assign_category('roc-auc', aucs['combined'])
            return results

        # 1) Special template for AUC test-group (if available) 
        all_syns = [syn for syns in self.metric_synonyms.values() for syn in syns]
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in sorted(all_syns, key=len, reverse=True)) + r')\b'
            r'.{0,20}?'
            r'(\d+\.?\d+)\s*(%?)',
            re.IGNORECASE
        )
        for m in pattern.finditer(text):
            found = m.group(1).lower()
            val   = float(m.group(2)) / (100 if m.group(3) == '%' else 1)
            for key, syns in self.metric_synonyms.items():
                if found in syns:
                    results[key] = self.assign_category(key, val)
                    break
        # 3) Fallback: if no metrics are found —
        #    take the first mention of a percentage and assign it to proxy_metric
        if not results:
            pct = re.search(r'(\d+\.?\d+)\s*%', text)
            if pct:
                val = float(pct.group(1)) / 100
                # choose proxy_metric thresholds
                results['proxy_metric'] = self.assign_category('proxy_metric', val)

        return results

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
        Returns a Series with columns
         - composite_metric
         - weighted_score
         - weighted_category
        """
        # 1) Select *one* task according to our task_priority
        task = None
        for t in self.task_priority:
            if row.get(t, 0) == 1:
                task = t
                break

        # 2) Composite: the first metric from the priority list with a non-empty value
        composite = None
        if task:
            for m in self.task_metric_priority[task]:
                val = row.get(m)
                if isinstance(val, str) and val.strip() and val.lower() != 'unknown':
                    composite = val  # keep textual "Very High" etc.
                    break

        # 3) Weighted: collect all categorized metrics and calculate dynamic weights
        weighted_score    = None
        weighted_category = None
        if task:
            prio_list = self.task_metric_priority[task]
            N = len(prio_list)
            # the most important metric gets weight N, the next N-1, ... the last — 1
            weights = {m: (N - idx) for idx, m in enumerate(prio_list)}

            total_w = 0
            acc     = 0
            for m, w in weights.items():
                cat = row.get(m)
                if isinstance(cat, str) and cat.strip() and cat.lower() != 'unknown':
                    score = self.category_scores.get(cat.lower())
                    if score is not None:
                        acc   += score * w
                        total_w += w

            if total_w > 0:
                weighted_score    = acc / total_w
                # find the closest category
                closest           = min(
                    self.category_scores.items(),
                    key=lambda kv: abs(kv[1] - weighted_score)
                )[0]
                weighted_category = closest

        return pd.Series({
            'composite_metric':   composite,
            'weighted_score':     weighted_score,
            'weighted_category':  weighted_category
        })

    def process_excel_file(self):
        try:
            # Load Excel file
            print(f"Loading: {self.filtered_path}")
            df = pd.read_excel(self.filtered_path) # For testing, limit the number of rows by  add '.head(100)'
            for col in ['Article Title','Abstract','Author Keywords']:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)  
            self.check_columns(df)
            self.add_keywords_to_matcher(self.cancer_keywords)
            self.add_keywords_to_matcher(self.ai_keywords)
            self.add_keywords_to_matcher(self.task_keywords)
            # Create binary classification for cancer types
            print("Creating binary classification for cancer types...") 
            df_cancer = df.progress_apply(lambda row: self.categorize_binary(row, self.cancer_keywords), axis=1)

            # Create binary classification for AI models task type
            print("Creating categorization for task_type of models...")
            df_task = df.progress_apply(self.categorize_task, axis=1)

            # Create binary classification for AI models
            print("Creating binary classification for AI models...")
            df_ai_model = df.progress_apply(lambda row: self.categorize_binary(row, self.ai_keywords), axis=1)

            # Process accuracy categories
            print("Classifying articles by model accuracy...")
            perf_df = df['Abstract'].progress_apply(lambda txt: pd.Series(self.classify_performance(txt)))

            # 1) collect performance + tasks
            df_perf_and_task = pd.concat([perf_df, df_task], axis=1)

            # 2) calculate composite + weighted over that DF
            print("Computing composite and weighted metrics...")
            extra = df_perf_and_task.progress_apply(
            self.compute_composite_and_weighted,
            axis=1
            )

            # 3) add these columns to perf_df
            perf_df = pd.concat([perf_df, extra], axis=1)

            # 4) finally combine EVERYTHING into one DF
            df_combined = pd.concat([
                df,
                perf_df,
                df_cancer,
                df_ai_model,
                df_task
            ], axis=1)

            # 5) save DF
            output_file = results_dir / f"{self.filtered_path.stem}_binary_classification.xlsx"
            df_combined.to_excel(output_file, index=False)
            print(f"Saved results to: {output_file}")
       
        except Exception as e:
            print(f"Error processing file: {e}")

if __name__ == '__main__':
    cancer_classifier = CancerClassifier()
    cancer_classifier.process_excel_file()