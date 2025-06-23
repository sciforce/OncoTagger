import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import re
import logging

class CancerClassifier:
    metric_synonyms = {
        'accuracy':       ['accuracy', 'acc', 'precision', 'ppv', 'positive predictive value', 'overall accuracy', 'classification accuracy', 'correct classification rate'],
        'sensitivity':    ['sensitivity', 'recall', 'tpr', 'true positive rate'],
        'specificity':    ['specificity', 'tnr', 'true negative rate'],
        'f1-score':       ['f1-score', 'f1 score', 'f1'],
        'roc-auc':        ['roc-auc', 'roc auc', 'rocauc', 'auc', 'aucroc', 'auroc', 'area under roc curve', 'area under the roc curve', 'area under receiver operating characteristic', 'area under the receiver operating characteristic curve', 'auc-roc', 'auc roc'],
        'pr-auc':         ['pr-auc', 'pr auc', 'prauc',
            'area under pr curve', 'area under the pr curve',
            'area under precision-recall curve',
            'area under precision recall curve'],
        'balanced accuracy': ['balanced accuracy', 'balanced-accuracy', 'bacc',
            'balanced classification accuracy'],
        'mcc':            ['mcc', 'matthews correlation coefficient', 'matthews correlation-coefficient', 'matthews cc'],
        "cohen's kappa":  ["cohen's kappa", 'cohen kappa', 'kappa',
            'cohen-kappa', 'cohen kappa coefficient'],
        'dice':           ['dice', 'dice score', 'dice-score', 'dice coefficient',
            'dice similarity coefficient', 'dsc'],
        'iou':            ['iou', 'intersection over union', 'jaccard', 'jaccard index', 'jaccard-index'],
        'hd95':           ['hd95', 'hausdorff distance', 'hausdorff 95th percentile',
            'hausdorff-95', 'hd 95', 'hd-95', 'hausdorff'],
        'mae':            ['mae', 'mean absolute error', 'l1 error', 'l1 loss'],
        'rmse':           ['rmse', 'root mean squared error', 'root mean square error',
            'rms error', 'rms-error', 'rms loss', 'rms-loss'],
        'r2':             ['r2', 'r²', 'r^2', 'coefficient of determination',
            'r-squared', 'r squared'],
        'c-index':        ['c-index', 'concordance index', 'concordance-index',
            'c index', 'concordance', "harrell's c"]
    }

    def __init__(self):
        # Load keywords for cancer types and AI models
        self.cancer_keywords = pd.read_csv('cancer_keywords.csv')
        self.ai_keywords = pd.read_csv('ai_keywords.csv')
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self.file_path = 'filtered_dataset.xlsx'
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
        
        # Пріоритетність поля для перевірки
        fields_priority = ['Article Title', 'Abstract', 'Author Keywords']

        for field in fields_priority:
            field_text = str(row[field])
            logging.info(f"Checking '{field}': {field_text}")

            # Обробляємо текст з кожного поля
            matched_keywords = self.process_matched_text(field_text)

            # Пройдемося по всіх типах раку і перевіримо, чи є ключові слова
            for key_type in keywords_df.columns:
                keywords_list = keywords_df[key_type].dropna()
                if any(key_word in matched_keywords for key_word in keywords_list):
                    binary_result[key_type] = 1
            # Якщо знайшли ключові слова в полі, що є пріоритетним (але шукаємо всі слова в цьому полі)
            if any(binary_result.values()):
                break  # Переходимо до наступного рядка, якщо є знаходження в цьому полі

        return pd.Series(binary_result)
        
    @staticmethod
    def extract_auc_by_group(text: str) -> dict:
        """
        Парсит предложения вида
        "… were 0.81, 0.80, and 0.68 in the training group and 0.91, 0.80, and 0.81 in the test group …"
        и возвращает словарь { model_name: auc_value } по тестовой группе.
        """
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

    @staticmethod
    def assign_category(metric: str, value: float) -> str:

        thresholds = {
            'accuracy':       [(0.95,'Very High'), (0.85,'High'), (0.75,'Medium'),
                               (0.65,'Low'),      (0.00,'Very Low')],
            'sensitivity':    [(0.90,'Very High'), (0.80,'High'), (0.70,'Medium'),
                               (0.60,'Low'),      (0.00,'Very Low')],
            'specificity':    [(0.90,'Very High'), (0.80,'High'), (0.70,'Medium'),
                               (0.60,'Low'),      (0.00,'Very Low')],
            'precision':      [(0.95,'Very High'), (0.85,'High'), (0.75,'Medium'),
                               (0.65,'Low'),      (0.00,'Very Low')],
            'recall':         [(0.95,'Very High'), (0.85,'High'), (0.70,'Medium'),
                               (0.50,'Low'),      (0.00,'Very Low')],
            'f1-score':       [(0.85,'Very High'), (0.75,'High'), (0.60,'Medium'),
                               (0.50,'Low'),      (0.00,'Very Low')],
            'roc-auc':        [(0.90,'Very High'), (0.80,'High'), (0.70,'Medium'),
                               (0.60,'Low'),      (0.00,'Very Low')],
            'balanced accuracy': [(0.90,'Very High'), (0.80,'High'), (0.70,'Medium'),
                                  (0.60,'Low'),      (0.00,'Very Low')],
            'mcc':            [(0.70,'Very High'), (0.50,'High'), (0.30,'Medium'),
                               (0.10,'Low'),      (0.00,'Very Low')],
            "cohen's kappa":  [(0.70,'Very High'), (0.50,'High'), (0.30,'Medium'),
                               (0.10,'Low'),      (0.00,'Very Low')],
            'dice':           [(0.80,'Very High'), (0.70,'High'), (0.60,'Medium'),
                               (0.50,'Low'),      (0.00,'Very Low')],
            'iou':            [(0.70,'Very High'), (0.60,'High'), (0.40,'Medium'),
                               (0.30,'Low'),      (0.00,'Very Low')],
            'hd95':           [(1,    'Very High'), (4,    'High'),   (9,    'Medium'),
                               (19,   'Low'),      (float('inf'),'Very Low')],
            'mae':            [(1,    'Very High'), (3,    'High'),   (6,    'Medium'),
                               (12,   'Low'),      (float('inf'),'Very Low')],
            'rmse':           [(1,    'Very High'), (3,    'High'),   (6,    'Medium'),
                               (12,   'Low'),      (float('inf'),'Very Low')],
            'r2':             [(0.85,'Very High'), (0.70,'High'), (0.50,'Medium'),
                               (0.30,'Low'),      (0.00,'Very Low')],
            'c-index':        [(0.80,'Very High'), (0.70,'High'), (0.60,'Medium'),
                               (0.55,'Low'),      (0.00,'Very Low')],
        }
        for cutoff, label in thresholds.get(metric.lower(), []):
            
            if metric in ('hd95','mae','rmse'):
                if value <= cutoff:
                    return label
            else:
                if value >= cutoff:
                    return label
        return 'Unknown'

    @staticmethod
    def classify_performance(text: str) -> dict:
        """
        Парсит все упоминания метрик по синонимам.
        Если не нашлось ни одной — берёт первый %-ок и кладёт под proxy_metric.
        """
        results = {}

        # 1) Спец-шаблон для AUC test-group (если есть) 
        aucs = CancerClassifier.extract_auc_by_group(text)
        if 'combined' in aucs:
            results['roc-auc'] = CancerClassifier.assign_category('roc-auc', aucs['combined'])
            return results

        # 2) Общий парсинг по синонимам
        all_syns = [syn for syns in CancerClassifier.metric_synonyms.values() for syn in syns]
        pattern = re.compile(
            r'\b(' + '|'.join(re.escape(s) for s in sorted(all_syns, key=len, reverse=True)) + r')\b'
            r'.{0,20}?'
            r'(\d+\.?\d+)\s*(%?)',
            re.IGNORECASE
        )
        for m in pattern.finditer(text):
            found = m.group(1).lower()
            val   = float(m.group(2)) / (100 if m.group(3) == '%' else 1)
            for key, syns in CancerClassifier.metric_synonyms.items():
                if found in syns:
                    results[key] = CancerClassifier.assign_category(key, val)
                    break
        # 3) Фоллбэк: если не нашли ни одной метрики —
        #    берём первое упоминание процента и заводим proxy_metric
        if not results:
            pct = re.search(r'(\d+\.?\d+)\s*%', text)
            if pct:
                val = float(pct.group(1)) / 100
                # выберем в качестве шаблона пороги 'accuracy'
                results['proxy_metric'] = CancerClassifier.assign_category('accuracy', val)

        return results

    def check_columns(self, df):
        required_columns = ['Article Title', 'Author Keywords', 'Abstract', 'Publication Year']
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")

    def process_excel_file(self):
        try:
            # Load Excel file
            print(f"Loading file: {self.file_path}")
            df = pd.read_excel(self.file_path)  # For testing, limit the number of rows by  'add .head(100)'
            self.check_columns(df)
            self.add_keywords_to_matcher(self.cancer_keywords)
            self.add_keywords_to_matcher(self.ai_keywords)
            # Create binary classification for cancer types
            print("Creating binary classification for cancer types...")
            df_cancer = df.progress_apply(lambda row: self.categorize_binary(row, self.cancer_keywords), axis=1)

            # Create binary classification for AI models
            print("Creating binary classification for AI models...")
            df_ai_model = df.progress_apply(lambda row: self.categorize_binary(row, self.ai_keywords), axis=1)

            # Process accuracy categories
            print("Classifying articles by model accuracy...")
            perf_df = df['Abstract'].progress_apply(
            lambda txt: pd.Series(self.classify_performance(txt))
            )
            df = pd.concat([df, perf_df], axis=1)

            # Combine result with the original DataFrame
            df_combined = pd.concat([df, 
                                     df_cancer, 
                                     df_ai_model
                                     ], axis=1)

            # Save the file with binary classification
            output_file = self.file_path.replace('.xlsx', '_binary_classification.xlsx')
            df_combined.to_excel(output_file, index=False)
            print(f"File successfully saved: {output_file}")

        except Exception as e:
            print(f"Error processing file: {e}")

if __name__ == '__main__':
    cancer_classifier = CancerClassifier()
    cancer_classifier.process_excel_file()