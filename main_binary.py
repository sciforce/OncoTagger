import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import re
import logging

class CancerClassifier:
    def __init__(self):
        # Load keywords for cancer types and AI models
        self.cancer_keywords = pd.read_csv('cancer_keywords.csv')
        self.ai_keywords = pd.read_csv('ai_keywords.csv')
        self.nlp = spacy.load('en_core_web_sm')
        self.matcher = Matcher(self.nlp.vocab)
        self.file_path = '1-13429.xlsx'
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
        fields_priority = ['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus']

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
                
                    # Якщо знайдено ключове слово у полі з вищим пріоритетом, перериваємо цикл
                    if field == 'Article Title' or field == 'Abstract':
                        break
            # Якщо знайшли ключові слова в полі з пріоритетом, виходимо з циклу
            if any(binary_result.values()):
                break
    
        return pd.Series(binary_result)
        
    @staticmethod
    def classify_accuracy(description):
        if not isinstance(description, str):
            return "Unknown"

        accuarcy = {
            "Very high accuracy (≥ 95%)": [['outstanding performance', 'clinically reliable', 'superior classification', 'exceptional accuracy'], r'\b(0\.(9[5-9]\d*)|[9][5-9]\.\d+|100)(\%)?\b'],
            "High accuracy (90% - 94.9%)": [['high accuracy', 'reliable for diagnosis', 'good prediction', 'clinically useful'], r'\b(0\.(9[0-4]\d*)|[9][0-4]\.\d+|[9][0-4])(\%)?\b'],
            "Medium accuracy (80% - 89.9%)": [['moderate accuracy', 'acceptable performance', 'reasonable prediction', 'risk assessment'], r'\b(0\.(8[0-9]\d*)|[8][0-9]\.\d+|[8][0-9])(\%)?\b'],
            "Low accuracy (70% - 79.9%)": [['low accuracy', 'requires improvement', 'preliminary assessment', 'limited clinical use'], r'\b(0\.(7[0-9]\d*)|[7][0-9]\.\d+|[7][0-9])(\%)?\b'],
            "Very low accuracy (< 70%)": [['very low accuracy', 'unreliable', 'not suitable for clinical use', 'requires significant improvement'], r'\b(0\.(6\d+|[0-6]\.\d+|[0-6]))(\%)?\b']
        }

        for category, (keywords, pattern) in accuarcy.items():
            if any(word in description.lower() for word in keywords) or re.search(pattern, description):
                return category

        return "Unknown"

    def check_columns(self, df):
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Publication Year']
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
            df['Accuracy_Category'] = df['Abstract'].progress_apply(self.classify_accuracy)

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