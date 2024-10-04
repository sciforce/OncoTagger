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
        self.file_path = '1-6437.xlsx'
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
            logging.info(f"Adding keywords for: {keyword_type}")
            keywords_list = keywords[keyword_type].dropna()
            logging.info(f"Keywords: {keywords_list}")
            for keyword in keywords_list:
                logging.info(f"Keyword: {keyword}")
                keyword = keyword.lower()
                parts = keyword.split('-')
                if len(parts) == 2:
                    pattern1 = [{'LOWER': keyword.replace('-', '')}]  # case without hyphen
                    pattern2 = [{'LOWER': parts[0]}, {'LOWER': parts[1]}]  # case with space
                    pattern3 = [{'LOWER': parts[0]}, {'IS_PUNCT': True}, {'LOWER': parts[1]}]  # case with hyphen or other punctuation
                    self.matcher.add(keyword, [pattern1, pattern2, pattern3])
                else:
                    # if no hyphen, add only the pattern without hyphen
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
        # First check only 'Article Title'
        title_text = str(row['Article Title'])
        logging.info(f"Article Title: {title_text}")
        matched = self.process_matched_text(title_text)
        for key_type in keywords_df.columns:
            keywords_list = keywords_df[key_type].dropna()
            if any(key_word in matched for key_word in keywords_list):
                binary_result[key_type] = 1

        # If keywords are not found in 'Article Title', check other fields
        if all(value == 0 for value in binary_result.values()):
            fields = ['Author Keywords', 'Keywords Plus', 'Abstract']
            for field in fields:
                field_text = str(row[field])
                matched_additional = self.process_matched_text(field_text)

                for key_type in keywords_df.columns:
                    keywords_list = keywords_df[key_type].dropna()
                    if any(key_word in matched_additional for key_word in keywords_list):
                        binary_result[key_type] = 1
        return pd.Series(binary_result)

    @staticmethod
    def classify_accuracy(description):
        if not isinstance(description, str):
            return "Unknown"

        very_high_accuracy_pattern = r'\b(0\.(9[5-9]\d*)|[9][5-9]\.\d+|100)(\%)?\b'
        high_accuracy_pattern = r'\b(0\.(9[0-4]\d*)|[9][0-4]\.\d+|[9][0-4])(\%)?\b'
        medium_accuracy_pattern = r'\b(0\.(8[0-9]\d*)|[8][0-9]\.\d+|[8][0-9])(\%)?\b'
        low_accuracy_pattern = r'\b(0\.(7[0-9]\d*)|[7][0-9]\.\d+|[7][0-9])(\%)?\b'
        very_low_accuracy_pattern = r'\b(0\.(6\d+|[0-6]\.\d+|[0-6]))(\%)?\b'

        very_high_keywords = ['outstanding performance', 'clinically reliable', 'superior classification', 'exceptional accuracy']
        high_keywords = ['high accuracy', 'reliable for diagnosis', 'good prediction', 'clinically useful']
        medium_keywords = ['moderate accuracy', 'acceptable performance', 'reasonable prediction', 'risk assessment']
        low_keywords = ['low accuracy', 'requires improvement', 'preliminary assessment', 'limited clinical use']
        very_low_keywords = ['very low accuracy', 'unreliable', 'not suitable for clinical use', 'requires significant improvement']

        if re.search(very_high_accuracy_pattern, description) or any(word in description.lower() for word in very_high_keywords):
            return "Very high accuracy (≥ 95%)"
        elif re.search(high_accuracy_pattern, description) or any(word in description.lower() for word in high_keywords):
            return "High accuracy (90% - 94.9%)"
        elif re.search(medium_accuracy_pattern, description) or any(word in description.lower() for word in medium_keywords):
            return "Medium accuracy (80% - 89.9%)"
        elif re.search(low_accuracy_pattern, description) or any(word in description.lower() for word in low_keywords):
            return "Low accuracy (70% - 79.9%)"
        elif re.search(very_low_accuracy_pattern, description) or any(word in description.lower() for word in very_low_keywords):
            return "Very low accuracy (< 70%)"

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
            df = pd.read_excel(self.file_path).head(5)  # For testing, limit the number of rows
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