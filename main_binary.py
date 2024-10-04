import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import re
import logging
import csv

# Configure the logging
logging.basicConfig(
    filename='app.log',  # Set the filename
    filemode='w',        # Set the file mode ('a' for append, 'w' for overwrite)
    level=logging.DEBUG, # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Set the format
)   
# Функція для видалення апострофів у тексті (наприклад, "barrett's" -> "barretts")
def preprocess_text_smart(text):
    # Видаляє апострофи тільки всередині слів, залишаючи цілі терміни
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)
    return text

# Завантажуємо модель spaCy
nlp = spacy.load('en_core_web_sm')

matcher = Matcher(nlp.vocab)

keywords_with_hyphens = ['triple-negative', 'ti-rads', 'fine-needle', 'multilevel-graph', 'bladder-cancer', 'early-stage', 'pancreatic-cancer', 'gastric-cancer', 'whole-brain', 'non-small', 'microsatellite-instability', 'androgen-dependent', 'castration-resistant', 'non-muscle-invasive', 'muscle-invasive', 'signet-ring', 'serous-endometrial', 'low-grade', 'high-grade', 't1-weighted', 't2-weighted', 'head-and-neck', 'gray-level', 'multi-gray', 'diffuse-intrinsic', 'small-cell', 'large-cell', 'endometrioid-type', 'neuroendocrine-tumors', 'cutaneus-squamous-cell', 'basal-cell', 'clear-cell', 'low-grade', 'non-seminoma', 'solid-pseudopapillary', 'pan-cancer', 'multi-cancer', 'extra-adrenal', 'catecholamine-secreting', 'gastrointestinal-stromal', 'low-grade-myofibroblastic', 'non-neoplastic', 'low-grade-fibromyxoid', 'smooth-muscle', 'soft-tissue', 'epithelioid-sarcoma', 'chronic-lymphocytic', 'acute-lymphoblastic', 'multiple-myeloma', 'stage-iii', 'stage-iv', 'barrett-esophagus', 'ck5/6', 'vascular-endothelial', 'endometrial-stromal', 'non-hodgkin', 'hodgkin-lymphoma', 'superficial-spreading', 'sentinel-lymph-node', 'abcde-criteria', 'breast-ductal-carcinoma', 'ductal-carcinoma', 'artificial-neural-network', 'convolutional-neural-network', 'long-short-term', 'large-cell-neuroendocrine', 'support-vector', 'data-driven', 'non-hodgkin-lymphoma', 'multi-cancer-type', 'organs-at-risk', 'wide-area', 'ovarian-cancer', 'pancreatobiliary-type']

for keyword in keywords_with_hyphens:
    parts = keyword.split('-')
    
    # Провіряємо, що parts містить как мінімум два елементи
    if len(parts) == 2:
        pattern1 = [{'LOWER': keyword.replace('-', '')}]  # case without hyphen
        pattern2 = [{'LOWER': parts[0]}, {'LOWER': parts[1]}]  # case with space
        pattern3 = [{'LOWER': parts[0]}, {'IS_PUNCT': True}, {'LOWER': parts[1]}]  # case with hyphen or other punctuation
        matcher.add(keyword, [pattern1, pattern2, pattern3])
    else:
        # якщо дефіса немає, додаємо тільки шаблон без дефіса
        pattern1 = [{'LOWER': keyword}]  # Оригинальная строка
        matcher.add(keyword, [pattern1])

matcher.add(keyword, [pattern1])

def match_keywords(text):
    doc = nlp(text)
    matches = matcher(doc)
    matched_keywords = set()
    for _, start, end in matches:  # Use underscore to ignore match_id
        span = doc[start:end].text
        matched_keywords.add(span.lower())
    return matched_keywords

# Функція для обробки тексту: лематизація та пошук ключових слів
def process_matched_text(text):
    combined_text = preprocess_text_smart(text.lower())
    
    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    matched_keywords = match_keywords(lemmatized_text)
    
    # Логирование для отладки
    logging.debug(f"Text: {text}")
    logging.debug(f"Lemmatized Text: {lemmatized_text}")
    logging.debug(f"Matched Keywords: {matched_keywords}")
    
    return matched_keywords

# Функция для бінарной классификации статей за типами рака
def categorize_cancer_binary(row, cancer_keywords):
    cancer_result = {cancer: 0 for cancer in cancer_keywords.columns}
    
    # Сначала проверяем только 'Article Title'
    title_text = str(row['Article Title'])
    matched_cancers = process_matched_text(title_text)
    
    # Логирование для отладки
    logging.debug(f"Title Text: {title_text}")
    logging.debug(f"Matched Cancers in Title: {matched_cancers}")
    
    for cancer_type in cancer_keywords.columns:
        cancer_keywords_list = cancer_keywords[cancer_type].dropna()
        if any(cancer_keyword in matched_cancers for cancer_keyword in cancer_keywords_list):
            cancer_result[cancer_type] = 1
    
    # Если ключевые слова не найдены в 'Article Title', проверяем остальные поля
    if all(value == 0 for value in cancer_result.values()):
        fields = ['Author Keywords', 'Keywords Plus', 'Abstract']
        for field in fields:
            field_text = str(row[field])
            matched_cancers = process_matched_text(field_text)
            
            # Логирование для отладки
            logging.debug(f"Field Text: {field_text}")
            logging.debug(f"Matched Cancers in {field}: {matched_cancers}")
            
            for cancer_type in cancer_keywords.columns:
                if any(cancer_keyword in matched_cancers for cancer_keyword in cancer_keywords_list):
                    cancer_result[cancer_type] = 1
    
    return pd.Series(cancer_result)

# Функция для бінарной классификации статей за моделями ИИ
def categorize_ai_model_binary(row, ai_keywords):
    ai_result = {ai_model: 0 for ai_model in ai_keywords.columns}
    
    # Сначала проверяем только 'Article Title'
    title_text = str(row['Article Title'])
    matched_ai_models = process_matched_text(title_text)
    
    # Логирование для отладки
    logging.debug(f"Title Text: {title_text}")
    logging.debug(f"Matched AI Models in Title: {matched_ai_models}")
    
    for ai_model in ai_keywords.columns:
        if any(keyword in matched_ai_models for keyword in ai_keywords[ai_model].dropna()):
            ai_result[ai_model] = 1
    
    # Если ключевые слова не найдены в 'Article Title', проверяем остальные поля
    if all(value == 0 for value in ai_result.values()):
        fields = ['Author Keywords', 'Keywords Plus', 'Abstract']
        for field in fields:
            field_text = str(row[field])
            matched_ai_models = process_matched_text(field_text)
            
            # Логирование для отладки
            logging.debug(f"Field Text: {field_text}")
            logging.debug(f"Matched AI Models in {field}: {matched_ai_models}")
            
            for ai_model in ai_keywords.columns:
                if any(keyword in matched_ai_models for keyword in ai_keywords[ai_model].dropna()):
                    ai_result[ai_model] = 1
    
    return pd.Series(ai_result)


# Функція для класифікації точності моделей
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

# Основна функція для обробки файлу
def process_excel_file(file_path):
    try:
         # Завантаження Excel файлу
        print(f"Завантаження файлу: {file_path}")
        df = pd.read_excel(file_path).head(100) # Для тестування можна обмежити кількість рядків

        # Перевірка, чи існують необхідні колонки
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Publication Year']
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")

        # Завантажуємо ключові слова для типів раку та моделей ШІ
        cancer_keywords = pd.read_csv('cancer_keywords.csv')
        ai_keywords = pd.read_csv('ai_keywords.csv')

        # Створення бінарної класифікації для типів раку
        print("Створення бінарної класифікації для типів раку...")
        with tqdm(total=len(df), desc="Типи раку") as pbar:
            df_cancer = df.progress_apply(lambda row: categorize_cancer_binary(row, cancer_keywords), axis=1)
            pbar.update(len(df))

        # Створення бінарної класифікації для моделей ШІ
        print("Створення бінарної класифікації для моделей ШІ...")
        with tqdm(total=len(df), desc="Моделі ШІ") as pbar:
            df_ai_model = df.progress_apply(lambda row: categorize_ai_model_binary(row, ai_keywords), axis=1)
            pbar.update(len(df))

        # Обробка категорій точності
        print("Класифікація статей за точністю моделей...")
        with tqdm(total=len(df), desc="Точність моделей") as pbar:
            df['Accuracy_Category'] = df.progress_apply(lambda description: classify_accuracy(description), axis=1)
            pbar.update(len(df))

        # Об'єднуємо результат з початковим DataFrame
        df_combined = pd.concat([df, df_cancer, df_ai_model], axis=1)

        # Збереження файлу з бінарною класифікацією
        output_file = file_path.replace('.xlsx', '_binary_classification.xlsx')
        df_combined.to_excel(output_file, index=False)
        print(f"Файл успішно збережено: {output_file}")
    
    except Exception as e:
        print(f"Помилка під час обробки файлу: {e}")

# Додавання прогресу з tqdm
tqdm.pandas()

# Виклик основної функції
if __name__ == "__main__":
    path_to_excel_file = '1-6437.xlsx'
    process_excel_file(path_to_excel_file)