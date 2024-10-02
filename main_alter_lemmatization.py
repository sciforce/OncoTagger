import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import re
import logging

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

# Функція для класифікації статей за типом раку
keywords = pd.read_csv('cancer_keywords.csv')

def match_keywords(text):
    doc = nlp(text)
    matches = matcher(doc)
    matched_keywords = set()
    for match_id, start, end in matches:
        span = doc[start:end].text
        matched_keywords.add(span.lower())
    return matched_keywords


# Function to process matched text
def process_matched_text(text):
    combined_text = preprocess_text_smart(text.lower())
    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    matched_cancers = match_keywords(combined_text)
    return lemmatized_text, matched_cancers

def categorize_cancer(row):
    # Process each field separately and collect results into lists
    fields = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract']

    cancer_types = []
    for field in fields:
        field_text = str(row[field])
        lemmatized_text, matched_cancers = process_matched_text(field_text)

    # Check for cancer types in matched keywords
        for cancer_type in keywords.columns:
            if any(cancer_keyword in matched_cancers for cancer_keyword in keywords[cancer_type].dropna()):
                cancer_types.append((field,cancer_type))

    # Additional check for cancer types in lemmatized text
        for cancer_type in keywords.columns:
            if any(f" {cancer_keyword} " in f" {lemmatized_text} " for cancer_keyword in keywords[cancer_type].dropna()):
                cancer_types.append((field,cancer_type))

    if len(cancer_types) > 0:
        return cancer_types
    return 'unknown'

# Функція для класифікації статей за моделями ШІ
def categorize_ai_model(row):
    ai_keywords = pd.read_csv('ai_keywords.csv')
    fields = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract']
    ai_types = []
    for field in fields:
        field_text = str(row[field])
        combined_text = preprocess_text_smart(field_text.lower())
        doc = nlp(combined_text)
        lemmatized_text = ' '.join([token.lemma_ for token in doc])

        for ai_type in ai_keywords.columns:
            if any(keyword in lemmatized_text for keyword in ai_keywords[ai_type].dropna()):
                ai_types.append((field, ai_type))
    if len(ai_types) > 0:
        return ai_types
    return 'unknown'

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

# Підрахунок кількості статей за категоріями без групування по роках і збереження результатів
def count_and_save_overall(df):
    print("Підрахунок кількості статей для кожної категорії...")
    # Підрахунок типів раку
    cancer_counts = df['Cancer Type'].value_counts()
    cancer_counts_df = cancer_counts.reset_index()
    cancer_counts_df.columns = ['Cancer Type', 'Count']
    cancer_counts_df.to_excel(output_path_cancer, index=False)

    # Підрахунок моделей ШІ
    ai_model_counts = df['AI Model'].value_counts()
    ai_model_counts_df = ai_model_counts.reset_index()
    ai_model_counts_df.columns = ['AI Model', 'Count']
    ai_model_counts_df.to_excel(output_path_ai, index=False)

    # Підрахунок категорій точності
    accuracy_category_counts = df['Accuracy_Category'].value_counts()
    accuracy_category_df = accuracy_category_counts.reset_index()
    accuracy_category_df.columns = ['Accuracy_Category', 'Count']
    accuracy_category_df.to_excel(output_path_accuracy, index=False)

# Підрахунок кількості статей за категоріями по роках і збереження результатів
def count_by_year_and_save(df, column_name, output_path):
    print(f"Підрахунок статей за роками для {column_name}...")
    grouped_counts = df.groupby(['Publication Year', column_name]).size().reset_index(name='Count')
    grouped_counts.to_excel(output_path, index=False)

# Основна функція обробки файлу
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

        # Обробка типів раку
        print("Класифікация статей по типам раку...")
        with tqdm(total=len(df), desc="Типи раку") as pbar:
            df['Cancer Type'] = df.progress_apply(lambda row: categorize_cancer(row), axis=1)
            pbar.update()
        print(f"Категоризировано {len(df)} статей по типам раку.")
        
        # Обробка моделей ШІ
        print("Класифікація статей за моделями ШІ...")
        with tqdm(total=len(df), desc="Моделі ШІ") as pbar:
            df['AI Model'] = df.progress_apply(lambda row: categorize_ai_model(row), axis=1)
            pbar.update()
        print(f"Категоризовано {len(df)} статей за моделями ШІ.")
        
        # Обробка категорій точності
        print("Класифікація статей за точністю моделей...")
        with tqdm(total=len(df), desc="Точність моделей") as pbar:
            df['Accuracy_Category'] = df.progress_apply(lambda description: classify_accuracy(description), axis=1)
            pbar.update()
        print(f"Категоризовано {len(df)} статей за точністю моделей.")

        # Збереження результатів у початковий файл
        print("Збереження оновленого файлу з класифікацією...")
        df.to_excel('1-6437_categorized.xlsx', index=False)

        # Шляхи до вихідних файлів для загальних підрахунків
        global output_path_cancer, output_path_ai, output_path_accuracy
        output_path_cancer = file_path.replace('.xlsx', '_cancer_counts.xlsx')
        output_path_ai = file_path.replace('.xlsx', '_ai_model_counts.xlsx')
        output_path_accuracy = file_path.replace('.xlsx', '_accuracy_category_counts.xlsx')

        # Підрахунок та збереження загальних результатів
        count_and_save_overall(df)

        # Шляхи до вихідних файлів для підрахунків по роках
        output_path_cancer_year = file_path.replace('.xlsx', '_cancer_by_year.xlsx')
        output_path_ai_year = file_path.replace('.xlsx', '_ai_model_by_year.xlsx')
        output_path_accuracy_year = file_path.replace('.xlsx', '_accuracy_category_by_year.xlsx')

        # Підрахунок та збереження результатів по роках
        count_by_year_and_save(df, 'Cancer Type', output_path_cancer_year)
        count_by_year_and_save(df, 'AI Model', output_path_ai_year)
        count_by_year_and_save(df, 'Accuracy_Category', output_path_accuracy_year)

        print(f"File processed and saved successfully: {file_path}")
        print(f"Overall counts saved to: {output_path_cancer}, {output_path_ai}, {output_path_accuracy}")
        print(f"Yearly counts saved to: {output_path_cancer_year}, {output_path_ai_year}, {output_path_accuracy_year}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Додавання прогресу з tqdm
tqdm.pandas()

# Виклик основної функції
if __name__ == "__main__":
    path_to_excel_file = '1-6437.xlsx'
    process_excel_file(path_to_excel_file)