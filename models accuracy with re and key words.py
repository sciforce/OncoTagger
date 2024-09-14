import pandas as pd
import re

# Функція для класифікації точності на основі числових значень і ключових слів
def classify_accuracy(description):
    # Перевірка, чи значення є рядком
    if not isinstance(description, str):
        return "Unknown"

    # Регулярні вирази для діапазонів числових значень точності
    very_high_accuracy_pattern = r'\b(0\.(9[5-9]\d*)|[9][5-9]\.\d+|100)(\%)?\b'
    high_accuracy_pattern = r'\b(0\.(9[0-4]\d*)|[9][0-4]\.\d+|[9][0-4])(\%)?\b'
    medium_accuracy_pattern = r'\b(0\.(8[0-9]\d*)|[8][0-9]\.\d+|[8][0-9])(\%)?\b'
    low_accuracy_pattern = r'\b(0\.(7[0-9]\d*)|[7][0-9]\.\d+|[7][0-9])(\%)?\b'
    very_low_accuracy_pattern = r'\b(0\.(6\d+|[0-6]\.\d+|[0-6]))(\%)?\b'
    
    # Ключові слова для опису високої продуктивності та клінічного значення
    very_high_keywords = ['outstanding performance', 'clinically reliable', 'superior classification', 'exceptional accuracy']
    high_keywords = ['high accuracy', 'reliable for diagnosis', 'good prediction', 'clinically useful']
    medium_keywords = ['moderate accuracy', 'acceptable performance', 'reasonable prediction', 'risk assessment']
    low_keywords = ['low accuracy', 'requires improvement', 'preliminary assessment', 'limited clinical use']
    very_low_keywords = ['very low accuracy', 'unreliable', 'not suitable for clinical use', 'requires significant improvement']

    # Перевірка на наявність числових значень у тексті
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

# Шлях до Excel файлу
path_to_excel_file = r'D:\results\1-6466.xlsx'

# Завантажуємо Excel файл у DataFrame
df = pd.read_excel(path_to_excel_file)

# Переконайтесь, що колонка має правильну назву (наприклад, 'Abstract')
# Якщо назва колонки інша, замініть 'Abstract' на відповідну назву
df['Accuracy_Category'] = df['Abstract'].apply(classify_accuracy)

# Зберігаємо оновлений DataFrame назад в Excel
df.to_excel(path_to_excel_file, index=False)

print("Категоризація завершена, файл оновлено.")
