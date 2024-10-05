import pandas as pd
import logging
from tqdm import tqdm

# Налаштування рівня логування
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Основний клас для аналізу
class ArticleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        # Додаємо нові стовпці для підрахунку типів раку і моделей ІІ
        self.df['number_of_cancer_types'] = 0
        self.df['various_cancers'] = 0
        self.df['not_specified'] = 0
        # Список стовпців для підрахунку типів раку
        self.cancer_columns = [
            'breast cancer', 'colorectal cancer', 'prostate cancer', 'lung cancer',
            'brain cancer', 'cervical cancer', 'liver cancer', 'stomach cancer',
            'endometrial cancer', 'skin cancer', 'ovarian cancer', 'head and neck cancer',
            'renal cancer', 'mesothelioma', 'parathyroid cancer', 'gallbladder cancer',
            'occult primary cancer', 'vaginal cancer', 'vulvar cancer', 'penile cancer',
            'neuroendocrine tumors', 'mediastinal tumors', 'bone cancers', 'melanoma',
            'sarcoma', 'oncohematologic malignancies', 'bladder cancer', 'esophageal cancer',
            'thyroid cancer', 'testicular cancer', 'pancreatic cancer', 'various cancers'
        ]
        # Список стовпців для підрахунку моделей ІІ
        self.ai_columns = [
            'Linear/Logistic Regression', 'Decision Trees / Random Forests', 'Support Vector Machines (SVM)',
            'Convolutional Neural Networks (CNN)', 'Recurrent Neural Networks (RNN/LSTM)', 'Generative Adversarial Networks (GANs)',
            'Artificial Neural Networks (ANN)', 'Text Classification', 'Recommendation Systems', 'Genomic Models',
            'Clinical Decision Support Systems', 'Autoencoder', 'U-Net Models', 'Gradient Boosting Models',
            'Information Extraction', 'Ensemble'
        ]

    # Функція для підрахунку типів раку і створення міток
    def count_cancer_types(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Підрахунок типів раку"):
            count = row[self.cancer_columns].sum()
            self.df.at[index, 'number_of_cancer_types'] = count
            if count > 1:
                self.df.at[index, 'various_cancers'] = 1
            elif count == 0:
                self.df.at[index, 'not_specified'] = 1

    # Функція для підрахунку моделей ІІ
    def count_ai_models(self):
        self.df['number_of_ai_models'] = self.df[self.ai_columns].sum(axis=1)

    # Функція для підрахунку даних за роками
    def count_by_years(self, columns, sheet_name, output_writer):
        df_year = self.df.groupby(['Publication Year'])[columns].sum()
        df_year.to_excel(output_writer, sheet_name=sheet_name)

    # Функція для підрахунку категорій точності
    def count_accuracy_categories(self, output_writer):
        accuracy_counts = self.df['Accuracy_Category'].value_counts()
        accuracy_counts.to_excel(output_writer, sheet_name='Accuracy Categories')

    # Функція для підрахунку категорій точності за роками
    def count_accuracy_by_year(self, output_writer):
        df_accuracy_by_year = self.df.groupby(['Publication Year', 'Accuracy_Category']).size().unstack(fill_value=0)
        df_accuracy_by_year.to_excel(output_writer, sheet_name='Accuracy by Year')

    # Основна функція для запуску аналізу
    def run_analysis(self):
        self.count_cancer_types()  # Підрахунок типів раку і створення міток
        self.count_ai_models()  # Підрахунок моделей ІІ
        
        # Створення вихідного файлу з кількома листами
        output_file = self.file_path.replace('.xlsx', '_analysis.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            # 1. Лист з частотою згадуваності типів раку
            self.df[self.cancer_columns].sum().to_excel(writer, sheet_name='Cancer Types Frequency')
            # 2. Лист з частотою моделей ІІ
            self.df[self.ai_columns].sum().to_excel(writer, sheet_name='AI Models Frequency')
            # 3. Лист з категоріями точності
            self.count_accuracy_categories(writer)
            # 4. Лист з розподілом типів раку за роками
            self.count_by_years(self.cancer_columns, 'Cancer Types by Year', writer)
            # 5. Лист з розподілом моделей ІІ за роками
            self.count_by_years(self.ai_columns, 'AI Models by Year', writer)
            # 6. Лист з розподілом категорій точності за роками
            self.count_accuracy_by_year(writer)
        
        logging.info(f"Аналіз завершено. Файл збережено: {output_file}")


# Запуск аналізу
if __name__ == "__main__":
    input_file = '1-13429_binary_classification.xlsx'  # Використовуємо правильний вхідний файл
    analyzer = ArticleAnalyzer(input_file)
    analyzer.run_analysis()
