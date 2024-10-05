import pandas as pd
import logging
from tqdm import tqdm

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class ArticleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        self.df['number_of_cancer_types'] = 0
        self.df['various_cancers'] = 0
        self.df['not_specified'] = 0
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
        self.ai_columns = [
            'Linear/Logistic Regression', 'Decision Trees / Random Forests', 'Support Vector Machines (SVM)',
            'Convolutional Neural Networks (CNN)', 'Recurrent Neural Networks (RNN/LSTM)', 'Generative Adversarial Networks (GANs)',
            'Artificial Neural Networks (ANN)', 'Text Classification', 'Recommendation Systems', 'Genomic Models',
            'Clinical Decision Support Systems', 'Autoencoder', 'U-Net Models', 'Gradient Boosting Models',
            'Information Extraction', 'Ensemble'
        ]

   
    def count_cancer_types(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Подсчет типов рака"):
            count = row[self.cancer_columns].sum()
            self.df.at[index, 'number_of_cancer_types'] = count
            if count > 1:
                self.df.at[index, 'various_cancers'] = 1
            elif count == 0:
                self.df.at[index, 'not_specified'] = 1

    
    def count_ai_models(self):
        self.df['number_of_ai_models'] = self.df[self.ai_columns].sum(axis=1)

    
    def count_by_years(self, columns, sheet_name, output_writer):
        df_year = self.df.groupby(['Publication Year'])[columns].sum()
        df_year.to_excel(output_writer, sheet_name=sheet_name)

    
    def count_accuracy_by_year(self, output_writer):
        df_accuracy_by_year = self.df.groupby(['Publication Year', 'Accuracy_Category']).size().unstack(fill_value=0)
        df_accuracy_by_year.to_excel(output_writer, sheet_name='Accuracy by Year')

    
    def run_analysis(self):
        self.count_cancer_types()  
        self.count_ai_models() 
        
        
        output_file = self.file_path.replace('.xlsx', '_analysis.xlsx')
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            
            self.df[self.cancer_columns].sum().to_excel(writer, sheet_name='Cancer Types Frequency')
            self.df[self.ai_columns].sum().to_excel(writer, sheet_name='AI Models Frequency')
            self.count_accuracy_categories(writer)
            self.count_by_years(self.cancer_columns, 'Cancer Types by Year', writer)
            self.count_by_years(self.ai_columns, 'AI Models by Year', writer)
            self.count_by_years(['Accuracy_Category'], 'Accuracy by Year', writer)
        
        logging.info(f"Анализ завершен. Файл сохранен: {output_file}")

if __name__ == "__main__":
    input_file = '1-6437_binary_classification.xlsx'  # Можем заменить на динамический вход
    analyzer = ArticleAnalyzer(input_file)
    analyzer.run_analysis()