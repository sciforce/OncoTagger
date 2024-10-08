import pandas as pd
import logging
from tqdm import tqdm
import main_binary as mb

class ArticleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.df['number_of_cancer_types'] = self.df.get('number_of_cancer_types', 0)
        self.df['how_many_cancer_studied'] = self.df.get('how_many_cancer_studied', 'not specified')

        self.cancer_columns = mb.CancerClassifier().cancer_keywords.columns
        
        self.ai_columns = mb.CancerClassifier().ai_keywords.columns

    def count_cancer_types(self):
        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Counting cancer types"):
            count = row[self.cancer_columns].sum()
            self.df.at[index, 'number_of_cancer_types'] = count

            if count > 1:
                self.df.at[index, 'how_many_cancer_studied'] = 'various cancers'
            elif count == 1:
                cancer_type = [cancer for cancer in self.cancer_columns if row[cancer] == 1]
                if cancer_type:
                    self.df.at[index, 'how_many_cancer_studied'] = f'just one cancer - {cancer_type[0]}'
            else:
                self.df.at[index, 'how_many_cancer_studied'] = 'not specified'

    def count_ai_models(self):
        self.df['number_of_ai_models'] = self.df[self.ai_columns].sum(axis=1)

    def count_by_years(self, columns, sheet_name, output_writer):
        df_year = self.df.groupby('Publication Year')[columns].sum()
        df_year.to_excel(output_writer, sheet_name=sheet_name)

    def count_accuracy_categories(self, output_writer):
        accuracy_counts = self.df['Accuracy_Category'].value_counts()
        accuracy_counts.to_excel(output_writer, sheet_name='Accuracy Categories')

    def count_accuracy_by_year(self, output_writer):
        df_accuracy_by_year = self.df.groupby(['Publication Year', 'Accuracy_Category']).size().unstack(fill_value=0)
        df_accuracy_by_year.to_excel(output_writer, sheet_name='Accuracy by Year')

    def count_cancer_type_distribution(self, output_writer):
        cancer_distribution = self.df['how_many_cancer_studied'].value_counts()
        cancer_distribution.to_excel(output_writer, sheet_name='Cancer Type Distribution')

    def run_analysis(self):
        self.count_cancer_types()
        self.count_ai_models()
        
        output_file = self.file_path.replace('.xlsx', '_analysis.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            self.df[self.cancer_columns].sum().to_excel(writer, sheet_name='Cancer Types Frequency')
            self.df[self.ai_columns].sum().to_excel(writer, sheet_name='AI Models Frequency')
            self.count_accuracy_categories(writer)
            self.count_by_years(self.cancer_columns, 'Cancer Types by Year', writer)
            self.count_by_years(self.ai_columns, 'AI Models by Year', writer)
            self.count_accuracy_by_year(writer)
            self.count_cancer_type_distribution(writer)
        
        logging.info(f"Analysis complete. File saved: {output_file}")

if __name__ == "__main__":
    input_file = '1-13429_binary_classification.xlsx'
    analyzer = ArticleAnalyzer(input_file)
    analyzer.run_analysis()
