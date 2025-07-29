import pandas as pd
import logging
from tqdm import tqdm
import main_binary as mb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ArticleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        # read the final binary file
        self.df = pd.read_excel(file_path)
        # configure logging
        logging.basicConfig(
            filename='app.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # create columns for counts if there are none
        self.df['number_of_cancer_types'] = self.df.get('number_of_cancer_types', 0)
        self.df['how_many_cancer_studied'] = self.df.get('how_many_cancer_studied', 'cancer type is not specified')

        # lists binary columns from main_binary.py
        self.cancer_columns = mb.CancerClassifier().cancer_keywords.columns
        self.ai_columns     = mb.CancerClassifier().ai_keywords.columns
        self.task_columns   = mb.CancerClassifier().task_keywords.columns

    def count_cancer_types(self):
        """Count how many types of cancer are mentioned in each article"""
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Counting cancer types"):
            count = int(row[self.cancer_columns].sum())
            self.df.at[idx, 'number_of_cancer_types'] = count

            if count > 1:
                self.df.at[idx, 'how_many_cancer_studied'] = 'various cancers'
            elif count == 1:
                cancer = next((c for c in self.cancer_columns if row[c] == 1), None)
                if cancer:
                    self.df.at[idx, 'how_many_cancer_studied'] = f'just one cancer - {cancer}'
            else:
                self.df.at[idx, 'how_many_cancer_studied'] = 'not specified'

    def count_ai_models(self):
        """Count how many AI models are used in each article"""
        self.df['number_of_ai_models'] = self.df[self.ai_columns].sum(axis=1)

    def count_task_categories(self, writer):
        """Total articles per task category"""
        freq = self.df[self.task_columns].sum()
        freq.to_excel(writer, sheet_name='Task Categories Frequency')

    def count_tasks_by_year(self, writer):
        """Dynamics of task categories by publication year"""
        df_ty = self.df.groupby('Publication Year')[self.task_columns].sum()
        df_ty.to_excel(writer, sheet_name='Tasks by Year')

    def count_by_years(self, columns, sheet_name, writer):
        """Generic: sum the specified binary columns by year"""
        df_year = self.df.groupby('Publication Year')[columns].sum()
        df_year.to_excel(writer, sheet_name=sheet_name)

    def count_cancer_type_distribution(self, writer):
        """Distribution by how_many_cancer_studied field"""
        dist = self.df['how_many_cancer_studied'].value_counts()
        dist.to_excel(writer, sheet_name='Cancer Type Distribution')

    def count_composite_totals(self, writer):
        """Total number of articles by composite_metric category"""
        self.df['composite_metric'].value_counts(dropna=False)\
               .to_excel(writer, sheet_name='Composite Total')

    def count_weighted_totals(self, writer):
        """Total number of articles by weighted_category"""
        self.df['weighted_category'].value_counts(dropna=False)\
               .to_excel(writer, sheet_name='Weighted Total')

    def count_metric_by_year(self, metric, writer):
        """Distribution of a defined metric by year"""
        df_my = (
            self.df
            .groupby(['Publication Year', metric])
            .size()
            .unstack(fill_value=0)
        )
        sheet = f'{metric.replace("_", " ").title()} by Year'
        df_my.to_excel(writer, sheet_name=sheet)
    def count_metric_by_task(self, metric, writer):
        """Distribute the given metric across tasks"""
        rows = []
        for t in self.task_columns:
            df_t = self.df[self.df[t] == 1]
            vc = df_t[metric].value_counts(dropna=False)
            rows.append(vc.rename(t))
        df_mt = pd.concat(rows, axis=1).fillna(0).astype(int)
        sheet = f'{metric.replace("_", " ").title()} by Task'
        df_mt.to_excel(writer, sheet_name=sheet)

    def crosstab_tasks_vs(self, bins, writer, sheet_name):
        ct = pd.DataFrame(index=self.task_columns, columns=bins)
        for t in self.task_columns:
            df_t = self.df[self.df[t] == 1]
            ct.loc[t] = df_t[bins].sum()
        # drop any all-zero/empty rows (shouldn’t really be necessary, but just in case)
        ct = ct.loc[ct.index.notnull()]
        ct = ct.fillna(0).astype(int)
        ct.to_excel(writer, sheet_name=sheet_name)

    def crosstab_metric_vs(self, metric, bins, writer, sheet_name):
        cats = sorted(self.df[metric].dropna().unique())
        ct = pd.DataFrame(index=cats, columns=bins)
        for c in cats:
            df_c = self.df[self.df[metric] == c]
            ct.loc[c] = df_c[bins].sum()
        # drop any unintended blank index
        ct = ct.loc[ct.index.notnull()]
        ct = ct.fillna(0).astype(int)
        ct.to_excel(writer, sheet_name=sheet_name)

    def run_analysis(self):
        # 1) Counts in rows
        self.count_cancer_types()
        self.count_ai_models()

        # 2) Open a new Excel for the results
        output_file = self.file_path.replace('.xlsx', '_analysis.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            # — 1) Task categories
            self.count_task_categories(writer)
            self.count_tasks_by_year(writer)

            # — 2) Cancer / AI frequency
            self.df[self.cancer_columns].sum().to_excel(writer, sheet_name='Cancer Types Frequency')
            self.df[self.ai_columns].sum().to_excel(writer, sheet_name='AI Models Frequency')

            # — 3) Distribution how_many_cancer_studied
            self.count_cancer_type_distribution(writer)

            # — 4) Composite & Weighted totals
            self.count_composite_totals(writer)
            self.count_weighted_totals(writer)

            # — 5) Metrics by Year
            for metric in ('composite_metric', 'weighted_category', 'roc-auc'):
                self.count_metric_by_year(metric, writer)

            # — 6) Metrics by Task
            for metric in ('composite_metric', 'weighted_category', 'roc-auc'):
                self.count_metric_by_task(metric, writer)

            # — 7) Crosstabs: Task × Cancer / AI
            self.crosstab_tasks_vs(self.cancer_columns, writer, 'Task x Cancer')
            self.crosstab_tasks_vs(self.ai_columns,     writer, 'Task x AI Models')

            # — 8) Crosstabs: Metric × Cancer / AI
            self.crosstab_metric_vs('composite_metric',  self.cancer_columns, writer, 'Composite x Cancer')
            self.crosstab_metric_vs('composite_metric',  self.ai_columns,     writer, 'Composite x AI')
            self.crosstab_metric_vs('weighted_category', self.cancer_columns, writer, 'Weighted x Cancer')
            self.crosstab_metric_vs('weighted_category', self.ai_columns,     writer, 'Weighted x AI')
            self.crosstab_metric_vs('roc-auc',           self.cancer_columns, writer, 'ROC-AUC x Cancer')
            self.crosstab_metric_vs('roc-auc',           self.ai_columns,     writer, 'ROC-AUC x AI')

        logging.info(f"Analysis complete. File saved: {output_file}")


if __name__ == "__main__":
    input_file = 'filtered_dataset_binary_classification.xlsx'
    analyzer = ArticleAnalyzer(input_file)
    analyzer.run_analysis()
