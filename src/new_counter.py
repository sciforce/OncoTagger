import pandas as pd
import logging
from tqdm import tqdm
import main_binary as mb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import pycountry

script_dir   = Path(__file__).parent.resolve()
project_root = script_dir.parent
sources_dir  = project_root / 'sources'
results_dir  = project_root / 'data' / 'results'

def extract_country(address: str) -> str:
    if pd.isna(address):
        return None
    return address.split()[-1].rstrip('.,;')

df_cn = pd.read_csv(sources_dir / 'country_synonyms.csv')
country_synonyms = dict(zip(df_cn['raw'], df_cn['normalized']))

def normalize_country(raw: str) -> str:
    name = raw.strip()
    tokens = [tok for tok in name.split() if not any(ch.isdigit() for ch in tok)]
    name_clean = ' '.join(tokens)
    key = name_clean.title()
    if key in country_synonyms:
        return country_synonyms[key]
    try:
        return pycountry.countries.lookup(key).name
    except LookupError:
        return key

class ArticleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        
        logging.basicConfig(
            filename='app.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        self.df['number_of_cancer_types'] = self.df.get('number_of_cancer_types', 0)
        self.df['how_many_cancer_studied'] = self.df.get('how_many_cancer_studied', 'cancer type is not specified')

        classifier = mb.CancerClassifier()
        self.cancer_columns = classifier.cancer_keywords.columns
        self.ai_columns     = classifier.ai_keywords.columns
        self.task_columns   = classifier.task_keywords.columns
        
        self.metric_order = ['very high', 'high', 'medium', 'low', 'very low', 'no metrics reported']

        # Нормалізація колонок з метриками для уникнення конфліктів регістру
        for m in ['composite_metric', 'weighted_category', 'roc-auc']:
            if m in self.df.columns:
                self.df[m] = self.df[m].fillna('no metrics reported').astype(str).str.lower().str.strip()

    def count_cancer_types(self):
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Counting cancer types..."):
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
        self.df['number_of_ai_models'] = self.df[self.ai_columns].sum(axis=1)

    def count_countries(self, writer):
        if 'Reprint Addresses' not in self.df.columns:
            return
        raw = self.df['Reprint Addresses'].dropna().apply(extract_country).dropna()
        mapped = raw.map(lambda t: country_synonyms.get(t, t))
        df_countries = mapped.value_counts().reset_index()
        df_countries.columns = ['Country', 'Count']
        df_countries.to_excel(writer, sheet_name='Country Counts', index=False)

    def count_task_categories(self, writer):
        df_tasks = self.df[self.task_columns].sum().sort_values(ascending=False).reset_index()
        df_tasks.columns = ['Task Category', 'Count']
        df_tasks.to_excel(writer, sheet_name='Task Categories Frequency', index=False)

    def count_frequency(self, columns, sheet_name, col_header, writer):
        df_freq = self.df[columns].sum().sort_values(ascending=False).reset_index()
        df_freq.columns = [col_header, 'Count']
        df_freq.to_excel(writer, sheet_name=sheet_name, index=False)

    def count_cancer_type_distribution(self, writer):
        df_dist = self.df['how_many_cancer_studied'].value_counts().reset_index()
        df_dist.columns = ['Cancer Study Type', 'Count']
        df_dist.to_excel(writer, sheet_name='Cancer Type Distribution', index=False)

    def count_ordered_metric_totals(self, col_name, sheet_name, writer):
        if col_name not in self.df.columns:
            return
        vc = self.df[col_name].value_counts(dropna=False)
        df_ordered = vc.reindex(self.metric_order).fillna(0).astype(int).reset_index()
        df_ordered.columns = [sheet_name, 'Count']
        df_ordered.to_excel(writer, sheet_name=sheet_name, index=False)

    def count_tasks_by_year(self, writer):
        df_ty = self.df.groupby('Publication Year')[self.task_columns].sum()
        df_ty.index.name = 'Publication Year'
        df_ty.to_excel(writer, sheet_name='Tasks by Year')

    def count_by_years(self, columns, sheet_name, writer):
        df_year = self.df.groupby('Publication Year')[columns].sum()
        df_year.index.name = 'Publication Year'
        df_year.to_excel(writer, sheet_name=sheet_name)

    def count_metric_by_year(self, metric, writer):
        if metric not in self.df.columns: return
        df_my = self.df.groupby(['Publication Year', metric]).size().unstack(fill_value=0)
        
        existing_cols = [c for c in self.metric_order if c in df_my.columns]
        df_my = df_my[existing_cols]
        
        sheet = f'{metric.replace("_", " ").title()} by Year'
        df_my.index.name = 'Publication Year'
        df_my.to_excel(writer, sheet_name=sheet)

    def count_metric_by_task(self, metric, writer):
        if metric not in self.df.columns: return
        rows = []
        for t in self.task_columns:
            df_t = self.df[self.df[t] == 1]
            vc = df_t[metric].value_counts(dropna=False)
            rows.append(vc.rename(t))
        
        df_mt = pd.concat(rows, axis=1).fillna(0).astype(int)
        existing_idx = [idx for idx in self.metric_order if idx in df_mt.index]
        df_mt = df_mt.reindex(existing_idx).fillna(0).astype(int)
        
        sheet = f'{metric.replace("_", " ").title()} by Task'
        df_mt.index.name = metric.replace('_', ' ').title()
        df_mt.to_excel(writer, sheet_name=sheet)

    def crosstab_tasks_vs(self, bins, writer, sheet_name):
        ct = pd.DataFrame(index=self.task_columns, columns=bins)
        for t in self.task_columns:
            df_t = self.df[self.df[t] == 1]
            ct.loc[t] = df_t[bins].sum()
        ct = ct.loc[ct.index.notnull()].fillna(0).astype(int)
        ct.index.name = 'Task Category'
        ct.to_excel(writer, sheet_name=sheet_name)

    def crosstab_metric_vs(self, metric, bins, writer, sheet_name):
        if metric not in self.df.columns: return
        cats = [c for c in self.metric_order if c in self.df[metric].dropna().unique()]
        ct = pd.DataFrame(index=cats, columns=bins)
        for c in cats:
            df_c = self.df[self.df[metric] == c]
            ct.loc[c] = df_c[bins].sum()
        ct = ct.loc[ct.index.notnull()].fillna(0).astype(int)
        ct.index.name = metric.replace('_', ' ').title()
        ct.to_excel(writer, sheet_name=sheet_name)

    def run_analysis(self):
        self.count_cancer_types()
        self.count_ai_models()

        output_file = self.file_path.replace('.xlsx', '_analysis.xlsx')
        with pd.ExcelWriter(output_file) as writer:
            
            self.count_countries(writer)
            self.count_task_categories(writer)
            
            self.count_frequency(self.cancer_columns, 'Cancer Types Frequency', 'Cancer Type', writer)
            self.count_frequency(self.ai_columns, 'AI Models Frequency', 'AI Model', writer)

            self.count_cancer_type_distribution(writer)

            self.count_ordered_metric_totals('composite_metric', 'Composite Total', writer)
            self.count_ordered_metric_totals('weighted_category', 'Weighted Total', writer)

            for metric in ('composite_metric', 'weighted_category', 'roc-auc'):
                self.count_metric_by_year(metric, writer)
                self.count_metric_by_task(metric, writer)

            self.crosstab_tasks_vs(self.cancer_columns, writer, 'Task x Cancer')
            self.crosstab_tasks_vs(self.ai_columns,     writer, 'Task x AI Models')

            self.crosstab_metric_vs('composite_metric',  self.cancer_columns, writer, 'Composite x Cancer')
            self.crosstab_metric_vs('composite_metric',  self.ai_columns,     writer, 'Composite x AI')
            self.crosstab_metric_vs('weighted_category', self.cancer_columns, writer, 'Weighted x Cancer')
            self.crosstab_metric_vs('weighted_category', self.ai_columns,     writer, 'Weighted x AI')
            self.crosstab_metric_vs('roc-auc',           self.cancer_columns, writer, 'ROC-AUC x Cancer')
            self.crosstab_metric_vs('roc-auc',           self.ai_columns,     writer, 'ROC-AUC x AI')
            
            self.count_tasks_by_year(writer)
            self.count_by_years(self.cancer_columns, 'Cancer Types by Year', writer)
            self.count_by_years(self.ai_columns,     'AI Models by Year',   writer)

            top10_cancers = self.df[self.cancer_columns].sum().nlargest(10).index.tolist()
            self.count_by_years(top10_cancers, 'Top-10 Cancers by Year', writer)

            top10_models = self.df[self.ai_columns].sum().nlargest(10).index.tolist()
            self.count_by_years(top10_models, 'Top-10 AI Models by Year', writer)

        logging.info(f"Analysis complete. File saved: {output_file}")

if __name__ == "__main__":
    input_file = str(results_dir / 'filtered_dataset_binary_classification.xlsx')
    analyzer = ArticleAnalyzer(input_file)
    analyzer.run_analysis()