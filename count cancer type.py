import pandas as pd

def count_cancer_types_and_save(file_path, output_path):
 
    df = pd.read_excel(file_path)
    
    
    if 'Cancer Type' not in df.columns:
        raise ValueError("Колонка 'Cancer Type' не найдена в файле.")
    
 
    cancer_counts = df['Cancer Type'].value_counts()

    
    cancer_counts_df = cancer_counts.reset_index()
    cancer_counts_df.columns = ['Cancer Type', 'Count']
    
   
    cancer_counts_df.to_excel(output_path, index=False)

    return output_path


file_path = r'D:\results\\1-6466.xlsx'
output_path = r'D:\results\cancer_type_counts.xlsx'


new_file_path = count_cancer_types_and_save(file_path, output_path)

print(f"Counts were saved to: {new_file_path}")
