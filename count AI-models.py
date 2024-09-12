import pandas as pd

def count_ai_models_and_save(file_path, output_path):
    
    df = pd.read_excel(file_path)
    
    
    if 'AI Model' not in df.columns:
        raise ValueError("Колонка 'AI Model' не найдена в файле.")
    
    
    ai_model_counts = df['AI Model'].value_counts()

    
    ai_model_counts_df = ai_model_counts.reset_index()
    ai_model_counts_df.columns = ['AI Model', 'Count']
    
    
    ai_model_counts_df.to_excel(output_path, index=False)

    return output_path

file_path = r'D:\results\\1-6466.xlsx'
output_path = r'D:\results\ai_model_counts.xlsx'

new_file_path = count_ai_models_and_save(file_path, output_path)

print(f"Counts were saved to: {new_file_path}")
