import pandas as pd

def remove_duplicates_by_doi(file_path):
    try:
        df = pd.read_excel(file_path)
        
        if 'DOI' not in df.columns:
            raise ValueError("Column 'DOI' not found.")
        
        # delete duplicates 'DOI'
        df_cleaned = df.drop_duplicates(subset='DOI', keep='first')
        
        # save updated file
        output_file_path = file_path.replace('.xlsx', '_cleaned.xlsx')
        df_cleaned.to_excel(output_file_path, index=False)
        
        print(f"Duplicates was deleted, file saved as: {output_file_path}")
    
    except Exception as e:
        print(f"Error while proccesing: {e}")

file_path = r'D:\results\\new_dataset.xlsx'

remove_duplicates_by_doi(file_path)
