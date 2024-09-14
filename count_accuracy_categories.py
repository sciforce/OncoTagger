import pandas as pd

def count_accuracy_categories_and_save_test(file_path, output_path):
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Check if the necessary columns are present
    if 'auto accuracy' not in df.columns or 'manual accuracy' not in df.columns:
        raise ValueError("Columns 'auto accuracy' and/or 'manual accuracy' not found in the file.")
    
    # Count accuracy categories for auto accuracy
    auto_accuracy_counts = df['auto accuracy'].value_counts()
    
    # Count accuracy categories for manual accuracy
    manual_accuracy_counts = df['manual accuracy'].value_counts()
    
    # Convert to DataFrame for saving
    auto_accuracy_df = auto_accuracy_counts.reset_index()
    auto_accuracy_df.columns = ['auto accuracy', 'Count']
    
    manual_accuracy_df = manual_accuracy_counts.reset_index()
    manual_accuracy_df.columns = ['manual accuracy', 'Count']
    
    # Save the results into separate sheets in an Excel file
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        auto_accuracy_df.to_excel(writer, sheet_name='Auto Accuracy Counts', index=False)
        manual_accuracy_df.to_excel(writer, sheet_name='Manual Accuracy Counts', index=False)

    return output_path

# Define file paths
file_path = r'D:\results\top_100 _without_unknown_and_retracted.xlsx'
output_path = r'D:\results\accuracy_counts.xlsx'

# Call the function
new_file_path = count_accuracy_categories_and_save_test(file_path, output_path)

print(f"Counts were saved to: {new_file_path}")
