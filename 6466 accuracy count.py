import pandas as pd

def count_accuracy_category_and_save(file_path, output_path):
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    # Check if the necessary column is present
    if 'Accuracy_Category' not in df.columns:
        raise ValueError("Column 'Accuracy_Category' not found in the file.")
    
    # Count accuracy categories for the Accuracy_Category column
    accuracy_category_counts = df['Accuracy_Category'].value_counts()
    
    # Convert to DataFrame for saving
    accuracy_category_df = accuracy_category_counts.reset_index()
    accuracy_category_df.columns = ['Accuracy_Category', 'Count']
    
    # Save the result to an Excel file
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        accuracy_category_df.to_excel(writer, sheet_name='Accuracy Category Counts', index=False)

    return output_path

# Define file paths
file_path = r'D:\results\1-6466.xlsx'  # Path to the input Excel file
output_path = r'D:\results\accuracy_category_counts.xlsx'  # Path to the output Excel file

# Call the function
new_file_path = count_accuracy_category_and_save(file_path, output_path)

print(f"Counts were saved to: {new_file_path}")
