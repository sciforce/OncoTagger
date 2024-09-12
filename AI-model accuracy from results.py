import pandas as pd
import re

# Function to classify AI model accuracy based on ranges
def categorize_accuracy(abstract):
    # Convert abstract to string in case it's not
    if not isinstance(abstract, str):
        abstract = str(abstract)
    
    # Regular expression to find both floating point numbers and percentages
    pattern = r'(\d{1,2}\.\d{1,3}|\d{2,3})(%?)'
    
    # Find all matches in the text
    accuracies = re.findall(pattern, abstract)
    
    # Convert all found values to percentages if necessary
    accuracy_values = []
    for acc in accuracies:
        value, percent = acc
        value = float(value)
        if percent:  # If the value is a percentage
            value = value / 100
        accuracy_values.append(value)
    
    # Classify based on accuracy ranges
    for value in accuracy_values:
        if 0.95 <= value <= 1.00:
            return 'High accuracy (≥ 95%)'
        elif 0.90 <= value < 0.95:
            return 'Moderate-high accuracy (90% - 94.9%)'
        elif 0.80 <= value < 0.90:
            return 'Moderate accuracy (80% - 89.9%)'
        elif 0.70 <= value < 0.80:
            return 'Low accuracy (70% - 79.9%)'
        elif value < 0.70:
            return 'Very low accuracy (< 70%)'
    
    return 'Accuracy not found'

# Function to load and process the Excel file, then save the results
def categorize_articles_accuracy_in_excel_and_save(file_path):
    try:
        # Open the Excel file
        df = pd.read_excel(file_path)

        # Check if the 'Abstract' column is present
        if 'Abstract' not in df.columns:
            raise ValueError("The 'Abstract' column is missing in the Excel file")

        # Handle missing data by filling with empty strings
        df['Abstract'] = df['Abstract'].fillna('')

        # Apply the accuracy categorization function to each row
        df['Accuracy Category'] = df['Abstract'].apply(categorize_accuracy)

        # Save the updated file back to the same location
        df.to_excel(file_path, index=False)

        print(f"File processed and saved successfully: {file_path}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Path to the Excel file
path_to_excel_file = r'D:\results\\1-6466.xlsx'

# Run the file processing function
categorize_articles_accuracy_in_excel_and_save(path_to_excel_file)
