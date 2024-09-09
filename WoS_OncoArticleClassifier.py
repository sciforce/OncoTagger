import pandas as pd
import spacy

# load spaCy model
nlp = spacy.load('en_core_web_sm')

# detect key words for cancer types
def categorize_cancer(description):
    keywords = {
        'breast cancer': ['breast cancer', 'mammogram', 'BI-RADS', 'HER2', 'ductal carcinoma', 'lobular carcinoma'],
        'rectal cancer': ['rectal cancer', 'colon cancer', 'colorectal cancer', 'rectum', 'sigmoid colon'],
        'prostate cancer': ['prostate cancer', 'PSA', 'prostate biopsy', 'prostatic adenocarcinoma'],
        'lung cancer': ['lung cancer', 'NSCLC', 'SCLC', 'ALK mutation'],
        'brain cancer': ['brain tumor', 'glioma', 'astrocytoma', 'meningioma', 'glioblastoma'],
        'cervical cancer': ['cervical cancer', 'HPV', 'Pap-smear', 'cervical intraepithelial neoplasia'],
        'liver cancer': ['liver cancer', 'HCC', 'hepatocellular carcinoma', 'cirrhosis', 'hepatitis B', 'AFP'],
        'stomach cancer': ['stomach cancer', 'gastric cancer', 'gastric adenocarcinoma', 'Helicobacter pylori'],
        'endometrial cancer': ['endometrial cancer', 'uterine cancer', 'endometrial carcinoma', 'hyperplasia'],
        'skin cancer': ['skin cancer', 'melanoma', 'basal cell carcinoma', 'squamous cell carcinoma']
    }

    # Text processing with spaCy
    doc = nlp(description.lower())
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    for cancer_type, cancer_keywords in keywords.items():
        for keyword in cancer_keywords:
            if keyword.lower() in lemmatized_text:
                return cancer_type
    return 'unknown'

# Loading Excel file and adding new column with results
def categorize_articles_in_excel_and_save_in_same_file(file_path):
    # open Excel file
    df = pd.read_excel(file_path)

    # Checking if there is a description column
    if 'abstract' not in df.columns:
        raise ValueError("Column 'abstract' not found in the Excel file.")

    # Creating a new column for cancer types
    df['Cancer Type'] = df['abstract'].apply(categorize_cancer)

    # Save to the same file with the updated column
    df.to_excel(file_path, index=False)

    return file_path

# Processing a specific file 
path_to_excel_file = r'D:\results\\1-100_test_file.xlsx'
categorize_articles_in_excel_and_save_in_same_file(path_to_excel_file)