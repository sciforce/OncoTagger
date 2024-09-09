import pandas as pd
import spacy
import numpy as np

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Detect key words for cancer types
def categorize_cancer(row):
    keywords = {
        'breast cancer': ['breast cancer', 'breast ultrasound', 'abus', 'mammography', 'mammogram', 'bi-rads', 'ductal carcinoma', 'lobular carcinoma', 'breast biopsies'],
        'colorectal cancer': ['rectal cancer', 'colon cancer', 'colorectal cancer', 'rectum', 'sigmoid colon'],
        'prostate cancer': ['prostate cancer', 'psa', 'prostate biopsy', 'prostatic adenocarcinoma'],
        'lung cancer': ['lung cancer', 'nsclc', 'sclc', 'alk mutation', 'lung adenocarcinoma', 'iladc'],
        'brain cancer': ['brain tumor', 'glioma', 'astrocytoma', 'meningioma', 'glioblastoma'],
        'cervical cancer': ['cervical cancer', 'cervical tumor', 'hpv', 'pap-smear', 'cervical intraepithelial neoplasia'],
        'liver cancer': ['liver cancer', 'hepatic tumor', 'hcc', 'hepatocellular carcinoma', 'cirrhosis', 'hepatitis b', 'afp'],
        'stomach cancer': ['stomach cancer', 'gastric tumor', 'gastric cancer', 'gastric adenocarcinoma', 'helicobacter pylori'],
        'endometrial cancer': ['endometrial cancer', 'uterine cancer', 'endometrial carcinoma'],
        'skin cancer': ['skin cancer', 'skin tumors', 'skin tumor', 'basal cell carcinoma', 'squamous cell carcinoma'],
        'ovarian cancer': ['ovarian cancer', 'figo', 'ovarian tumors', 'ovarian tumor', 'adnexal masses'],
        'head and neck cancer': ['head and neck cancer', 'pharyngeal cancer'],
        'renal cancer': ['renal cell carcinoma', 'clear cell renal cell carcinoma', 'ccrcc', 'fuhrman grading system', 'kidney tumor'],
        'mesothelioma': ['mesothelioma', 'pleural mesothelioma'],
        'melanoma': ['melanoma', 'cutaneous melanoma', 'malignant melanoma', 'skin melanoma', 'metastatic melanoma', 'melanocytic nevus', 'BRAF mutation', 'NRAS mutation', 'KIT mutation', 'superficial spreading melanoma', 'nodular melanoma', 'acral lentiginous melanoma', 'lentigo maligna melanoma', 'amelanotic melanoma', 'melanoma in situ', 'Clark level', 'Breslow depth', 'sentinel lymph node biopsy', 'immunotherapy for melanoma', 'targeted therapy for melanoma', 'melanoma staging', 'ABCDE criteria'],
        'sarcoma': ['sarcoma', 'soft tissue sarcoma', 'osteosarcoma', 'Ewing sarcoma', 'leiomyosarcoma', 'liposarcoma', 'rhabdomyosarcoma', 'GIST', 'gastrointestinal stromal tumor', 'synovial sarcoma'],
        'oncohematologic malignancies': ['leukemia', 'lymphoma', 'multiple myeloma', 'acute myeloid leukemia', 'AML', 'chronic lymphocytic leukemia', 'CLL', 'acute lymphoblastic leukemia', 'ALL', 'Hodgkin\'s lymphoma', 'non-Hodgkin\'s lymphoma', 'plasma cell neoplasms', 'bone marrow biopsy'],
        'bladder cancer': ['bladder cancer', 'urothelial carcinoma', 'transitional cell carcinoma', 'hematuria', 'Bacillus Calmette-Guérin', 'BCG therapy', 'TURBT', 'non-muscle invasive bladder cancer', 'muscle-invasive bladder cancer', 'cystoscopy'],
        'esophageal cancer': ['esophageal cancer', 'esophageal adenocarcinoma', 'esophageal squamous cell carcinoma', 'Barrett\'s esophagus', 'esophagectomy', 'dysphagia', 'GERD', 'gastroesophageal reflux disease', 'endoscopic resection'],
        'thyroid cancer': ['thyroid cancer', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma', 'medullary thyroid carcinoma', 'anaplastic thyroid carcinoma', 'thyroidectomy', 'TSH', 'thyroid-stimulating hormone', 'thyroid nodules', 'radioiodine therapy'],
        'testicular cancer': ['testicular cancer', 'germ cell tumors', 'seminoma', 'non-seminoma', 'orchiectomy', 'alpha-fetoprotein', 'AFP', 'beta-HCG', 'testicular mass'],
        'pancreatic cancer': ['pancreatic neuroendocrine neoplasms', 'pancreatic tumor', 'pancreatic cancer', 'pancreatic neoplasm', 'pancreatic adenocarcinoma', 'pancreatic neuroendocrine tumors', 'PNET'],
        'various cancers': ['multiple cancers', 'various cancer types', 'different cancer types', 'cancer detection across multiple types', 'broad cancer diagnostic model', 'pan-cancer', 'multi-cancer detection', 'multi-cancer', 'tumor agnostic', 'common cancer markers', 'ai model for cancer diagnosis', 'deep learning for various cancer detection']
    }

    # Concatenate the text from all relevant columns into one string
    combined_text = ' '.join([
        str(row['Article Title']).lower(), 
        str(row['Author Keywords']).lower(), 
        str(row['Keywords Plus']).lower(), 
        str(row['Abstract']).lower()
    ])

    # Process the text with spaCy
    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    # Checking for specific cancer type keywords
    for cancer_type, cancer_keywords in keywords.items():
        if any(keyword in lemmatized_text for keyword in cancer_keywords):
            return cancer_type

    # If no specific cancer type is found, return 'unknown'
    return 'unknown'

# Function to load and process Excel file, then save the result
def categorize_articles_in_excel_and_save_in_same_file(file_path):
    try:
        # Open the Excel file
        df = pd.read_excel(file_path)

        # Checking if the necessary columns exist
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract']
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")

        # Handle missing data by filling with empty strings
        df[required_columns] = df[required_columns].fillna('')

        # Creating a new column 'Cancer Type' with the results of the keyword search
        df['Cancer Type'] = df.apply(categorize_cancer, axis=1)

        # Save the updated dataframe back to the same file
        df.to_excel(file_path, index=False)

        print(f"File processed and saved successfully: {file_path}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Processing a specific file 
path_to_excel_file = r'D:\results\\1-6466.xlsx'
categorize_articles_in_excel_and_save_in_same_file(path_to_excel_file)
