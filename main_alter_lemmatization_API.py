import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import requests
import os
import re

json_url = 'https://drive.google.com/uc?export=download&id=1i50BOBdypzWtMj6gcTxfJXg9iFyLDUWb'
json_filename = 'oncoclassifier-cace8ffc29fb.json'

def download_json(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File {filename} downloaded successfully.")
    else:
        print(f"Download error: {response.status_code}")

def connect_to_google_sheets(sheet_id, worksheet_index=0):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('path_to_your_credentials_file.json', scope)
    client = gspread.authorize(creds)

    # Відкриваємо Google Sheets по ID
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(worksheet_index)
    
    # Завантажуємо данні в DataFrame
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)
    
    return df, worksheet

# Оновлення даних знову в Google Sheets
def update_google_sheets(worksheet, df):
    worksheet.update([df.columns.values.tolist()] + df.values.tolist())

# Функція оброблення даних
def process_google_sheet(sheet_id):
    # Підключаємось до Google Sheets і завантажуємо дані
    df, worksheet = connect_to_google_sheets(sheet_id)

# Завантажуємо модель spaCy
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Функція для видалення апострофів у тексті (наприклад, "barrett's" -> "barretts")
def preprocess_text_smart(text):
    # Видаляє апострофи тільки всередині слів, залишаючи цілі терміни
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)
    return text

keywords_with_hyphens = [ 'triple-negative', 'non-small', 'microsatellite-instability', 'androgen-dependent', 'castration-resistant', 'non-muscle-invasive', 'muscle-invasive', 'signet-ring', 'serous-endometrial', 'low-grade', 'high-grade', 't1-weighted', 't2-weighted', 'gray-level', 'multi-gray', 'diffuse-intrinsic', 'small-cell', 'large-cell', 'endometrioid-type', 'neuroendocrine-tumors', 'cutaneus-squamous-cell', 'basal-cell', 'clear-cell', 'low-grade', 'non-seminoma', 'solid-pseudopapillary', 'pan-cancer', 'multi-cancer', 'extra-adrenal', 'catecholamine-secreting', 'gastrointestinal-stromal', 'low-grade-myofibroblastic', 'low-grade-fibromyxoid', 'smooth-muscle', 'soft-tissue', 'epithelioid-sarcoma', 'chronic-lymphocytic', 'acute-lymphoblastic', 'multiple-myeloma', 'stage-iii', 'stage-iv', 'barrett-esophagus', 'ck5/6', 'vascular-endothelial', 'endometrial-stromal', 'non-hodgkin', 'hodgkin-lymphoma', 'superficial-spreading', 'sentinel-lymph-node', 'abcde-criteria', 'breast-ductal-carcinoma', 'ductal-carcinoma', 'artificial-neural-network', 'convolutional-neural-network', 'long-short-term', 'large-cell-neuroendocrine', 'support-vector', 'data-driven', 'non-hodgkin-lymphoma', 'multi-cancer-type', 'organs-at-risk', 'wide-area' ]

for keyword in keywords_with_hyphens:
    parts = keyword.split('-')
    
    # Провіряємо, що parts містить как мінімум два елементи
    if len(parts) == 2:
        pattern1 = [{'LOWER': keyword.replace('-', '')}]  # case without hyphen
        pattern2 = [{'LOWER': parts[0]}, {'LOWER': parts[1]}]  # case with space
        pattern3 = [{'LOWER': parts[0]}, {'IS_PUNCT': True}, {'LOWER': parts[1]}]  # case with hyphen or other punctuation
        matcher.add(keyword, [pattern1, pattern2, pattern3])
    else:
        # якщо дефіса немає, додаємо тільки шаблон без дефіса
        pattern1 = [{'LOWER': keyword}]  # Оригинальная строка
        matcher.add(keyword, [pattern1])

# Функція для класифікації статей за типом раку
keywords = {
    'breast cancer': ['breast cancer', 'breast carcinoma', 'invasive ductal carcinoma', 'breast carcinomas', 'breast adenocarcinoma', 'breast adenocarcinomas', 'breast malignancy', 'breast malignancies', 'tnbc',  'triple-negative breast cancer', 'invasive breast carcinoma', 'breast ductal carcinoma in situ', 'invasive lobular carcinoma', 'breast ultrasound', 'mammograms', 'breast masses', 'breast tumor', 'breast tumors', 'mcf-7', 'breast neoplasm', 'breast neoplasms', 'abus', 'mammography', 'mammogram', 'bi-rads', 'lobular carcinoma', 'breast biopsies', 'breast biopsy'],
    'colorectal cancer': ['colorectal cancer', 'colorectal carcinoma', 'colorectal carcinomas', 'msi-h', 'colorectal carcinoma', 'colorectal carcinomas', 'colorectal adenocarcinoma', 'colorectal adenocarcinomas', 'colorectal malignancy', 'colon carcinoma', 'colon carcinomas', 'colon adenocarcinoma', 'colon adenocarcinomas', 'colon malignancy', 'colon malignancies', 'colorectal malignancies', 'microsatellite instability', 'msi', 'colorectal cancers', 'colon cancer', 'colon cancers', 'rectal cancer', 'rectal cancers', 'colon tumor', 'colon tumors', 'rectal tumor', 'rectal tumors', 'colon neoplasm', 'colon neoplasms', 'rectal neoplasm', 'rectal neoplasms', 'adenocarcinoma of the colon', 'adenocarcinoma of the rectum', 'familial adenomatous polyposis', 'lynch syndrome', 'rectum', 'sigmoid colon', 'anal cancer', 'anal cancers', 'anal tumor', 'anal tumors', 'anal neoplasm', 'anal neoplasms', 'squamous cell carcinoma of the anus', 'anal squamous cell carcinoma', 'anorectal cancer', 'anorectal cancers', 'anorectal tumor', 'anorectal tumors', 'anorectal neoplasm', 'anorectal neoplasms', 'perianal cancer', 'perianal cancers', 'perianal tumor', 'perianal tumors', 'perianal neoplasm', 'perianal neoplasms'],
    'prostate cancer': ['prostate cancer', 'prostate carcinoma', 'tmprss2-erg', 'prostate cancers', 'prostate tumor', 'prostate tumors', 'prostate neoplasm', 'prostate neoplasms', 'prostatic adenocarcinoma', 'gleason score', 'androgen-dependent prostate cancer', 'benign prostatic hyperplasia', 'abiraterone', 'enzalutamide', 'prostate carcinomas', 'prostate adenocarcinoma', 'prostate adenocarcinomas', 'prostate malignancy', 'prostate malignancies', 'apalutamide', 'castration-resistant', 'psa', 'prostate biopsy', 'prostatic adenocarcinoma'],
    'lung cancer': ['lung cancer', 'ros1', 'thoracic cancers', 'lung cancers', 'lung tumor', 'lung tumors', 'lung neoplasm', 'lung neoplasms', 'non-small cell lung cancer', 'non-small cell lung cancers', 'small cell lung cancer', 'small cell lung cancers', 'bronchoalveolar carcinoma', 'lung biopsy', 'bronchoscopy', 'videobronchoscopy', 'lung squamous cell carcinoma', 'large cell lung carcinoma', 'lung carcinoma', 'lung carcinomas', 'lung adenocarcinoma', 'lung adenocarcinomas', 'lung malignancy', 'lung malignancies', 'pulmonary carcinoma', 'pulmonary carcinomas', 'pulmonary adenocarcinoma', 'pulmonary adenocarcinomas', 'pulmonary malignancy', 'pulmonary malignancies', 'nsclc', 'sclc', 'alk mutation', 'lung adenocarcinoma', 'iladc'],
    'brain cancer': ['posterior fossa tumors', 'retinoblastoma', 'central nervous system', 'cns', 'brain image', 'brain images', 'schwannomas', 'schwannoma', 'glial fibrillary acidic protein', 'gfap', 'anterior fossa tumors', 'brain cancer', 'brain cancers', 'brain tumor', 'brain tumors', 'brain neoplasm', 'brain neoplasms', 'glioma', 'gliomas', 'astrocytoma', 'glioblastoma multiforme', 'ependymoma', 'oligodendroglioma', 'brain carcinoma', 'brain carcinomas', 'brain adenocarcinoma', 'brain adenocarcinomas', 'brain malignancy', 'brain malignancies', 'cerebral carcinoma', 'cerebral carcinomas', 'cerebral adenocarcinoma', 'cerebral adenocarcinomas', 'cerebral malignancy', 'cerebral malignancies', 'anaplastic astrocytoma', 'medulloblastoma', 'diffuse intrinsic pontine glioma', 'astrocytomas', 'meningioma', 'glioblastoma', 'cerebellum cancer', 'cerebellum cancers', 'cerebellum tumor', 'cerebellum tumors', 'cerebellum neoplasm', 'cerebellum neoplasms', 'corpus callosum cancer', 'corpus callosum cancers', 'corpus callosum tumor', 'corpus callosum tumors', 'corpus callosum neoplasm', 'corpus callosum neoplasms', 'pituitary cancer', 'pituitary cancers', 'pituitary tumor', 'pituitary tumors', 'pituitary neoplasm', 'pituitary neoplasms', 'hypothalamus cancer', 'hypothalamus cancers', 'hypothalamus tumor', 'hypothalamus tumors', 'hypothalamus neoplasm', 'hypothalamus neoplasms', 'brainstem cancer', 'brainstem cancers', 'brainstem tumor', 'brainstem tumors', 'brainstem neoplasm', 'brainstem neoplasms', 'medulla oblongata cancer', 'medulla oblongata cancers', 'medulla oblongata tumor', 'medulla oblongata tumors', 'medulla oblongata neoplasm', 'medulla oblongata neoplasms'],
    'cervical cancer': ['cervical carcinoma', 'cervical carcinomas', 'cervical adenocarcinoma', 'cervical adenocarcinomas', 'cervical malignancy', 'cervical malignancies', 'cervical cancer', 'cervical cancers', 'cervical tumor', 'cervical tumors', 'cervical neoplasm', 'cervical neoplasms', 'hela', 'hpv', 'pap-smear', 'cervical intraepithelial neoplasia'],
    'liver cancer': ['liver cancer', 'liver cancers','liver carcinoma', 'liver carcinomas', 'liver adenocarcinoma', 'hepatic carcinoma', 'hepatic carcinomas', 'hepatic adenocarcinoma', 'hepatic adenocarcinomas', 'hepatic malignancy', 'hepatic malignancies', 'liver adenocarcinomas', 'liver malignancy', 'liver malignancies', 'liver tumor', 'liver tumors', 'liver neoplasm', 'liver neoplasms', 'hepatocellular carcinoma', 'hepatic adenoma', 'fibrolamellar carcinoma', 'hepatic hemangioma', 'cholangiocarcinoma', 'intrahepatic cholangiocarcinoma', 'hcc', 'hepatocellular carcinoma', 'cirrhosis', 'hepatitis b', 'afp', 'cholangioscopy'],
    'stomach cancer': ['stomach cancer', 'gastrointestinal tumor', 'gastrin', 'stomach cancers', 'gastric cancer', 'gastric cancers', 'stomach tumor', 'stomach tumors', 'gastric tumor', 'gastric tumors', 'stomach neoplasm', 'stomach carcinoma', 'stomach carcinomas', 'stomach adenocarcinoma', 'stomach adenocarcinomas', 'stomach malignancy', 'stomach malignancies', 'gastric carcinoma', 'gastric carcinomas', 'gastric adenocarcinoma', 'gastric adenocarcinomas', 'gastric malignancy', 'gastric malignancies', 'stomach neoplasms', 'gastric adenocarcinoma', 'linitis plastica', 'signet ring cell carcinoma', 'mucosa-associated lymphoid tissue lymphoma', 'gastric neuroendocrine tumor', 'upper gastrointestinal tract carcinoma', 'gastrointestinal cancer', 'gastric adenocarcinoma', 'helicobacter pylori'],
    'endometrial cancer': ['endometrial cancer', 'endometrioid', 'endometrial cancers', 'endometrial tumor', 'endometrial tumors', 'endometrial neoplasm', 'endometrial neoplasms', 'uterine cancer', 'uterine cancers', 'uterine tumor', 'uterine tumors', 'uterine neoplasm', 'uterine neoplasms', 'endometrial carcinoma', 'endometrial carcinomas', 'uterine carcinoma', 'uterine carcinomas', 'endometrial carcinoma', 'endometrial carcinomas', 'endometrial adenocarcinoma', 'endometrial adenocarcinomas', 'endometrial malignancy', 'endometrial malignancies', 'uterine adenocarcinoma', 'uterine adenocarcinomas', 'endometrioid cancer', 'endometrioid cancers', 'endometrioid tumor', 'endometrioid tumors', 'endometrioid neoplasm', 'endometrioid neoplasms', 'serous endometrial cancer', 'serous endometrial cancers', 'serous endometrial tumor', 'serous endometrial tumors', 'serous endometrial neoplasm', 'serous endometrial neoplasms'],
    'skin cancer': ['keratoacanthoma', 'skin cancer', 'skin cancers', 'skin tumor', 'hand tumors', 'skin tumors', 'skin neoplasm', 'cutaneous squamous cell carcinoma', 'skin neoplasms', 'basal cell carcinoma', 'skin carcinoma', 'skin carcinomas', 'skin adenocarcinoma', 'skin adenocarcinomas', 'skin malignancy', 'skin malignancies'],
    'ovarian cancer': ['ovarian cancer', 'ca125', 'pax8', 'ovarian cancers', 'ovarian tumor', 'ovarian tumors', 'ovarian neoplasm', 'ovarian neoplasms', 'serous ovarian carcinoma', 'endometrioid ovarian carcinoma', 'clear cell ovarian carcinoma', 'ovarian carcinoma', 'ovarian carcinomas', 'ovarian adenocarcinoma', 'ovarian adenocarcinomas', 'ovarian malignancy', 'ovarian malignancies', 'mucinous ovarian carcinoma', 'fallopian tube', 'low-grade serous ovarian carcinoma', 'granulosa cell tumor', 'dysgerminoma', 'ovarian germ cell tumor', 'figo', 'adnexal masses'],
    'head and neck cancer': ['neck tumor', 'jaws', 'dental cyst', 'teeth', 'laryngeal squamous cell carcinoma', 'neck tumors','head and neck', 'head and neck cancer', 'head and neck cancers', 'head and neck tumor', 'head and neck tumors', 'head and neck neoplasm', 'head and neck neoplasms', 'laryngeal cancer', 'laryngeal cancers', 'laryngeal tumor', 'laryngeal tumors', 'laryngeal neoplasm', 'laryngeal neoplasms', 'pharyngeal cancer', 'pharyngeal cancers', 'pharyngeal tumor', 'pharyngeal tumors', 'pharyngeal neoplasm', 'pharyngeal neoplasms', 'oropharyngeal cancer', 'oropharyngeal cancers', 'oropharyngeal tumor', 'oropharyngeal tumors', 'oropharyngeal neoplasm', 'oropharyngeal neoplasms', 'hypopharyngeal cancer', 'hypopharyngeal cancers', 'hypopharyngeal tumor', 'hypopharyngeal tumors', 'hypopharyngeal neoplasm', 'hypopharyngeal neoplasms', 'nasopharyngeal cancer', 'nasopharyngeal cancers', 'nasopharyngeal tumor', 'nasopharyngeal tumors', 'nasopharyngeal neoplasm', 'nasopharyngeal neoplasms', 'oral cancer', 'oral cancers', 'oral tumor', 'oral tumors', 'oral neoplasm', 'oral neoplasms', 'tongue cancer', 'tongue cancers', 'tongue tumor', 'tongue tumors', 'tongue neoplasm', 'tongue neoplasms', 'lip cancer', 'lip cancers', 'lip tumor', 'lip tumors', 'lip neoplasm', 'lip neoplasms', 'salivary gland cancer', 'salivary gland cancers', 'salivary gland tumor', 'salivary gland tumors', 'salivary gland neoplasm', 'salivary gland neoplasms', 'sinus cancer', 'sinus cancers', 'sinus tumor', 'sinus tumors', 'sinus neoplasm', 'sinus neoplasms', 'maxillary sinus cancer', 'maxillary sinus cancers', 'maxillary sinus tumor', 'maxillary sinus tumors', 'maxillary sinus neoplasm', 'maxillary sinus neoplasms', 'larynx cancer', 'nasopharyngeal carcinoma', 'nasopharynx'],
    'renal cancer': ['renal cell carcinoma', 'kidney tumour', 'nephroblastoma', 'kidney diseases', 'kidney segmentation', 'renal carcinoma', 'renal carcinomas', 'renal adenocarcinoma', 'renal adenocarcinomas', 'renal malignancy', 'renal malignancies', 'uroplakin', 'clear cell renal cell carcinoma', 'malignant renal neoplasms', 'renal neoplasm', 'cystic renal lesions', 'ccrcc', 'fuhrman grading system', 'renal cancer', 'renal cancers', 'renal tumor', 'renal tumors', 'renal neoplasm', 'renal carcinoma', 'renal carcinomas', 'renal neoplasms', 'kidney cancer', 'kidney cancers', 'kidney tumor', 'kidney tumors', 'kidney neoplasm', 'kidney neoplasms', 'renal cell carcinoma', 'clear cell renal cell carcinoma', 'papillary renal cell carcinoma', 'chromophobe renal cell carcinoma', 'renal oncocytoma', 'collecting duct carcinoma'],
    'mesothelioma': ['mesothelioma', 'pleural mesothelioma'],
    'vaginal cancer': [ 'vaginal cancer', 'vaginal cancers', 'vaginal tumor', 'vaginal squamous cell carcinoma', 'vaginal intraepithelial neoplasia', 'vaginal malignancy', 'vaginal tumors', 'vaginal neoplasm', 'vaginal carcinoma', 'vaginal carcinomas', 'vaginal adenocarcinoma', 'vaginal adenocarcinomas', 'vaginal malignancy', 'vaginal malignancies', 'vaginal neoplasms', 'vaginal malignancy', 'vaginal malignancies', 'vaginal carcinoma', 'vaginal carcinomas', 'vaginal adenocarcinoma', 'vaginal adenocarcinomas' ],
    'vulvar cancer': [ 'vulvar cancer', 'vulvar cancers', 'vulvar squamous cell carcinoma', 'vulvar intraepithelial neoplasia', 'vulvar malignancy', 'vulvar tumor', 'vulvar tumors', 'vulvar neoplasm', 'vulvar neoplasms', 'vulvar malignancy', 'vulvar malignancies', 'vulvar carcinoma', 'vulvar carcinoma', 'vulvar carcinomas', 'vulvar adenocarcinoma', 'vulvar adenocarcinomas', 'vulvar malignancy', 'vulvar malignancies', 'vulvar carcinomas', 'vulvar adenocarcinoma', 'vulvar adenocarcinomas' ],
    'penile cancer': [ 'penile cancer', 'penile cancers', 'penile squamous cell carcinoma', 'penile intraepithelial neoplasia', 'penile tumor', 'penile tumors', 'penile neoplasm', 'penile neoplasms', 'penile malignancy', 'penile malignancies', 'penile carcinoma', 'penile carcinomas', 'penile adenocarcinoma', 'penile carcinoma', 'penile carcinomas', 'penile adenocarcinoma', 'penile adenocarcinomas', 'penile malignancy', 'penile malignancies', 'penile adenocarcinomas' ],
    'neuroendocrine tumors': ['neuroendocrine carcinoma', 'pleomorphic adenoma', 'parotid gland', 'salivary gland', 'neuroendocrine carcinomas', 'neuroendocrine adenocarcinoma', 'neuroendocrine adenocarcinomas', 'neuroendocrine malignancy', 'neuroendocrine malignancies', 'phaeochromocytoma', 'pheochromocytoma', 'phaeochromocytomas', 'pheochromocytomas', 'adrenal medullary tumor', 'adrenal medullary tumors', 'chromaffin cell tumor', 'chromaffin cell tumors', 'adrenal tumor', 'adrenal tumors', 'adrenal neoplasm', 'adrenal neoplasms', 'adrenal medullary neoplasm', 'adrenal medullary neoplasms', 'catecholamine-secreting tumor', 'catecholamine-secreting tumors', 'extra-adrenal paraganglioma', 'extra-adrenal paragangliomas', 'paraganglioma', 'paragangliomas', 'parasympathetic paraganglioma', 'sympathetic paraganglioma', 'carotid body tumor', 'carotid body tumors', 'chromaffin paraganglioma', 'neuroendocrine tumor', 'neuroendocrine tumors', 'neuroendocrine neoplasm', 'neuroendocrine neoplasms', 'neuroendocrine carcinoma', 'neuroendocrine carcinomas', 'small round cell neuroendocrine tumor', 'large cell neuroendocrine tumor', 'high-grade neuroendocrine tumor', 'low-grade neuroendocrine tumor', 'sympathetic tumor', 'sympathetic tumors', 'sympathetic neoplasm', 'sympathetic neoplasms', 'parasympathetic tumor', 'parasympathetic tumors', 'parasympathetic neoplasm', 'parasympathetic neoplasms',   'SDHB mutation', 'SDHD mutation', 'VHL mutation', 'NF1 mutation'],
    'mediastinal tumors': ['mediastinal tumors', 'mediastinal cancer', 'mediastinal neoplasm', 'mediastinal neoplasms', 'mediastinal tumor', 'mediastinal carcinoma', 'mediastinal carcinomas', 'mediastinal adenocarcinoma', 'mediastinal adenocarcinomas', 'mediastinal malignancy', 'mediastinal malignancies', 'thymic epithelial tumors', 'tets', 'thymic carcinomas', 'thymomas', 'neurilemoma', 'germ cell tumor', 'anterior mediastinal mass', 'mediastinal mass', 'thymic mass', 'thymic hyperplasia', 'malignant thymoma', 'thymic neuroendocrine tumor', 'invasive thymoma', 'thymic cyst', 'germ cell tumor of the mediastinum', 'teratoma of the mediastinum', 'mediastinal teratoma', 'mediastinal seminoma', 'neurogenic tumor of the mediastinum'],
    'bone cancers': ['bone cancer', 'sacral tumors', 'primary bone tumor', 'malignant bone tumor', "paget's disease of bone", 'giant cell tumor of bone', 'chordoma', 'adamantinoma', 'chondroma', 'enchondroma', 'vertebral tumor', 'bone carcinoma', 'bone carcinomas', 'bone adenocarcinoma', 'bone adenocarcinomas', 'bone malignancy', 'bone malignancies', 'spinal bone tumor', 'coccygeal tumor', 'pelvic bone tumor', 'iliac bone tumor', 'ischial bone tumor', 'pubic bone tumor', 'scapular tumor', 'clavicular tumor', 'sternal tumor', 'rib bone tumor', 'humeral bone tumor', 'femoral bone tumor', 'tibial bone tumor', 'fibrous dysplasia of bone', 'osteoblastoma', 'osteoma'],
    'melanoma': ['melanoma', 'mc1r', 'dermoscopy', 'dermatoscopy', 'cutaneous melanoma', 'malignant melanoma', 'skin melanoma', 'metastatic melanoma', 'melanocytic nevus', 'kit mutation', 'superficial spreading melanoma', 'nodular melanoma', 'acral lentiginous melanoma', 'lentigo maligna melanoma', 'amelanotic melanoma', 'melanoma in situ', 'clark level', 'breslow depth', 'sentinel lymph node biopsy', 'immunotherapy for melanoma', 'targeted therapy for melanoma', 'melanoma staging', 'abcde criteria'],
    'sarcoma': [ 'sarcoma', 'desmin', 'smarcb1', 'acta2', 'myod1', 'myogenin', 'pax3-foxo1', 'pax7-foxo1', 'smooth muscle actin', 'h-caldesmon', 'sarcomas', 'soft tissue sarcoma', 'soft tissue sarcomas', 'osteosarcoma', 'osteosarcomas', 'ewing sarcoma', 'ewing sarcomas', 'leiomyosarcoma', 'leiomyosarcomas', 'liposarcoma', 'liposarcomas', 'rhabdomyosarcoma', 'rhabdomyosarcomas', 'gastrointestinal stromal tumor', 'gastrointestinal stromal tumors', 'gist', 'gists', 'epithelioid sarcoma', 'epithelioid sarcomas', 'epithelioid sarcoma of soft tissue', 'giant cell fibroblastoma', 'giant cell fibroblastomas', 'gcf', 'dermatofibrosarcoma protuberans', 'dermatofibrosarcoma', 'dfsp', 'fibrosarcoma', 'fibrosarcomas', 'low-grade myofibroblastic sarcoma', 'low-grade myofibroblastic sarcomas', 'lgms', 'low-grade fibromyxoid sarcoma', 'low-grade fibromyxoid sarcomas', 'rhabdomyosarcoma tumor', 'rhabdomyosarcoma tumors', 'leiomyosarcoma of soft tissue', 'leiomyosarcoma of uterus', 'leiomyosarcoma of stomach', 'leiomyosarcoma of retroperitoneum', 'liposarcoma of soft tissue', 'liposarcoma of retroperitoneum', 'gastrointestinal stromal tumor of stomach', 'gastrointestinal stromal tumor of small intestine', 'gastrointestinal stromal tumor of esophagus', 'gastrointestinal stromal tumors of stomach', 'gastrointestinal stromal tumors of small intestine', 'gastrointestinal stromal tumors of esophagus', 'ewing tumor', 'ewing tumors', 'ewing neoplasm', 'ewing neoplasms', 'leiomyosarcoma tumor', 'leiomyosarcoma tumors', 'leiomyosarcoma neoplasm', 'leiomyosarcoma neoplasms', 'liposarcoma tumor', 'liposarcoma tumors', 'liposarcoma neoplasm', 'liposarcoma neoplasms', 'epithelioid sarcoma tumor', 'epithelioid sarcoma tumors', 'epithelioid sarcoma neoplasm', 'epithelioid sarcoma neoplasms', 'giant cell fibroblastoma tumor', 'giant cell fibroblastoma tumors', 'giant cell fibroblastoma neoplasm', 'giant cell fibroblastoma neoplasms', 'dermatofibrosarcoma tumor', 'dermatofibrosarcoma tumors', 'dermatofibrosarcoma neoplasm', 'dermatofibrosarcoma neoplasms', 'dfsp tumor', 'dfsp tumors', 'dfsp neoplasm', 'dfsp neoplasms', 'fibrosarcoma tumor', 'fibrosarcoma tumors', 'fibrosarcoma neoplasm', 'fibrosarcoma neoplasms', 'low-grade myofibroblastic sarcoma tumor', 'low-grade myofibroblastic sarcoma tumors', 'low-grade myofibroblastic sarcoma neoplasm', 'low-grade myofibroblastic sarcoma neoplasms', 'rhabdomyosarcoma of soft tissue', 'rhabdomyosarcoma of muscles', 'rhabdomyosarcoma of connective tissue', 'rhabdomyosarcoma neoplasm', 'rhabdomyosarcoma neoplasms', 'uterine sarcoma', 'uterine sarcomas', 'endometrial stromal sarcoma', 'synovial sarcoma'],
    'oncohematologic malignancies': ['leukemia', 'burkitt lymphoma', 'burkitt lymphomas', 'hematologic neoplasms', 'sternal puncture', 'cd3', 'cd20', 'bcl2', 'bcl6', 'CD138', 'syndecan-1', 'cd38', 'myeloperoxidase', 'fms like tyrosine kinase 3', 'nucleophosmin', 'ccaat/enhancer binding protein alpha', 'hematologic malignancy', 'hematologic malignancies', 'mpo', 'flt3', 'npm1', 'cebpa', 'cyclic adp ribose hydrolase', 'bone marrow transplant', 'bone marrow transplantation', 'dlbcl', 'myelodysplastic syndrome', 'myeloproliferative diseases', 'myeloproliferative disease', 'myelodysplastic syndromes', 'lymphomas', 'hematological malignancies', 'blood cancer', 'blood cancers', 'myeloid neoplasm', 'myeloid neoplasms', 'histiocytic and dendritic cell neoplasms', 'dendritic cell neoplasm', 'histiocytic cell neoplasm', 'lymphoid neoplasm', 'myeloma', 'haematological malignancies', 'liquid tomors', 'leukemias', 'lymphoma', 'multiple myeloma', 'acute myeloid leukemia', 'chronic lymphocytic leukemia', 'acute lymphoblastic leukemia', "hodgkin's lymphoma", "non-hodgkin's lymphoma", 'plasma cell neoplasms', 'plasma cell neoplasm', 'bone marrow biopsy', 'myelomas'],
    'bladder cancer': ['bladder cancer', 'bladder cancers', 'bladder tumor', 'bladder tumors', 'bladder neoplasm', 'bladder neoplasms', 'urothelial carcinoma', 'non-muscle invasive bladder cancer', 'muscle-invasive bladder cancer', 'squamous cell carcinoma of the bladder', 'bladder adenocarcinoma', 'urothelial carcinoma', 'bladder tumor', 'transitional cell carcinoma', 'hematuria', 'bacillus calmette-guérin', 'bcg therapy', 'turbt', 'nmibc', 'n-mibc', 'm-ibc', 'mibc' 'cystoscopy'],
    'esophageal cancer': ['esophageal cancer', "barrett's neoplasia", "barrett's esophagus", 'esophageal adenocarcinoma', 'esophageal squamous cell carcinoma', 'esophagectomy', 'dysphagia', 'gerd', 'gastroesophageal reflux disease', 'esophageal carcinoma', 'esophageal carcinomas', 'esophageal adenocarcinoma', 'esophageal adenocarcinomas', 'esophageal malignancy', 'esophageal malignancies'],
    'thyroid cancer': ['thyroid cancer', 'thyroid tumors', 'thyroid carcinoma', 'thyroid carcinomas', 'thyroid adenocarcinoma', 'thyroid adenocarcinomas', 'thyroid malignancy', 'thyroid malignancies', 'ret', 'ret/ptc', 'thyroid nodules', 'medullary thyroid carcinoma', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma', 'follicular thyroid adenoma', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma', 'medullary thyroid carcinoma', 'anaplastic thyroid carcinoma', 'thyroidectomy', 'tsh', 'thyroid-stimulating hormone', 'thyroid nodules', 'radioiodine therapy'],
    'testicular cancer': ['testicular cancer', 'testicular cancers', 'testicular tumor', 'testicular tumors', 'testicular carcinoma', 'testicular carcinomas', 'testicular adenocarcinoma', 'testicular adenocarcinomas', 'testicular malignancy', 'testicular malignancies', 'testicular neoplasm', 'testicular neoplasms', 'germ cell tumors', 'seminoma', 'non-seminoma', 'orchiectomy', 'alpha-fetoprotein', 'afp', 'beta-hcg', 'testicular mass'],
    'pancreatic cancer': ['pancreatic neuroendocrine neoplasms', 'glucagon', 'men1', 'pancreatic ductal adenocarcinoma', 'pdac', 'pancreatic cancer', 'pancreatic cancers', 'pancreatic tumor', 'pancreatic carcinoma', 'pancreatic carcinomas', 'pancreatic adenocarcinoma', 'pancreatic adenocarcinomas', 'pancreatic malignancy', 'pancreatic malignancies', 'pancreatic tumors', 'pancreatic neoplasm', 'pancreatic neoplasms', 'pancreatic ductal adenocarcinoma', 'mucinous cystic neoplasm of the pancreas', 'intraductal papillary mucinous neoplasm', 'solid pseudopapillary neoplasm of the pancreas', 'pancreatic adenocarcinoma', 'pancreatic neuroendocrine tumors', 'pancreatic neuroendocrine tumor', 'pancreas cancer', 'pancreas tumor', 'pancreas cancers', 'pancreas tumors', 'pancreas neoplasms', 'pancreas neoplasm', 'pnet'],
    'various cancers': ['multiple cancers', 'various diseases', 'subclonal selection', 'genotype matrix', 'tumor phylogeny', 'variant allele frequencies', 'methylation data', 'cancer origin determination', 'cell lines', 'cell line', 'circulating tumor cells', 'circulating tumor cell', 'several subtypes', 'cancer spheroids', 'cancer cell model', 'multi-cancer', 'ras/mek/erk', 'ck7', 'ck18', 'ck8', 'ctnnb1', 'p63', 'p53', 'tp53', 'ck5/6', 'pd-l1', 'pdl', 'pten', 'myc', 'cdkn2a', 'various tumor', 'various tumors', 'organs-at-risk', 'oars', 'multiple cancer types',  'various carcinoma', 'various carcinomas', 'various adenocarcinoma', 'various adenocarcinomas', 'various malignancy', 'various malignancies', 'pan-cancer', 'organoid growth', 'segmentation of cscs', 'ed visit risk among patients with cancer', 'nci-60', 'plwc', 'tumor exomes', 'vascular endothelial growth factor receptor', 'vegfr-2', 'different cancers', 'tumor angiogenic factors', 'extracellular matrix', 'breast-colorectal-endometrial','more than two hd cancers of interest', 'human tumors', 'multi-cancer-type', 'tumor type prediction', 'various cancer types', 'pan-cancer', 'different cancer types', 'cancer detection across multiple types', 'broad cancer diagnostic model', 'pan-cancer', 'multi-cancer detection', 'multi-cancer', 'tumor agnostic', 'common cancer markers', 'ai model for cancer diagnosis', 'deep learning for various cancer detection', 'various cancers', 'various atll cancer']
}

def match_keywords(text):
    doc = nlp(text)
    matches = matcher(doc)
    matched_keywords = set()
    for match_id, start, end in matches:
        span = doc[start:end].text
        matched_keywords.add(span.lower())
    return matched_keywords

def categorize_cancer(row):
    # Об'єднуємо текст в одну строку і застосовуємо предобробку для видалення апострофів
    combined_text = preprocess_text_smart(' '.join([str(row['Article Title']).lower(), str(row['Author Keywords']).lower(), str(row['Keywords Plus']).lower(), str(row['Abstract']).lower()]))

    # Застосовуємо spaCy до об'єднаного тексту
    doc = nlp(combined_text)

    # Лематизуємо текст
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    # Перевіряємо через matcher для слів з дефісами
    matched_keywords = match_keywords(combined_text)

    # Якщо знайдені збіги з ключовими словами, повертаємо тип раку
    for cancer_type, cancer_keywords in keywords.items():
        if any(keyword in matched_keywords for keyword in cancer_keywords):
            return cancer_type

    # Додаткова перевірка для ключових слів без дефісів
    for cancer_type, cancer_keywords in keywords.items():
        if any(f" {keyword} " in f" {lemmatized_text} " for keyword in cancer_keywords):
            return cancer_type

    return 'unknown'

# Функція для класифікації статей за моделями ШІ
def categorize_ai_model(row):
    ai_keywords = {
        'Linear/Logistic Regression': ['logistic regression', 'linear regression', 'regression model', 'lasso regression', 'stepwise regression'],
        'Decision Trees / Random Forests': ['decision tree', 'llm', 'prognostic model', 'decision trees', 'random forest', 'classification tree', 'tree-based model',  'out-of-bag error', 'gini impurity', 'entropy', 'pruning', 'max depth', 'min samples split', 'min samples leaf', 'random state', 'tree depth', 'forest ensemble', 'random feature selection', 'variable importance', 'bootstrap samples', 'oob score', 'splitting criterion', 'Logic Learning Machine'],
        'Support Vector Machines (SVM)': ['support vector machine', 'svm', 'kernel svm', 'svm classifier', 'rbf kernel', 'polynomial kernel'],
        'Convolutional Neural Networks (CNN)': ['image classification', 'PathML', 'image recognition', 'medical image analysis' 'efficientnet', 'mobilenet', 'shufflenet', 'vggnet', 'squeeznet', 'medical image segmentation', 'feature pyramid networks', 'fpn', 'fully connected layers', 'stride', 'filter size', 'pooling layers', 'dilated convolution', 'transform', 'wavelet transform', 'feature maps', 'pooling', 'spatial pyramid pooling', 'spp', 'tensorflow', 'keras', 'diffusion-weighted imaging', 'dwi', 'texture feature analysis', 'intelligent imaging technology', 't2-weighted imaging', 't1-weighted imaging', 'gray level co-occurrence matrix', 'multi-gray level size zone matrix', 'glcm', 'mglszm',  'convolutional neural network', 'radiomics', 'matrix-assisted laser desorption/ionization', 'image segmentation', 'whole slide images', 'whole slide image', 'WSI', 'MALDI', 'cnn', 'deep cnn', 'densenet', 'resnet', 'inception', 'alexnet', 'multi-layered cnn', 'convnet', 'convolution neural network'],
        'Recurrent Neural Networks (RNN/LSTM)': ['sequence model', 'time-series analysis', 'temporal data analysis', 'speech recognition', 'sentiment analysis', 'time-series prediction', 'gated recurrent unit', 'gru', 'bi-directional lstm', 'attention mechanism', 'sequence-to-sequence models', 'time-series forecasting', 'speech recognition', 'language modeling', 'vanishing gradient', 'exploding gradient', 'recurrent connections', 'recurrent neural network', 'rnn', 'long short-term memory', 'lstm', 'gru', 'sequential model', 'gated recurrent units'],
        'Generative Adversarial Networks (GANs)': ['wasserstein gan', 'image synthesis',  'wgan', 'synthetic data', 'adversarial training', 'unsupervised image generation', 'conditional gan', 'cgan', 'progressive gan', 'pgan', 'cyclegan', 'stylegan', 'pix2pix', 'latent space', 'discriminator loss', 'generator loss', 'mode collapse', 'unsupervised learning', 'image-to-image translation', 'image generation', 'super-resolution', 'style transfer', 'data augmentation', 'synthetic data generation', 'generative adversarial network', 'gan', 'gans', 'generator network', 'discriminator network', 'adversarial training'],
        'Artificial Neural Networks (ANN)': ['stochastic gradient descent', 'Tumor Dynamic Neural-ODE', 'TDNODE', 'adam optimizer', 'drop-out layers', 'regularization', 'hyperparameter tuning', 'weight initialization', 'activation functions', 'relu', 'leaky relu', 'sigmoid', 'tanh', 'batch normalization', 'momentum', 'early stopping', 'learning rate decay', 'xavier initialization', 'he initialization', 'mini-batch gradient descent', 'edge computing', 'fused weighted model', 'fused weighted deep extreme learning', 'federated learning', 'deep neural network', 'adaptive transfer-learning-based deep cox neural network', 'atrcn', 'transfer learning', 'cox neural network', 'artificial neural network', 'ann', 'feedforward neural network', 'backpropagation', 'multilayer perceptron', 'mlp', 'neural network model', 'deep learning network', 'neural network classifier'],
        'Text Classification': ['text classification', 'natural language processing', 'nlp', 'sentiment analysis', 'topic modeling', 'text mining', 'named entity recognition'],
        'Recommendation Systems': ['recommendation system', 'recommender system', 'collaborative filtering', 'content-based filtering', 'user-item interaction', 'matrix factorization'],
        'Genomic Models': ['genomic model', 'genomic prediction', 'gene expression profiling', 'rna sequencing', 'genotype-phenotype model', 'genetic model'],
        'Clinical Decision Support Systems': ['clinical decision support system', 'cdss', 'clinical support tool', 'decision support system', 'computer-aided diagnosis'],
        'Autoencoder': ['variational autoencoder', 'vae', 'denoising autoencoder', 'sparse autoencoder', 'contractive autoencoder', 'latent vector', 'reconstruction error', 'bottleneck layer', 'dimensionality reduction', 'data compression', 'autoencoder', 'deepc', 'deept2vec', 'transcriptomic feature vectors', 'tfv'],
        'U-Net Models': ['3d u-net', 'residual u-net', 'attention u-net', 'u-net++', 'u-net plus plus', 'skip connections', 'encoder-decoder architecture', 'multi-scale feature extraction', 'region of interest', 'feature map merging', 'unet', 'u-net', 'wsunet', 'weakly supervised unet', 'semantic segmentation'],
        'Gradient Boosting Models': ['gradient-boosted decision trees', 'gbdt', 'regularization', 'early stopping', 'shrinkage', 'decision boundary', 'weak learners', 'out-of-bag evaluation', 'cross-validation', 'hyperparameter tuning', 'learning rate', 'tree depth', 'number of estimators', 'boosted trees', 'gradient boosting', 'gradient boosting machine', 'gbm', 'boosting algorithm', 'boosted trees', 'gradient boosting trees', 'ensemble learning', 'boosted decision trees', 'adaboost', 'adaptive boosting', 'xgboost', 'extreme gradient boosting', 'categorical boosting', 'sequential model correction'],
        'Information Extraction': ['information extraction', 'feature extraction', 'text analysis', 'entity extraction', 'information retrieval'],
        'Ensemble': ['ensemble', 'ensemble machine learning classification model', 'ensemble method', 'bagging', 'stacking', 'blending', 'voting classifier', 'catboost', 'ensemble learning', 'ensemble classifier', 'majority voting', 'bootstrap aggregation', 'ensemble model']
    }

    combined_text = ' '.join([str(row['Article Title']).lower(), str(row['Author Keywords']).lower(), str(row['Keywords Plus']).lower(), str(row['Abstract']).lower()])

    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    for ai_type, ai_keywords_list in ai_keywords.items():
        if any(keyword in lemmatized_text for keyword in ai_keywords_list):
            return ai_type

    return 'unknown'

# Функція для класифікації точності моделей
def classify_accuracy(description):
    if not isinstance(description, str):
        return "Unknown"

    very_high_accuracy_pattern = r'\b(0\.(9[5-9]\d*)|[9][5-9]\.\d+|100)(\%)?\b'
    high_accuracy_pattern = r'\b(0\.(9[0-4]\d*)|[9][0-4]\.\d+|[9][0-4])(\%)?\b'
    medium_accuracy_pattern = r'\b(0\.(8[0-9]\d*)|[8][0-9]\.\d+|[8][0-9])(\%)?\b'
    low_accuracy_pattern = r'\b(0\.(7[0-9]\d*)|[7][0-9]\.\d+|[7][0-9])(\%)?\b'
    very_low_accuracy_pattern = r'\b(0\.(6\d+|[0-6]\.\d+|[0-6]))(\%)?\b'

    very_high_keywords = ['outstanding performance', 'clinically reliable', 'superior classification', 'exceptional accuracy']
    high_keywords = ['high accuracy', 'reliable for diagnosis', 'good prediction', 'clinically useful']
    medium_keywords = ['moderate accuracy', 'acceptable performance', 'reasonable prediction', 'risk assessment']
    low_keywords = ['low accuracy', 'requires improvement', 'preliminary assessment', 'limited clinical use']
    very_low_keywords = ['very low accuracy', 'unreliable', 'not suitable for clinical use', 'requires significant improvement']

    if re.search(very_high_accuracy_pattern, description) or any(word in description.lower() for word in very_high_keywords):
        return "Very high accuracy (≥ 95%)"
    elif re.search(high_accuracy_pattern, description) or any(word in description.lower() for word in high_keywords):
        return "High accuracy (90% - 94.9%)"
    elif re.search(medium_accuracy_pattern, description) or any(word in description.lower() for word in medium_keywords):
        return "Medium accuracy (80% - 89.9%)"
    elif re.search(low_accuracy_pattern, description) or any(word in description.lower() for word in low_keywords):
        return "Low accuracy (70% - 79.9%)"
    elif re.search(very_low_accuracy_pattern, description) or any(word in description.lower() for word in very_low_keywords):
        return "Very low accuracy (< 70%)"

    return "Unknown"

# Підрахунок кількості статей за категоріями без групування по роках і збереження результатів
def count_and_save_overall(df):
    print("Підрахунок кількості статей для кожної категорії...")
    # Підрахунок типів раку
    cancer_counts = df['Cancer Type'].value_counts()
    cancer_counts_df = cancer_counts.reset_index()
    cancer_counts_df.columns = ['Cancer Type', 'Count']
    cancer_counts_df.to_excel(output_path_cancer, index=False)

    # Підрахунок моделей ШІ
    ai_model_counts = df['AI Model'].value_counts()
    ai_model_counts_df = ai_model_counts.reset_index()
    ai_model_counts_df.columns = ['AI Model', 'Count']
    ai_model_counts_df.to_excel(output_path_ai, index=False)

    # Підрахунок категорій точності
    accuracy_category_counts = df['Accuracy_Category'].value_counts()
    accuracy_category_df = accuracy_category_counts.reset_index()
    accuracy_category_df.columns = ['Accuracy_Category', 'Count']
    accuracy_category_df.to_excel(output_path_accuracy, index=False)

# Підрахунок кількості статей за категоріями по роках і збереження результатів
def count_by_year_and_save(df, column_name, output_path):
    print(f"Підрахунок статей за роками для {column_name}...")
    grouped_counts = df.groupby(['Publication Year', column_name]).size().reset_index(name='Count')
    grouped_counts.to_excel(output_path, index=False)

# Основна функція обробки файлу
def process_excel_file(file_path):
    try:
        # Завантаження Excel файлу
        print(f"Завантаження файлу: {file_path}")
        df = pd.read_excel(file_path)

        # Перевірка, чи існують необхідні колонки
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract', 'Publication Year']
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")

        # Обробка типів раку
        print("Класифікация статей по типам раку...")
        with tqdm(total=len(df), desc="Типи раку") as pbar:
            df['Cancer Type'] = df.apply(lambda row: categorize_cancer(row), axis=1)
            pbar.update(1)

        print(f"Категоризировано {len(df)} статей по типам раку.")
        
        # Обробка моделей ШІ
        print("Класифікація статей за моделями ШІ...")
        with tqdm(total=len(df), desc="Моделі ШІ") as pbar:
            df['AI Model'] = df.apply(lambda row: categorize_ai_model(row), axis=1)
            pbar.update(1)
        print(f"Категоризовано {len(df)} статей за моделями ШІ.")
        
        # Обробка категорій точності
        print("Класифікація статей за точністю моделей...")
        with tqdm(total=len(df), desc="Точність моделей") as pbar:
            df['Accuracy_Category'] = df['Abstract'].apply(lambda description: classify_accuracy(description))
            pbar.update(1)
        print(f"Категоризовано {len(df)} статей за точністю моделей.")

        # Збережіть оновлені дані назад у Google Sheets
        update_google_sheets(worksheet, df)

        # Збереження результатів у початковий файл
        print("Збереження оновленого файлу з класифікацією...")
        df.to_excel(file_path, index=False)

        # Шляхи до вихідних файлів для загальних підрахунків
        global output_path_cancer, output_path_ai, output_path_accuracy
        output_path_cancer = file_path.replace('.xlsx', '_cancer_counts.xlsx')
        output_path_ai = file_path.replace('.xlsx', '_ai_model_counts.xlsx')
        output_path_accuracy = file_path.replace('.xlsx', '_accuracy_category_counts.xlsx')

        # Підрахунок та збереження загальних результатів
        count_and_save_overall(df)

        # Шляхи до вихідних файлів для підрахунків по роках
        output_path_cancer_year = file_path.replace('.xlsx', '_cancer_by_year.xlsx')
        output_path_ai_year = file_path.replace('.xlsx', '_ai_model_by_year.xlsx')
        output_path_accuracy_year = file_path.replace('.xlsx', '_accuracy_category_by_year.xlsx')

        # Підрахунок та збереження результатів по роках
        count_by_year_and_save(df, 'Cancer Type', output_path_cancer_year)
        count_by_year_and_save(df, 'AI Model', output_path_ai_year)
        count_by_year_and_save(df, 'Accuracy_Category', output_path_accuracy_year)

        print(f"File processed and saved successfully: {file_path}")
        print(f"Overall counts saved to: {output_path_cancer}, {output_path_ai}, {output_path_accuracy}")
        print(f"Yearly counts saved to: {output_path_cancer_year}, {output_path_ai_year}, {output_path_accuracy_year}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Додавання прогресу з tqdm
tqdm.pandas()

# Виклик основної функції
path_to_excel_file = r'D:\results\\1-6437.xlsx'
process_excel_file(path_to_excel_file)

if __name__ == "__main__":
    # Скачиваем JSON файл
    download_json(json_url, json_filename)

    # ID Google Sheets
    sheet_id = '1mT6y-50KkHjWG5NMGZObih_2Oyk8Ir0F'

    # Запускаем обработку Google Sheets
    process_google_sheet(sheet_id)
    