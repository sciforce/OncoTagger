import pandas as pd
import spacy
import re
from tqdm import tqdm

# Завантажуємо модель spaCy
nlp = spacy.load('en_core_web_sm')

# Функція для класифікації статей за типом раку
def categorize_cancer(row):
    keywords = {
        'breast cancer': ['breast cancer', 'breast ultrasound', 'breast tumor', 'breast tumors', 'mcf-7', 'breast neoplasm', 'breast neoplasms', 'abus', 'mammography', 'mammogram', 'bi-rads', 'ductal carcinoma', 'lobular carcinoma', 'breast biopsies'],
        'colorectal cancer': ['rectal cancer', 'colon cancer', 'colorectal cancer', 'rectum', 'sigmoid colon'],
        'prostate cancer': ['prostate cancer', 'psa', 'prostate biopsy', 'prostatic adenocarcinoma'],
        'lung cancer': ['lung cancer', 'nsclc', 'sclc', 'alk mutation', 'lung adenocarcinoma', 'iladc'],
        'brain cancer': ['brain tumor', 'brain cancer', 'posterior fossa tumors', 'glioma', 'astrocytoma', 'meningioma', 'glioblastoma'],
        'cervical cancer': ['cervical cancer', 'hela', 'cervical tumor', 'hpv', 'pap-smear', 'cervical intraepithelial neoplasia'],
        'liver cancer': ['liver cancer', 'liver tumors', 'hepatic tumor', 'hcc', 'hepatocellular carcinoma', 'cirrhosis', 'hepatitis b', 'afp', 'cholangioscopy'],
        'stomach cancer': ['stomach cancer', 'upper gastrointestinal tract carcinoma', 'gastrointestinal cancer', 'gastric tumor', 'gastric cancer', 'gastric adenocarcinoma', 'helicobacter pylori'],
        'endometrial cancer': ['endometrial cancer', 'uterine cancer', 'endometrial carcinoma'],
        'skin cancer': ['skin cancer', 'skin tumors', 'skin tumor', 'basal cell carcinoma', 'squamous cell carcinoma'],
        'ovarian cancer': ['ovarian cancer', 'figo', 'ovarian tumors', 'ovarian tumor', 'adnexal masses'],
        'head and neck cancer': ['neck tumor', 'nasopharyngeal carcinoma', 'nasopharynx', 'neck tumors', 'tongue tumor', 'head and neck cancer', 'pharyngeal cancer', 'oral cancer'],
        'renal cancer': ['renal cell carcinoma', 'clear cell renal cell carcinoma', 'malignant renal neoplasms', 'renal neoplasm', 'cystic renal lesions', 'ccrcc', 'fuhrman grading system', 'kidney tumor'],
        'mesothelioma': ['mesothelioma', 'pleural mesothelioma'],
        'mediastinal tumors': ['mediastinal tumors', 'mediastinal cancer', 'mediastinal neoplasm', 'mediastinal neoplasms', 'mediastinal tumor', 'thymic epithelial tumors', 'tets', 'thymic carcinomas', 'thymomas', 'neurilemoma', 'germ cell tumor', 'anterior mediastinal mass', 'mediastinal mass', 'thymic mass', 'thymic hyperplasia', 'malignant thymoma', 'thymic neuroendocrine tumor', 'invasive thymoma', 'thymic cyst', 'germ cell tumor of the mediastinum', 'teratoma of the mediastinum', 'mediastinal teratoma', 'mediastinal seminoma', 'neurogenic tumor of the mediastinum'],
        'bone cancers': ['bone cancer', 'sacral tumors', 'primary bone tumor', 'malignant bone tumor', "paget's disease of bone", 'giant cell tumor of bone', 'chordoma', 'adamantinoma', 'chondroma', 'enchondroma', 'vertebral tumor', 'spinal bone tumor', 'coccygeal tumor', 'pelvic bone tumor', 'iliac bone tumor', 'ischial bone tumor', 'pubic bone tumor', 'scapular tumor', 'clavicular tumor', 'sternal tumor', 'rib bone tumor', 'humeral bone tumor', 'femoral bone tumor', 'tibial bone tumor', 'fibrous dysplasia of bone', 'osteoblastoma', 'osteoma'],
        'melanoma': ['melanoma', 'cutaneous melanoma', 'malignant melanoma', 'skin melanoma', 'metastatic melanoma', 'melanocytic nevus', 'braf mutation', 'nras mutation', 'kit mutation', 'superficial spreading melanoma', 'nodular melanoma', 'acral lentiginous melanoma', 'lentigo maligna melanoma', 'amelanotic melanoma', 'melanoma in situ', 'clark level', 'breslow depth', 'sentinel lymph node biopsy', 'immunotherapy for melanoma', 'targeted therapy for melanoma', 'melanoma staging', 'abcde criteria'],
        'sarcoma': ['sarcoma', 'soft tissue sarcoma', 'osteosarcoma', 'ewing sarcoma', 'leiomyosarcoma', 'liposarcoma', 'rhabdomyosarcoma', 'gist', 'gastrointestinal stromal tumor', 'synovial sarcoma', 'sarcomas'],
        'oncohematologic malignancies': ['leukemia', 'bone marrow transplant', 'bone marrow transplantation', 'dlbcl', 'myelodysplastic syndrome', 'myeloproliferative diseases', 'myeloproliferative disease', 'myelodysplastic syndromes', 'lymphomas', 'hematological malignancies', 'blood cancer', 'blood cancers', 'myeloid neoplasm', 'myeloid neoplasms', 'histiocytic and dendritic cell neoplasms', 'dendritic cell neoplasm', 'histiocytic cell neoplasm', 'lymphoid neoplasm', 'myeloma', 'haematological malignancies', 'liquid tomors', 'leukemias', 'lymphoma', 'multiple myeloma', 'acute myeloid leukemia', 'chronic lymphocytic leukemia', 'acute lymphoblastic leukemia', "hodgkin's lymphoma", "non-hodgkin's lymphoma", 'plasma cell neoplasms', 'plasma cell neoplasm', 'bone marrow biopsy'],
        'bladder cancer': ['bladder cancer', 'urothelial carcinoma', 'bladder tumor' 'transitional cell carcinoma', 'hematuria', 'bacillus calmette-guérin', 'bcg therapy', 'turbt', 'non-muscle invasive bladder cancer', 'muscle-invasive bladder cancer', 'cystoscopy'],
        'esophageal cancer': ['esophageal cancer', "barrett's neoplasia", "barrett's esophagus", 'esophageal adenocarcinoma', 'esophageal squamous cell carcinoma', 'esophagectomy', 'dysphagia', 'gerd', 'gastroesophageal reflux disease', 'endoscopic resection'],
        'thyroid cancer': ['thyroid cancer', 'thyroid nodules', 'medullary thyroid carcinoma', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma', 'follicular thyroid adenoma', 'papillary thyroid carcinoma', 'follicular thyroid carcinoma', 'medullary thyroid carcinoma', 'anaplastic thyroid carcinoma', 'thyroidectomy', 'tsh', 'thyroid-stimulating hormone', 'thyroid nodules', 'radioiodine therapy'],
        'testicular cancer': ['testicular cancer', 'germ cell tumors', 'seminoma', 'non-seminoma', 'orchiectomy', 'alpha-fetoprotein', 'afp', 'beta-hcg', 'testicular mass'],
        'pancreatic cancer': ['pancreatic neuroendocrine neoplasms', 'pancreatic ductal adenocarcinoma', 'pdac', 'pancreatic tumor', 'pancreatic cancer', 'pancreatic neoplasm', 'pancreatic adenocarcinoma', 'pancreatic neuroendocrine tumors', 'pnet'],
        'various cancers': ['multiple cancers', 'organs-at-risk', 'oars', 'multiple cancer types', 'pan-cancer', 'organoid growth', 'segmentation of cscs', 'ed visit risk among patients with cancer', 'nci-60', 'plwc', 'tumor exomes', 'vascular endothelial growth factor receptor', 'vegfr-2', 'different cancers', 'tumor angiogenic factors', 'extracellular matrix', 'breast-colorectal-endometrial','more than two hd cancers of interest', 'human tumors', 'multi-cancer-type', 'tumor type prediction', 'various cancer types', 'pan-cancer', 'different cancer types', 'cancer detection across multiple types', 'broad cancer diagnostic model', 'pan-cancer', 'multi-cancer detection', 'multi-cancer', 'tumor agnostic', 'common cancer markers', 'ai model for cancer diagnosis', 'deep learning for various cancer detection']
    }

    combined_text = ' '.join([
        str(row['Article Title']).lower(),
        str(row['Author Keywords']).lower(),
        str(row['Keywords Plus']).lower(),
        str(row['Abstract']).lower()
    ])

    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

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

    combined_text = ' '.join([
        str(row['Article Title']).lower(),
        str(row['Author Keywords']).lower(),
        str(row['Keywords Plus']).lower(),
        str(row['Abstract']).lower()
    ])

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
        print("Класифікація статей за типом раку...")
        df['Cancer Type'] = df.apply(categorize_cancer, axis=1)
        
        # Обробка моделей ШІ
        print("Класифікація статей за моделями ШІ...")
        df['AI Model'] = df.apply(categorize_ai_model, axis=1)
        
        # Обробка категорій точності
        print("Класифікація статей за точністю моделей...")
        df['Accuracy_Category'] = df['Abstract'].apply(classify_accuracy)

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