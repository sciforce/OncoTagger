import pandas as pd
import spacy
import re

# Завантажуємо модель spaCy
nlp = spacy.load('en_core_web_sm')

# Функція для класифікації статей за типом раку
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

    combined_text = ' '.join([
        str(row['Article Title']).lower(),
        str(row['Author Keywords']).lower(),
        str(row['Keywords Plus']).lower(),
        str(row['Abstract']).lower()
    ])

    doc = nlp(combined_text)
    lemmatized_text = ' '.join([token.lemma_ for token in doc])

    for cancer_type, cancer_keywords in keywords.items():
        if any(keyword in lemmatized_text for keyword in cancer_keywords):
            return cancer_type

    return 'unknown'

# Функція для класифікації статей за моделями ШІ
def categorize_ai_model(row):
    ai_keywords = {
        'Linear/Logistic Regression': ['logistic regression', 'linear regression', 'regression model', 'lasso regression', 'stepwise regression'],
        'Decision Trees / Random Forests': ['decision tree', 'prognostic model', 'decision trees', 'random forest', 'classification tree', 'tree-based model',  'out-of-bag error', 'gini impurity', 'entropy', 'pruning', 'max depth', 'min samples split', 'min samples leaf', 'random state', 'tree depth', 'forest ensemble', 'random feature selection', 'variable importance', 'bootstrap samples', 'oob score', 'splitting criterion'],
        'Support Vector Machines (SVM)': ['support vector machine', 'svm', 'kernel svm', 'svm classifier', 'rbf kernel', 'polynomial kernel'],
        'Convolutional Neural Networks (CNN)': ['image classification', 'image recognition', 'medical image analysis' 'efficientnet', 'mobilenet', 'shufflenet', 'vggnet', 'squeeznet', 'medical image segmentation', 'feature pyramid networks', 'fpn', 'fully connected layers', 'stride', 'filter size', 'pooling layers', 'dilated convolution', 'transform', 'wavelet transform', 'feature maps', 'pooling', 'spatial pyramid pooling', 'spp', 'tensorflow', 'keras', 'diffusion-weighted imaging', 'dwi', 'texture feature analysis', 'intelligent imaging technology', 't2-weighted imaging', 't1-weighted imaging', 'gray level co-occurrence matrix', 'multi-gray level size zone matrix', 'glcm', 'mglszm',  'convolutional neural network', 'radiomics', 'matrix-assisted laser desorption/ionization', 'image segmentation', 'whole slide images', 'whole slide image', 'WSI', 'MALDI', 'cnn', 'deep cnn', 'densenet', 'resnet', 'inception', 'alexnet', 'multi-layered cnn', 'convnet', 'convolution neural network'],
        'Recurrent Neural Networks (RNN/LSTM)': ['sequence model', 'time-series analysis', 'temporal data analysis', 'speech recognition', 'sentiment analysis', 'time-series prediction', 'gated recurrent unit', 'gru', 'bi-directional lstm', 'attention mechanism', 'sequence-to-sequence models', 'time-series forecasting', 'speech recognition', 'language modeling', 'vanishing gradient', 'exploding gradient', 'recurrent connections', 'recurrent neural network', 'rnn', 'long short-term memory', 'lstm', 'gru', 'sequential model', 'gated recurrent units'],
        'Generative Adversarial Networks (GANs)': ['wasserstein gan', 'image synthesis',  'wgan', 'synthetic data', 'adversarial training', 'unsupervised image generation', 'conditional gan', 'cgan', 'progressive gan', 'pgan', 'cyclegan', 'stylegan', 'pix2pix', 'latent space', 'discriminator loss', 'generator loss', 'mode collapse', 'unsupervised learning', 'image-to-image translation', 'image generation', 'super-resolution', 'style transfer', 'data augmentation', 'synthetic data generation', 'generative adversarial network', 'gan', 'gans', 'generator network', 'discriminator network', 'adversarial training'],
        'Artificial Neural Networks (ANN)': ['stochastic gradient descent', 'adam optimizer', 'drop-out layers', 'regularization', 'hyperparameter tuning', 'weight initialization', 'activation functions', 'relu', 'leaky relu', 'sigmoid', 'tanh', 'batch normalization', 'momentum', 'early stopping', 'learning rate decay', 'xavier initialization', 'he initialization', 'mini-batch gradient descent', 'edge computing', 'fused weighted model', 'fused weighted deep extreme learning', 'federated learning', 'deep neural network', 'adaptive transfer-learning-based deep cox neural network', 'atrcn', 'transfer learning', 'cox neural network', 'artificial neural network', 'ann', 'feedforward neural network', 'backpropagation', 'multilayer perceptron', 'mlp', 'neural network model', 'deep learning network', 'neural network classifier'],
        'Text Classification': ['text classification', 'natural language processing', 'nlp', 'sentiment analysis', 'topic modeling', 'text mining', 'named entity recognition'],
        'Recommendation Systems': ['recommendation system', 'recommender system', 'collaborative filtering', 'content-based filtering', 'user-item interaction', 'matrix factorization'],
        'Genomic Models': ['genomic model', 'genomic prediction', 'gene expression profiling', 'rna sequencing', 'genotype-phenotype model', 'genetic model'],
        'Clinical Decision Support Systems': ['clinical decision support system', 'cdss', 'clinical support tool', 'decision support system', 'computer-aided diagnosis'],
        'Autoencoder': ['variational autoencoder', 'vae', 'denoising autoencoder', 'sparse autoencoder', 'contractive autoencoder', 'latent vector', 'reconstruction error', 'bottleneck layer', 'dimensionality reduction', 'data compression', 'autoencoder', 'deepc', 'deept2vec', 'transcriptomic feature vectors', 'tfv'],
        'U-Net Models': ['3d u-net', 'residual u-net', 'attention u-net', 'u-net++', 'u-net plus plus', 'skip connections', 'encoder-decoder architecture', 'multi-scale feature extraction', 'region of interest', 'feature map merging', 'unet', 'u-net', 'wsunet', 'weakly supervised unet', 'semantic segmentation'],
        'Gradient Boosting Models': ['gradient-boosted decision trees', 'gbdt', 'regularization', 'early stopping', 'shrinkage', 'decision boundary', 'weak learners', 'out-of-bag evaluation', 'cross-validation', 'hyperparameter tuning', 'learning rate', 'tree depth', 'number of estimators', 'boosted trees', 'gradient boosting', 'gradient boosting machine', 'gbm', 'boosting algorithm', 'boosted trees', 'gradient boosting trees', 'ensemble learning', 'boosted decision trees', 'adaboost', 'adaptive boosting', 'xgboost', 'extreme gradient boosting', 'categorical boosting', 'sequential model correction'],
        'Information Extraction': ['information extraction', 'feature extraction', 'text analysis', 'entity extraction', 'information retrieval'],
        'Ensemble': ['ensemble', 'ensemble method', 'bagging', 'stacking', 'blending', 'voting classifier', 'catboost', 'ensemble learning', 'ensemble classifier', 'majority voting', 'bootstrap aggregation', 'ensemble model']
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
    
    very_high_keywords = ['outstanding performance', 'clinically reliable']
    high_keywords = ['high accuracy', 'clinically useful']
    medium_keywords = ['moderate accuracy', 'reasonable prediction']
    low_keywords = ['low accuracy', 'limited clinical use']
    very_low_keywords = ['very low accuracy', 'not suitable for clinical use']

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

# Функція для підрахунку та збереження типів раку
def count_cancer_types_and_save(df, output_path):
    cancer_counts = df['Cancer Type'].value_counts()
    cancer_counts_df = cancer_counts.reset_index()
    cancer_counts_df.columns = ['Cancer Type', 'Count']
    cancer_counts_df.to_excel(output_path, index=False)

# Функція для підрахунку та збереження моделей ШІ
def count_ai_models_and_save(df, output_path):
    ai_model_counts = df['AI Model'].value_counts()
    ai_model_counts_df = ai_model_counts.reset_index()
    ai_model_counts_df.columns = ['AI Model', 'Count']
    ai_model_counts_df.to_excel(output_path, index=False)

# Функція для підрахунку та збереження категорій точності
def count_accuracy_category_and_save(df, output_path):
    accuracy_category_counts = df['Accuracy_Category'].value_counts()
    accuracy_category_df = accuracy_category_counts.reset_index()
    accuracy_category_df.columns = ['Accuracy_Category', 'Count']
    accuracy_category_df.to_excel(output_path, index=False)

# Основна функція обробки файлу
def process_excel_file(file_path):
    try:
        # Завантаження Excel файлу
        df = pd.read_excel(file_path)

        # Перевірка, чи існують необхідні колонки
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract']
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            raise ValueError(f"Missing columns in the Excel file: {', '.join(missing_columns)}")

        # Обробка типів раку
        df['Cancer Type'] = df.apply(categorize_cancer, axis=1)
        
        # Обробка моделей ШІ
        df['AI Model'] = df.apply(categorize_ai_model, axis=1)
        
        # Обробка категорій точності
        df['Accuracy_Category'] = df['Abstract'].apply(classify_accuracy)

        # Збереження результатів у початковий файл
        df.to_excel(file_path, index=False)
        
        # Підрахунок та збереження результатів
        count_cancer_types_and_save(df, file_path.replace('.xlsx', '_cancer_counts.xlsx'))
        count_ai_models_and_save(df, file_path.replace('.xlsx', '_ai_model_counts.xlsx'))
        count_accuracy_category_and_save(df, file_path.replace('.xlsx', '_accuracy_category_counts.xlsx'))

        print(f"File processed and saved successfully: {file_path}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Виклик основної функції
path_to_excel_file = r'D:\results\\1-6466.xlsx'
process_excel_file(path_to_excel_file)
