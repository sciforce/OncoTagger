import pandas as pd
import re

input_file_path = r"D:\results\new_dataset_cleaned.xlsx"

df = pd.read_excel(input_file_path)

cancer_keywords = [
    'cancer', 'tumor', 'oncology', 'carcinoma', 'neoplasm', 'malignancy', 'adenocarcinoma', 
    'sarcoma', 'melanoma', 'lymphoma', 'leukemia', 'neoplastic', 'oncogenic', 'metastasis', 
    'breast cancer', 'colorectal cancer', 'prostate cancer', 'lung cancer', 'brain cancer', 
    'cervical cancer', 'liver cancer', 'stomach cancer', 'endometrial cancer', 'skin cancer', 
    'ovarian cancer', 'head and neck cancer', 'renal cancer', 'mesothelioma', 'parathyroid cancer', 
    'gallbladder cancer', 'occult primary cancer', 'vaginal cancer', 'vulvar cancer', 'penile cancer', 
    'neuroendocrine tumors', 'mediastinal tumors', 'bone cancers', 'melanoma', 'sarcoma', 
    'oncohematologic malignancies', 'bladder cancer', 'esophageal cancer', 'thyroid cancer', 
    'testicular cancer', 'pancreatic cancer', 'various cancers', 'posterior fossa tumors', 
    'cervical carcinoma', 'keratoacanthoma', 'neck tumor', 'renal cell carcinoma', 'neuroendocrine carcinoma', 
    'multiple cancers', 'rectal carcinoma', 'lung papillary adenocarcinoma', 'gastroesophageal cancer', 
    'oral squamous cell carcinoma', 'pleural mesothelioma', 'pleomorphic adenoma', 'primary bone tumor', 
    'desmoid tumors', 'urothelial cell carcinoma', 'ca125', 'testicular tumors', 'pancreatic duct carcinoma', 
    'gastrointestinal tumor', 'laryngopharyngeal cancer', 'cutaneous melanoma', 'mesenchymal neoplasms', 
    'cutaneous melanoma', 'neuroendocrine adenocarcinoma', 'giant cell tumor of bone', 'smarcb1', 
    'gastric cardiac cancer', 'renal malignant tumor', 'primary bone tumor', 'small cell lung cancers', 
    'cutaneous squamous cell carcinoma', 'neuroendocrine adenocarcinomas', 'lymphoid neoplasm', 
    'adenocarcinoma of the colon', 'astrocytomas', 'nasopharyngeal neoplasms', 'ewing sarcoma', 
    'anaplastic thyroid carcinoma', 'renal carcinoma', 'carcinomas of unknown primary', 'fibrosarcoma', 
    'liposarcoma', 'low-grade myofibroblastic sarcoma', 'rhabdomyosarcoma', 'uterine sarcoma', 
    'endometrial stromal sarcoma', 'synovial sarcoma', 'medulloblastoma', 'diffuse intrinsic pontine glioma', 
    'parathyroid carcinomas', 'spinal bone tumor', 'leiomyosarcoma', 'thymic epithelial tumor', 
    'giant cell fibroblastoma', 'chordoma', 'histiocytic cell neoplasm', 'neurogenic tumor', 
    'cutaneous melanoma', 'glial fibrillary acidic protein', 'brainstem tumor', 'cerebellum tumors', 
    'pituitary tumors', 'pineal tumor', 'paraganglioma', 'nasopharyngeal cancer', 'cranial neoplasm', 
    'hemangioendothelioma', 'metastatic melanoma', 'pharyngeal cancer', 'parathyroid cancers', 
    'occult primary neoplasm', 'desmoid tumor', 'pleomorphic adenoma', 'occult cancer', 'cutaneous tumors', 
    'adrenal carcinoma', 'adrenal neoplasm', 'occult malignancy', 'occult tumor', 'ewing tumor', 
    'ewing tumors', 'thyroid fine-needle aspiration', 'thyroid neoplasm', 'thyroid carcinoma', 
    'thyroid cytopathology', 'thyroid tumors', 'thymic carcinoma', 'thymic epithelial tumor', 'neoplasms', 
    'germ cell tumors', 'germ cell neoplasms', 'brainstem neoplasm', 'cervical adenocarcinoma', 
    'stomach cancer', 'stomach tumors', 'uterine carcinoma', 'pancreatic neuroendocrine tumors', 
    'gastroesophageal junction cancer', 'uterine cancer', 'uterine malignancies', 'skin tumors', 
    'skin carcinoma', 'renal carcinoma', 'renal malignancy', 'prostate carcinoma', 'prostate malignancy', 
    'esophageal adenocarcinoma', 'vulvar squamous cell carcinoma', 'ovarian carcinoma', 'parathyroid carcinoma', 
    'liver carcinoma', 'laryngeal carcinoma', 'pleural cancer', 'sarcoma tumor', 'mesothelioma tumor', 
    'osteosarcoma tumor', 'cervical carcinoma', 'squamous cell carcinoma', 'renal cancer', 
    'lung carcinoma', 'breast carcinoma', 'prostate cancer tumor'
]

ai_keywords = ['Linear Regression', 'Logistic Regression', 'Decision Trees', 'Random Forests', 
    'Support Vector Machines', 'SVM', 'Convolutional Neural Networks', 'CNN', 
    'Recurrent Neural Networks', 'RNN', 'LSTM', 'Generative Adversarial Networks', 
    'GANs', 'Artificial Neural Networks', 'ANN', 'Text Classification', 
    'Recommendation Systems', 'Genomic Models', 'Clinical Decision Support Systems', 
    'Autoencoder', 'U-Net Models', 'Gradient Boosting Models', 'Information Extraction', 
    'Ensemble', 'image classification', 'sequence model', 'wasserstein gan', 
    'stochastic gradient descent', 'text classification', 'recommendation system', 
    'genomic model', 'clinical decision support system', 'variational autoencoder', 
    '3d u-net', 'gradient-boosted decision trees', 'information extraction', 
    'ensemble', 'linear regression', 'llm', 'svm', 'PathML', 'time-series analysis', 
    'multilevel-graph neural network', 'image synthesis', 'Tumor Dynamic Neural-ODE', 
    'natural language processing', 'recommender system', 'genomic prediction', 
    'cdss', 'vae', 'residual u-net', 'gbdt', 'feature extraction', 'regression model', 
    'prognostic model', 'kernel svm', 'image recognition', 'temporal data analysis', 
    'wgan', 'TDNODE', 'nlp', 'collaborative filtering', 'gene expression profiling', 
    'clinical support tool', 'denoising autoencoder', 'attention u-net', 'regularization', 
    'text analysis', 'ensemble machine learning', 'classification model', 'lasso regression', 
    'decision tree', 'svm classifier', 'medical image analysis', 'efficientnet', 
    'speech recognition', 'synthetic data', 'adam optimizer', 'sentiment analysis', 
    'content-based filtering', 'rna sequencing', 'decision support system', 'sparse autoencoder', 
    'u-net++', 'early stopping', 'entity extraction', 'ensemble method', 
    'stepwise regression', 'random forest', 'rbf kernel', 'mobilenet', 'adversarial training', 
    'drop-out layers', 'topic modeling', 'user-item interaction', 'genotype-phenotype model', 
    'computer-aided diagnosis', 'contractive autoencoder', 'u-net plus plus', 'shrinkage', 
    'information retrieval', 'bagging', 'classification tree', 'polynomial kernel', 
    'shufflenet', 'time-series prediction', 'unsupervised image generation', 
    'regularization', 'text mining', 'matrix factorization', 'genetic model', 'latent vector', 
    'skip connections', 'decision boundary', 'stacking', 'tree-based model', 'vggnet', 
    'gated recurrent unit', 'conditional gan', 'hyperparameter tuning', 'named entity recognition', 
    'reconstruction error', 'encoder-decoder architecture', 'weak learners', 'blending', 
    'out-of-bag error', 'squeeznet', 'gru', 'cgan', 'weight initialization', 
    'bottleneck layer', 'multi-scale feature extraction', 'out-of-bag evaluation', 
    'voting classifier', 'gini impurity', 'medical image segmentation', 
    'bi-directional lstm', 'progressive gan', 'activation functions', 'dimensionality reduction', 
    'region of interest', 'cross-validation', 'catboost', 'entropy', 'feature pyramid networks', 
    'attention mechanism', 'pgan', 'relu', 'data compression', 'feature map merging', 
    'pruning', 'fpn', 'sequence-to-sequence models', 'cyclegan', 'leaky relu', 
    'autoencoder', 'unet', 'tree depth', 'max depth', 'fully connected layers', 
    'time-series forecasting', 'stylegan', 'sigmoid', 'deepc', 'u-net', 'majority voting', 
    'min samples split', 'filter size', 'language modeling', 'latent space', 
    'batch normalization', 'transcriptomic feature vectors', 'weakly supervised unet', 
    'boosted trees', 'ensemble model', 'random state', 'pooling layers', 'vanishing gradient', 
    'discriminator loss', 'momentum', 'tfv', 'semantic segmentation', 
    'gradient boosting', 'tree depth', 'dilated convolution', 'exploding gradient', 
    'generator loss', 'early stopping', 'gradient boosting machine', 'gbm', 'boosting algorithm', 
    'random feature selection', 'wavelet transform', 'recurrent neural network', 
    'unsupervised learning', 'xavier initialization', 'boosted trees', 'gradient boosting trees', 
    'ensemble learning', 'Logic Learning Machine', 'tensorflow', 'sequential model', 
    'data augmentation', 'fused weighted deep extreme learning', 'adaboost', 'adaptive boosting', 
    'xgboost', 'extreme gradient boosting', 'categorical boosting', 'sequential model correction', 
    'dwi', 'gan', 'deep neural network', 'generator network', 'cox neural network', 
    'intelligent imaging technology', 't2-weighted imaging', 'discriminator network', 
    'gan', 'artificial neural network', 'ann', 'feedforward neural network', 
    'multilayer perceptron', 'mlp', 'cnn', 'deep learning network', 
    'neural network classifier', 'whole slide images', 'WSI', 'MALDI', 'densenet', 
    'resnet', 'inception', 'alexnet', 'multi-layered cnn', 'convnet'
    ]

cancer_pattern = re.compile('|'.join(cancer_keywords), re.IGNORECASE)
ai_pattern = re.compile('|'.join(ai_keywords), re.IGNORECASE)

def filter_rows(row, cancer_pattern, ai_pattern):
    combined_text = ' '.join(row.astype(str)) 
    return bool(cancer_pattern.search(combined_text)) and bool(ai_pattern.search(combined_text))

filtered_df = df[df.apply(filter_rows, axis=1, cancer_pattern=cancer_pattern, ai_pattern=ai_pattern)]

output_file_path = r"D:\results\filtered_dataset.xlsx"
filtered_df.to_excel(output_file_path, index=False)

print(f"Фильтрация завершена. Результаты сохранены в {output_file_path}")