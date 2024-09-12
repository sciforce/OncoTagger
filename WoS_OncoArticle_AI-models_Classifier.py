import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')

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

    # Checking for specific AI-models type keywords
    for ai_type, ai_keywords_list in ai_keywords.items():
        if any(keyword in lemmatized_text for keyword in ai_keywords_list):
            return ai_type

    # If no specific cancer type is found, return 'unknown'
    return 'unknown'

# Function to load and process Excel file, then save the result
def categorize_articles_by_ai_in_excel_and_save(file_path):
    try:
        # Open the Excel file
        df = pd.read_excel(file_path)

        # Checking if the necessary columns exist
        required_columns = ['Article Title', 'Author Keywords', 'Keywords Plus', 'Abstract']
        if not all(column in df.columns for column in required_columns):
            missing_columns = [column for column in required_columns if column not in df.columns]
            raise ValueError(f"Отсутствуют колонки в Excel-файле: {', '.join(missing_columns)}")

        # Handle missing data by filling with empty strings
        df[required_columns] = df[required_columns].fillna('')

        # Creating a new column 'Cancer Type' with the results of the keyword search
        df['AI Model'] = df.apply(categorize_ai_model, axis=1)

        # Save the updated dataframe back to the same file
        df.to_excel(file_path, index=False)

        print(f"File processed and saved successfully: {file_path}")
    except Exception as e:
        print(f"Error processing the file: {e}")

# Processing a specific file 
path_to_excel_file = r'D:\results\\1-6466.xlsx'
categorize_articles_by_ai_in_excel_and_save(path_to_excel_file)
