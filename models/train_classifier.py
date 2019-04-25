import sys
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Table_1", con=engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
        Toeknize the text
        - url removal
        - punctuation removal
        - tokenize words
        - stopwords removal
        - lemmatize nouns
        - lemmatize verbs
        
        argumnets:
            text (string): input text to tokenize
        returns:
            lemmed_words (list) : list of tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # remove punctuation from text( all character to lower)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    
    # tokenize word
    words = word_tokenize(text)
    
    # remove stop words
    words = [word.strip() for word in words if word not in stopwords.words('english')]
    
    # initialize WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize noun
    lemmed_words = [lemmatizer.lemmatize(word, pos = 'n') for word in words]
    
    # lemmatize_verb
    lemmed_words = [lemmatizer.lemmatize(word, pos = 'v') for word in lemmed_words ]
    
    return lemmed_words

def my_f1_score(Y_test, Y_pred):
    """
        This function calculated f1_score for multiclass output
        
        arguments:
            Y_test : true values
            Y_pred: predicted values
        returns:
            f1_score
    """
    Y_test, Y_pred = np.array(Y_test), np.array(Y_pred)
    f1_scores = []
    for j, _ in enumerate(range(Y_test.shape[1])):
        f1_scores.append(f1_score(y_true = Y_test[:, j], y_pred = Y_pred[:, j], average = 'weighted'))
    return np.average(f1_score)


def build_model():
    """
        Build model using sklearn's Pipeline
        
        Args:
            None
        Returns:
            Pipeline Model
    """
    
    # using the custom tokenize function defined above
    model = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    
    # parameters with values  
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                   'vect__max_df': (0.5, 1.0),
                 'vect__max_features': (None, 5000, 10000),
                  'tfidf__use_idf': (True, False),
                  'clf__estimator__n_estimators': (10, 50, 100, 200)
                 }
    
    # make scorer for grid search
    scorer = make_scorer(my_f1_score, greater_is_better = True)
    
    cv_model = GridSearchCV(model, param_grid = parameters, scoring = scorer,
                            verbose = 2, n_jobs = -1, refit = True)
    return cv_model
    #return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluate model on X_test
        get precision, recall, f1-score for Y_test and y_pred
        
        arguments:
            mpdel (sklearn Pipeline): model
            X_test (numpy array): input to model
            Y_test (numpy array): true labels
            category_names (list): names of all categories 
    """
    
    # get the predicted output for X_test with input model
    Y_pred = model.predict(X_test)
    
    precisions, recalls, f1_scores = [], [], []
    
    Y_test, Y_pred = np.array(Y_test), np.array(Y_pred)
    for j, _ in enumerate(range(Y_test.shape[1])):
        
        precision_ = precision_score(y_true = Y_test[:, j], y_pred = Y_pred[:, j], average = 'weighted')
        recall_ = recall_score(y_true = Y_test[:, j], y_pred = Y_pred[:, j], average = 'weighted')
        f1_score_ = f1_score(y_true = Y_test[:, j], y_pred = Y_pred[:, j], average = 'weighted')
       
        precisions.append(precision_)
        recalls.append(recall_)
        f1_scores.append(f1_score_)
        
        print('Feature =  ', category_names[j], 'Precision = ', precision_, ' Recall = ', recall_, 'F1_score = ', f1_score_)
        
        
    print('\nFINAL\nprecision =  ', np.average(precisions), ' recall = ', np.average(recalls), ' f1_score = ', np.average(f1_scores))
    
    
    


def save_model(model, model_filepath):
    """
        Save trained model as pickle file
        
        arguments:
            model (Sklearn Pipeline object): model object
            model_filepath (path as .pkl): filepath to save trained model
        Return:
            None
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        #print(X.head())
        #print(Y.head())
        #print(category_names)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()