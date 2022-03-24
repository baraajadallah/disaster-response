import sys

import re
import pandas as pd
from sqlalchemy import create_engine
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Load and merge datasets
    input:
         database name
    outputs:
        X: messages 
        y: everything esle
        category names.
    """
    engine = create_engine('sqlite:///'+database_filepath).connect()
    df = pd.read_sql_table('DisasterResponse', engine)
    
    #Some more cleaning
    df.replace([np.inf,-np. inf], np.nan, inplace=True)
    # Drop rows with NaN
    df.dropna (inplace=True)
    
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(Y.columns)
    return X, Y, category_names
    
def tokenize(text):
    """ Normalize and tokenize
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    pipeline construction
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()), ('clf',                          MultiOutputClassifier(RandomForestClassifier()))]) 
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 20]
    }

    model = GridSearchCV(pipeline, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    inputs
        model
        X_test
        y_test
        category_names
    output:
        classification report
    """
    y_pred = model.predict(X_test)
    for col in range(len(category_names)):
        print(Y_test.columns[col])
        print(classification_report(Y_test.iloc[:,col], y_pred[:,col]))


def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
