import sys
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle


def load_data(database_filepath):
    
    '''
    - Import processed data from local database
    
    - Split dataset on X and Y
    '''
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df_1', con = engine)
    
    X = df.message.values
    
    Y = df.iloc[:, range(4,df.shape[1])]
    Y = Y.loc[:, Y.nunique() > 1]
    
    category_names = list(Y.columns.values)
    return X,Y,category_names




def tokenize(text):
    
    '''
    - normalize case and remove punctuation
    - tokenize text
    - lemmatize and remove stop words from each token
    
    '''
#     stopwords = stopwords.words('english')
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    
    '''
    - Pipeline tfidf vectorizer + Multioutput XGB classifier
    
    - Introduction of parameters grid (many of them commented out to save running time)
    
    -  GridsearhCV Initialisation 
    '''
    pipeline = Pipeline([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('clf', MultiOutputClassifier(XGBClassifier() ) )
        
                            ])

    parameters = {
#         'text_pipeline__vect__ngram_range':((1, 1), (1, 2), (2,2), (1,3)),
#         'text_pipeline__vect__max_df':(0.5, 0.75, 1.0),
#        'text_pipeline__vect__max_features': (None, 5000, 10000),
        'text_pipeline__tfidf__use_idf': (True, False)
#         'clf__estimator__n_estimators': [50, 100, 200, 500],
#         'clf__estimator__min_samples_split': [2, 3, 4, 5, 6]
    }


    gs = GridSearchCV( pipeline, param_grid = parameters,
                         scoring = "accuracy", cv = 5,
                        n_jobs = -1,  return_train_score = True, verbose = 3)

    return gs


def evaluate_model(model, X_test, y_test, category_names):
    '''
    - predict on the test data
    - classification report
    '''
    
    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    for i, col in enumerate(category_names): 
            print('------------------------## Category ##------------------------')
            print(col)
            print(classification_report(y_test[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    
    '''
    Save model in pkl extension to the local folder
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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