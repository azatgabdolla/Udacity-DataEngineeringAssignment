import sys
from sqlalchemy import create_engine
import re
import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
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

from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df_1', con = engine)
    X = df.message.values
    Y = df.iloc[:, range(4,df.shape[1])]
    Y = Y.loc[:, Y.nunique() > 1]
    category_names = list(Y.columns.values)
    return X,Y, category_names


def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline(steps = [('vect', CountVectorizer(tokenizer=tokenize)),
                             ('tfidf', TfidfTransformer()),
                            ('classifier', MultiOutputClassifier( LogisticRegression()
                                                                        ))])
    parameters = {
                  'classifier__estimator__penalty' : ['l1','l2'],
                  'classifier__estimator__C':[1, 3],
                  'classifier__estimator__solver' : ['liblinear', 'saga'],
                }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, 
                      verbose=3,
                      # scoring = 'accuracy',
                      cv = None
                     )

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    for column in Y_test.columns:
        col_loc = Y_test.columns.get_loc(column)
        print(column)
        print(classification_report(y_test.loc[:,column], [row[col_loc] for row in y_pred] ))
        print('accuracy', ([row[col_loc] for row in y_pred] == y_test.loc[:,column]).mean())


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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