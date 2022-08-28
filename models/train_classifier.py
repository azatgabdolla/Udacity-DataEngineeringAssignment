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

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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
    return pipeline


def evaluate_model(model, X_test, y_test, col_names):

    y_pred = model.predict(X_test)
    y_test = np.array(y_test)

    metrics = []

    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        accuracy = accuracy_score(y_pred[:, i], y_test[:, i])
        precision = precision_score(y_pred[:, i], y_test[:, i], average='micro')
        recall = recall_score(y_pred[:, i], y_test[:, i], average='micro')
        f1 = f1_score(y_pred[:, i], y_test[:, i], average='micro')

        metrics.append([accuracy, precision, recall, f1])

    print( pd.DataFrame(data=np.array(metrics),
                        index=col_names,
                        columns=['Accuracy', 'Precision', 'Recall', 'F1']))



def save_model(model, model_filepath):
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