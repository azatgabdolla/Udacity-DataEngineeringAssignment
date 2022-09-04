import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os


def load_data(messages_filepath, categories_filepath):
    
    '''
    - import messages and categories files
    
    - left join of two files by id
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,
                   on = 'id',
                   how = 'left')
    return df


def clean_data(df):
    
    '''
    - each cell in categories is split in 2:
        - the text part is used as a column name
        - the numerical part is classification type 0-1
    
    - categories are concatenated with messages 
    
    - drop duplicates if there're
    
    - remove rows where related is equal to 2. It's done to make assignment binary-classification 
    
    '''
    
    column_names = df.categories.str.split(';', expand = True).iloc[0,:]
    column_names = column_names.replace('-[0-9]', '', regex = True)
    categories = df.categories.str.split(';', expand = True)
    row = df.categories.str.split(';', expand = True).iloc[0,:]
    category_colnames = row.replace('-[0-9]', '', regex = True)
    categories.columns = category_colnames
    
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()
    df = df.loc[df['related'] != 2, :]
    
    return df


def save_data(df, database_filename):
    '''
    - initialise localdatabase engine
    - save output to database
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    #result = engine.execute("drop table if exists df_1;")
    df.to_sql('df_1',engine,   index=False, if_exists='replace')

def main():
    '''
    
    - pipeline of running etl function together with inputs
    
    '''
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()