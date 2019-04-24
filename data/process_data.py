import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        This function loads and merge message and categories data
        arguments:
            messages_filepath: messages data filepath 
            categories_filepath: categories data filepath
        return:
            (dataframe) : clean dataframe
            
    """
    # messages data
    messages_df = pd.read_csv(messages_filepath)
    # categories data
    categories_df = pd.read_csv(categories_filepath)
    # merge message and categories dataframes
    merged_df = messages_df.merge(categories_df, how = 'outer', on  = ['id'])
    
    return merged_df


def clean_data(df):
    """
        Clean dataframe by splitting categories column into  seperate category column and remove duplicates
       
        arguments:
            df (Dataframe):  pandas dataframe to clean
        returns:
            df (pandas dataframe)
    """
    # split and expand categories 
    categories_ = df['categories'].str.split(';', expand = True)
    # all new column names
    column_names = categories_.iloc[0, :].apply(lambda x: x.split('-')[0])
    # update the column names of categories with col_names
    categories_.columns = column_names
    # convert category values to numeric based on category column values
    for col in categories_.columns:
        # convert each column to numeric
        df[col] = pd.to_numeric(categories_[col].astype(str).apply(lambda x: x[-1]))
    
    # drop the categories column from df
    df.drop('categories', axis = 1, inplace = True)
                                
    # drop_duplicates
    df = df.drop_duplicates()
    
    return df
    

                  
        
    
    


def save_data(df, database_filename):
    """
        Saving data to given database_filename
        arguments:
            df (DataFrame) : dataframe to save
            database_filename (path): path to save databasefile
    """
    engine = create_engine('sqlite:///'+database_filename)
    
    df.to_sql('Table_1', engine, index=False, if_exists='replace')  


def main():
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