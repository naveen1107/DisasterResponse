import json
import plotly
import pandas as pd


# download necessary nltk packages
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
# load data
db_name = 'DisasterResponse.db'
engine = create_engine('sqlite:///../data/'+db_name)
df = pd.read_sql_table('Table_1', engine)
#print(df.head())
# load model
model_name = 'classifier.pkl'
model = joblib.load("../models/"+model_name)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    
    df_top_10_cat = df.iloc[:, 4:].sum(axis = 0).sort_values(ascending = False)[:10]
    top_10_cat_names = list(df_top_10_cat.index.values)
    top_10_cat_values = list(df_top_10_cat.values)
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }   
    ]
    
    graph_2 =    {
        'data': [
            Bar(
                x=top_10_cat_names,
                y=top_10_cat_values
            )
        ],

        'layout': {
            'title': 'Top 10 categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Categories"
            }
        }
    } 
    graphs.append(graph_2)
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()