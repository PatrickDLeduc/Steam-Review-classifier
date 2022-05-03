#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle, re
import nltk
from nltk.stem import PorterStemmer
app = Flask(__name__)


@app.route('/')
def temp():
    return render_template('template.html')

@app.route('/',methods=['POST','GET'])
def get_input():
    if request.method == 'POST':
        info = request.form['search']
        return redirect(url_for('run_pred',values=info))

@app.route('/run_pred/<values>')
def run_pred(values):
    import numpy as np

    replaceDict = dict({
    '{':" ", '}':" ", ',':"", '.':" ", '!':" ", '\\':" ", '/':" ", '$':" ", '%':" ",
    '^':" ", '?':" ", '\'':" ", '"':" ", '(':" ", ')':" ", '*':" ", '+':" ", '-':" ",
    '=':" ", ':':" ", ';':" ", ']':" ", '[':" ", '`':" ", '~':" ", '☑': ' ', '☐':' ',
    })

    rep = dict((re.escape(k),v) for k, v in replaceDict.items())
    pattern = re.compile('|'.join(rep.keys()))
    def replacer(text):
        """
        Removes punctuation and symbols
        """
        return rep[re.escape(text.group(0))]

    stopwords = nltk.corpus.stopwords.words("english")

    def remove_stopwords(lst):
        """
        Removes stopwords from list of word tokens
        Returns a string
        """
        return ' '.join(word for word in lst if word not in stopwords)

    def get_stems(lst):
        """
        Takes a list of words and stems them using PorterStemmer
        Returns a string with all words stemmed
        """
        ps = PorterStemmer()
        return ' '.join(ps.stem(w) for w in lst.split())
        
        
    # with open('tfidf_sentiment_best.pkl', 'rb') as file:
        # tf = pickle.load(file)
        
    # with open('sentiment_classifier_best.pkl', 'rb') as file:
        # pickle_model = pickle.load(file)
    # values = pd.Series(values)
    # values = values.str.replace(pattern, replacer).str.lower().str.split()
    # values = values.apply(remove_stopwords)
    # values = values.apply(get_stems)
    # values = tf.transform(values)
    

        
    # sent_model = pickle_model
    # sent_pred = sent_model.predict(values)
    
    
    with open('tfidf_genre_best.pkl', 'rb') as file:
        tf_genre = pickle.load(file)


    
    values = pd.Series(values)
    values = values.str.replace(pattern, replacer).str.lower().str.split()
    values = values.apply(remove_stopwords)
    values = values.apply(get_stems)
    values = tf_genre.transform(values)
    
    with open('genre_classifier_best.pkl', 'rb') as file:
        genre_model = pickle.load(file)
        
    genre_pred = genre_model.predict(values)
    
    return genre_pred[0]
    
    if sent_pred == False:
        return 'Our model predicts that the review is negative!'
    return 'Our model predicts that the review is positive!'
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)

