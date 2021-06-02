
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from CONST import MODELS

def prepare_data(df):
    """
        uses CountVectorizer to transform data, 
        and split data
        (models are already fitted)
    
        params:
            df: dataframe used
    
        returns:
            X_test,
            y_test
    """
    
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    y = df["feeling"]
    X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=.2, random_state=1)

    return X_test, y_test

def get_results(df, mod):
    """
        get results of model
    
        params:
            df: dataframe used,
            mod: model used
            
    
        returns:
            y_test: y_true values,
            y_probas: probabilities for each class
    """
    
    empty = st.empty()
    with empty:
        st.info("loading...")
    if mod == "Sequential":
        model, X_test, y_test = MODELS.get(mod)
    else:
        model = MODELS.get(mod)
        X_test, y_test = prepare_data(df)
    y_probas = model.predict_proba(X_test)

    with empty:
        st.info("Model loaded")
        time.sleep(2)
    empty.empty()
    return y_test, y_probas


def plot_results(y_true, y_probas, title, classes_to_plot=None):
    """
        plots ROC_AUC of each class
    
        params:
            y_true,
            y_probas, 
            title: title of the graph, 
            classes_to_plot: list of curves to plot
            
    
        returns:
            fig: matplotlib figure
    """
    
    fig, ax = plt.subplots()
    skplt.metrics.plot_roc(
        y_true, y_probas,
        classes_to_plot=classes_to_plot,
        ax=ax,
        title=title,
        cmap="tab20",
        plot_macro=False
    )
    return fig

def classify_input(input, df):
    """
        uses CountVectorizer on input,
        get predictions and probabilities
    
        params:
            input: user input,
            df: dataframe used
    
        returns:
            list of name, prediction and probabilities,
            for each model
    """
    
    #bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    #enc_bert = bert.encode(df['lemma'].astype('str'))
    
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    enc = vectorizer.transform([input])

    results = []
    for mod in MODELS:
        model = MODELS.get(mod)
        # if mod.endswith("(BERT)"):
        #     enc = enc_bert
        # else:
        #     enc = enc_vect
        pred = model.predict(enc)
        probas = model.predict_proba(enc)
        results.append([mod, pred, probas])
    return results

def show_pie_chart(df):
    """
        plot pie chart
    
        params:
            df: dataframe used
    
        returns:
            fig: matplotlib figure    
    """
    
    t = pd.crosstab(df.feeling, "freq")
    t.reset_index(inplace=True)

    fig = plt.figure(figsize=(10, 10))

    fig.patch.set_facecolor('white')
    # plot chart
    ax1 = plt.subplot(111, aspect='equal')
    t.plot(kind='pie', y='freq', ax=ax1, pctdistance=1.1, autopct='%1.1f%%',
        startangle=90, shadow=False, labels=None, legend=False, fontsize=14)
    plt.legend(labels=t['feeling'], bbox_to_anchor=(0, 0.85), fontsize=20)
    axes = plt.gca()
    axes.set_title(' Frequency of feelings', fontsize=20, color='Black')

    axes.xaxis.label.set_color("black")
    axes.yaxis.label.set_color("black")

    return fig
