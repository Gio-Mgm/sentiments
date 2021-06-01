import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib

def fit_logit():
    df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    y = df["feeling"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=1
    )

    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=100)
    model.fit(X_train, y_train)
    joblib.dump(model, './models/dumps/logistic_regression.joblib', compress=3)

def fit_logit_bert():
    df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)
    y = df["feeling"]
    bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    X = bert.encode(df["lemma"].astype('str'))
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=1
    )

    model = LogisticRegression(max_iter=1000, multi_class="multinomial", C=100)
    model.fit(X_train, y_train)
    joblib.dump(
        model, './models/dumps/logistic_regression_bert.joblib', compress=3)

fit_logit_bert()
