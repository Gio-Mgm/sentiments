import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib


df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1
)

model = LogisticRegression(max_iter=700, multi_class="multinomial", C=100)
model.fit(X_train, y_train)
joblib.dump(model, './models/dumps/logistic_regression.joblib', compress=3)
