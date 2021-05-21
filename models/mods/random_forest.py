import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

vectorizer = CountVectorizer(ngram_range=(1, 2))
y = df["sentiment"]
X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1
)

model = RandomForestClassifier(
    random_state=0,
    criterion="gini",
    class_weight="balanced_subsample",
    n_estimators=200,
    min_samples_leaf=5
)

model.fit(X_train, y_train)

joblib.dump(model, './models/dumps/random_forest.joblib', compress=3)
