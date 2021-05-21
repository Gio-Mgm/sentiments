import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib


df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
le = LabelEncoder()
y = le.fit_transform(df['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=1
)

model = XGBClassifier(
    random_state=0,
    booster="dart",
    learning_rate=0.02,
    n_estimators=200,
    n_jobs=-1,
    objective="multi:softmax",
    use_label_encoder=False,
    eval_metric="merror")

model.fit(X_train, y_train)

joblib.dump(model, './models/dumps/xgbc.joblib', compress=3)
