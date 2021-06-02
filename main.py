from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from CONST import MODELS

df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)

def predict_input(input, df):
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    vectorizer.fit_transform(df["lemma"].apply(lambda x: np.str_(x)))
    enc = vectorizer.transform([input])
    results = []
    for mod in MODELS:
            model = MODELS.get(mod)
            pred = model.predict(enc)[0]
            results.append([mod,pred])
    return results

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/prediction/{query}")
def get_prediction(query):
    response = predict_input(query, df)
    models = {}
    for el in response:
        models[el[0]] = str(el[1])

    output = {
        "query": query,
        "response": models
    }
    return output
