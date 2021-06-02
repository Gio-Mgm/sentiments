import joblib
import pandas as pd
import numpy as np
#import autokeras as ak
from sentence_transformers import SentenceTransformer
#from models.mods.model_keras_sequential import fit_sequential
#from models.mods.logistic_regression import fit_logit_bert
#from tensorflow.keras.models import load_model

PAGES = [
    "Visualisation",
    "Résultats",
    "Prédiction en direct"
]

CLASSES = [
    "negative",
    "neutral",
    "positive",
    "very negative",
    "very positive",
]

MODELS_NAMES = [
    "LogisticRegression",
    "RandomForestClassifier",
    "XGBoostClassifier",
    #"Sequential",
    #"Auto Keras"
]

MODELS_LOAD = [
    joblib.load('./models/dumps/logistic_regression.joblib'),
    joblib.load('./models/dumps/random_forest.joblib'),
    joblib.load('./models/dumps/xgbc.joblib'),
    #fit_sequential(),
    #load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
]


MODELS = dict(zip(MODELS_NAMES, MODELS_LOAD))

