import joblib
from models.autokeras_model import fit_sequential
from tensorflow.keras.models import load_model
import autokeras as ak

PAGES = [
    "Visualisation",
    "Résultats"
]

CLASSES = [
    "anger",
    "boredom",
    "enthusiasm",
    "fear",
    "fun",
    "happy",
    "hate",
    "love",
    "neutral",
    "relief",
    "sadness",
    "surprise",
    "worry"
]


MODELS_NAMES = [
    "LogisticRegression",
    "RandomForestClassifier",
    "XGBoostClassifier",
    "Sequential",
    "Auto Keras"
]
MODELS_LOAD = [
    joblib.load('./models/dumps/logistic_regression.joblib'),
    joblib.load('./models/dumps/random_forest.joblib'),
    joblib.load('./models/dumps/xgbc.joblib'),
    fit_sequential(),
    load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
]


MODELS_DICT = dict(zip(MODELS_NAMES, MODELS_LOAD))

