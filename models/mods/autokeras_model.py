import autokeras as ak
import kerastuner as kt
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense

def run_auto_keras():
    df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)


    y = np.array(df["sentiment"].astype("str"))
    X = np.array(df["lemma"].astype("str"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=1
    )

    weights = class_weight.compute_class_weight(
        'balanced',
        np.unique(y_train),
        y_train
    )
    class_weights = dict(zip([i for i in range(len(weights))], weights))

    # Initialize the structured data classifier.
    model = ak.TextClassifier(
        overwrite=True,
        max_trials=4,
        metrics="accuracy",
        objective=kt.Objective("accuracy", direction="max"),
        loss="categorical_crossentropy"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        min_delta=0.001,  # minimium amount of change to count as an improvement
        patience=3,  # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    model.fit(X_train, y_train, epochs=100,
            class_weight=class_weights, callbacks=[early_stopping])

    # Export as a Keras Model.
    model.export_model()

    print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

    try:
        model.save("./models/model_autokeras", save_format="tf")
    except Exception:
        model.save("./models/model_autokeras.h5")
