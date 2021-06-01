import autokeras as ak
import kerastuner as kt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense

def fit_sequential():
    df = pd.read_csv("./data/02/emotions_full.csv", index_col=0)
    train_df = pd.DataFrame()
    MAX_COUNT = 179
    MAX_WORDS = 5000
    for el in df.feeling.unique():
            to_add = df[df["feeling"] == el][:MAX_COUNT]
            train_df = pd.concat([train_df, to_add])

    train_df.feeling.value_counts()
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    d_ = df[["lemma"]]
    tokenizer.fit_on_texts(d_["lemma"].astype('str'))
    le = LabelEncoder()
    y = le.fit_transform(df['feeling'])
    X = tokenizer.texts_to_matrix(df["lemma"].astype('str'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=1
    )
    model = Sequential()
    model.add(Dense(208, activation='relu', input_shape=[MAX_WORDS]))
    model.add(Dropout(0.3))
    model.add(Dense(104, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(26, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(13, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train)

    # export_model = model.export_model()

    # print(type(model))  # <class 'tensorflow.python.keras.engine.training.Model'>

    # try:
    #     export_model.save(".models/model_sequential", save_format="tf")
    # except Exception:
    #     export_model.save(".models/model_sequential.h5")

    #joblib.dump(model, './models/dumps/keras_sequential.joblib', compress=3)
    return model, X_test, y_test
