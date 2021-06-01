import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv("./data/02/emotions_full.csv")

bert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
encoded = bert.encode(df['lemma'].astype('str'), show_progress_bar=True)
print(encoded.shape)
np.save("./data/02/encoded.npy", encoded)
test = np.load("./data/02/encoded.npy")
print(test.shape)
