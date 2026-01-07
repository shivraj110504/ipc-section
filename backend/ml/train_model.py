# trains ML model
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = "backend/data/processed/ipc_training_data.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text

df = pd.read_csv(DATA_PATH)
df["case_text"] = df["case_text"].apply(clean_text)
df["ipc_sections"] = df["ipc_sections"].apply(lambda x: str(x).split(","))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["ipc_sections"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=30000)),
    ("clf", OneVsRestClassifier(LogisticRegression(max_iter=3000)))
])

model.fit(df["case_text"], y)

joblib.dump(model, "ipc_model.pkl")
joblib.dump(mlb, "ipc_labels.pkl")

print("IPC model trained and saved")
