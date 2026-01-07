# precision / recall / F1
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text

df = pd.read_csv("backend/data/processed/ipc_training_data.csv")
df["case_text"] = df["case_text"].apply(clean_text)
df["ipc_sections"] = df["ipc_sections"].apply(lambda x: str(x).split(","))

model = joblib.load("ipc_model.pkl")
mlb = joblib.load("ipc_labels.pkl")

X = df["case_text"]
y = mlb.transform(df["ipc_sections"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=mlb.classes_))
