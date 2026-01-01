import os
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

def download_data():
    os.makedirs("raw", exist_ok=True)

    df = pd.read_csv(URL, header=None, names=COLUMNS)

    df.to_csv("raw/heart_disease.csv", index=False)
    print("Dataset downloaded successfully")

if __name__ == "__main__":
    download_data()
