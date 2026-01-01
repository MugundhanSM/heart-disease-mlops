from pathlib import Path
import pandas as pd

def test_dataset_exists():
    data_path = Path("data/processed/heart_disease_cleaned.csv")
    assert data_path.exists(), "Processed dataset does not exist"

def test_dataset_not_empty():
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")
    assert len(df) > 0, "Dataset is empty"

def test_target_column_exists():
    df = pd.read_csv("data/processed/heart_disease_cleaned.csv")
    assert "target" in df.columns, "Target column missing"
    assert df["target"].nunique() == 2, "Target column should have exactly two unique values"