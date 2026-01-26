import pandas as pd


FILENAME = "src/data/heart_failure_clinical_records_dataset.csv"

def load_dataset(file_name: str = FILENAME) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    return df

if __name__ == "__main__":
    df = load_dataset(FILENAME)
    print(f"Data before splitting {df.shape}")
