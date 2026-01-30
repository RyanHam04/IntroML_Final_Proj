import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict


FILENAME = "src/heart_failure_clinical_records_dataset.csv"
LABEL_COL = "DEATH_EVENT"


def load_dataset(file_name: str = FILENAME) -> pd.DataFrame:
    """
    Loads the data from the specified file path as a
    dataframe
    """
    df = pd.read_csv(file_name)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validating that the dataset has the expected structure.
    Raise validation errors if validation fails.
    """
    if df.empty:
        raise ValueError("Datadset is empty")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    if df[LABEL_COL].nunique() != 2:
        raise ValueError("Label column is not binary.")

    if df.isnull().any().any():
        print("Warning: Dataset contains missing values.")

    non_numeric_cols = (
        df.drop(LABEL_COL, axis=1).select_dtypes(exclude="number").columns
    )
    if len(non_numeric_cols) > 0:
        raise ValueError(f"Non-numeric feature columns found: {list(non_numeric_cols)}")

    if df.isnull().any().any():
        print("Warning: Dataset contains missing values.")


def split_data(
    df: pd.DataFrame, train_size: float, val_size: float, test_size: float
) -> Dict[str, pd.DataFrame]:
    """
    Splits the data into the defined sizes. The program does not check if the sizes
    are valid.

    This method will probably be trimmed as we use K-fold and do therefore not need
    a separate validation set for now. However, I will keep it here.

    BEFORE SUBMISSION READ THIS AND DELETE IF NEEDED
    """

    X = df.drop("DEATH_EVENT", axis=1)
    y = df["DEATH_EVENT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=67, stratify=y
    )

    temp_test = test_size / (val_size + test_size)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, train_size=temp_test, random_state=67, stratify=y_test
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def kfold_data(data):
    """
    As `split_data` divides the data into train/val/test,
    and K-fold requires only a train set, we combine val and train

    """
    X_train = pd.concat([data["X_train"], data["X_val"]])
    y_train = pd.concat([data["y_train"], data["y_val"]])

    return {
        "X_train": X_train,
        "y_train": y_train,
    }
