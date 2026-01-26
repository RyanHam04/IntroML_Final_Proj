import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame, train_size: float, val_size: float, test_size: float):
    if train_size + val_size + test_size > 1:
        raise ValueError("Distribution not allowed, will cause leakage")

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=67, stratify=y)

    temp_test = test_size / (val_size + test_size)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=temp_test, random_state=67, stratify=y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

