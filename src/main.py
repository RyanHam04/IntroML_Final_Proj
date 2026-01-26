from data import dataloader
from preprocess import preprocess

def main():
    df = dataloader.load_dataset()
    print(f"Data before splitting {df.shape}")
    X_train, X_val, X_test, y_train, y_val, y_test =  preprocess.split_data(df, 0.7, 0.2, 0.1)


if __name__ == "__main__":
    main()
