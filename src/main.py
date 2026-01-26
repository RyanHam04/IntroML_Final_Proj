from data import dataloader
from preprocess import preprocess

def main():
    df = dataloader.load_dataset()
    data =  preprocess.split_data(df, 0.7, 0.2, 0.1)


if __name__ == "__main__":
    main()
