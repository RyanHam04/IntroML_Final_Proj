import os
from src.data import load_dataset
import pandas as pd
import matplotlib.pyplot as plt


LABEL_COL = "DEATH_EVENT"
OUTPUT_DIR = "src/graphics"


def plot_feature_distribution(df: pd.DataFrame, feature: str):
    """
    Plots the distribution of a single feature.
    """
    plt.figure()
    plt.hist(df[feature])
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(f"Distribution of {feature}")
    plt.savefig(f"{OUTPUT_DIR}/{feature}_distribution.png")
    plt.close()

def graph_target_vs_another_feature(df: pd.DataFrame, feature: str):
    """
    Plots the distribution of the death event against other features.
    """
    
    alive = df[df[LABEL_COL] == 0][feature]
    deceased = df[df[LABEL_COL] == 1][feature]
    
    plt.figure()
    plt.hist(alive, bins=30, alpha=0.7, label="Alive")
    plt.hist(deceased, bins=30, alpha=0.7, label="Deceased")

    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(f"{feature} by Outcome")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{feature}_by_label.png")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset()

    numeric_features = df.drop(LABEL_COL, axis=1).select_dtypes(include="number").columns

    for feature in numeric_features:
        plot_feature_distribution(df, feature)
        graph_target_vs_another_feature(df, feature)


if __name__ == "__main__":
    main()
