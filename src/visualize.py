import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance


def plot_class_distribution(df, output_dir):
    plt.figure()
    df["Category"].value_counts().plot(kind="bar")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"))
    plt.close()

def plot_feature_correlation(df, output_dir):
    plt.figure(figsize=(10, 8))
    corr = df.drop(columns=["Category"]).corr()
    sns.heatmap(corr, cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"))
    plt.close()


def plot_feature_importance(model, X, y, feature_names, output_dir):
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importances = pd.Series(
        result.importances_mean,
        index=feature_names
    ).sort_values()

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

