import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def plot_feature_distributions(df):
    df.hist(bins=20, figsize=(16, 10), color="#5DADE2")
    plt.suptitle("Feature Distributions", fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix", fontsize=14)
    plt.show()
