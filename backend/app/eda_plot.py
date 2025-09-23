import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import io
import base64

# Use seaborn style
sns.set_theme(style="whitegrid")

def save_plot_bytes(fig):
    """Save matplotlib figure to in-memory bytes and return base64 string."""
    buf = io.BytesIO()                      # 1
    fig.savefig(buf, format="png", bbox_inches="tight")  # 2
    plt.close(fig)                          # 3
    buf.seek(0)                             # 4
    return base64.b64encode(buf.getvalue()).decode("utf-8")  # 5



def univariate_plots(df):
    """Generate histograms & boxplots for numerical, barplots for categorical."""
    images = {}

    # Numerical features
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        images[f"hist_{col}"] = save_plot_bytes(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        images[f"box_{col}"] = save_plot_bytes(fig)
        plt.close(fig)

    # Categorical features
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df[col], order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Countplot of {col}")
        plt.xticks(rotation=45)
        images[f"count_{col}"] = save_plot_bytes(fig)
        plt.close(fig)

    return images


def bivariate_plots(df, target=None, output_dir="eda_outputs"):
    """Generate scatterplots for numeric pairs and boxplots for numeric vs categorical."""
    images = {}
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Scatterplot matrix (pairplot)
    if len(num_cols) > 1:
        fig = sns.pairplot(df[num_cols])
        fig.figure.suptitle("Pairplot of Numerical Features", y=1.02)
        images["pairplot"] = save_plot_bytes(fig)
        plt.close(fig)
    # Numeric vs Target (if target provided)
    if target and target in cat_cols:
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[target], y=df[col], ax=ax)
            ax.set_title(f"{col} by {target}")
            images[f"box_{col}_by_{target}"] = save_plot_bytes(fig)
            plt.close(fig)

    return images


def correlation_heatmap(df, output_dir="eda_outputs"):
    """Generate correlation heatmap for numerical features."""
    
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = df[num_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        return {"correlation_heatmap": save_plot_bytes(fig)}
    return {}


def generate_eda(df, target=None, output_dir="eda_outputs"):
    """Run all EDA steps and save plots. Returns list of file paths."""
    paths = []
    paths.extend(univariate_plots(df, output_dir))
    paths.extend(bivariate_plots(df, target, output_dir))
    paths.extend(correlation_heatmap(df, output_dir))
    return paths


if __name__ == "__main__":
    # Example run with Iris dataset
    iris = pd.read_csv(Path(__file__).parent/'data'/'iris.csv')  # assumes species column
    plot_files = generate_eda(iris, target="species")
    print("EDA Plots saved:", plot_files)
