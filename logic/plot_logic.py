"""
plot_logic.py
All Matplotlib / Seaborn chart generators.
Every function returns a matplotlib Figure — zero Streamlit imports.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="darkgrid", palette="deep")


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Seaborn correlation heatmap for all numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(max(6, len(num_df.columns)), max(4, len(num_df.columns) - 1)))
    sns.heatmap(
        num_df.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.5, ax=ax
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def distribution_grid(df: pd.DataFrame, max_cols: int = 6) -> plt.Figure:
    """Plot histograms for all numeric columns in a responsive grid."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    n = len(num_cols)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No numeric columns found", ha="center")
        return fig

    cols_per_row = 3
    rows = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Univariate Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def missing_bar(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of missing value percentages per column."""
    miss = (df.isnull().mean() * 100).sort_values(ascending=True)
    miss = miss[miss > 0]
    if miss.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No missing values detected ✓", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(8, max(3, len(miss) * 0.5)))
    bars = ax.barh(miss.index, miss.values, color="salmon")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values by Column", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    fig.tight_layout()
    return fig


def roc_curve_plot(fpr, tpr, roc_auc: float) -> plt.Figure:
    """ROC curve with shaded AUC area."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.fill_between(fpr, tpr, alpha=0.15, color="darkorange")
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.03])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Credit Risk Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    return fig


def anomaly_scatter(df: pd.DataFrame, x_col: str, y_col: str, flag_col: str = "is_anomaly") -> plt.Figure:
    """Scatter plot highlighting anomalous transactions in red."""
    fig, ax = plt.subplots(figsize=(10, 6))
    normal    = df[df[flag_col] == 0]
    anomalies = df[df[flag_col] == 1]

    ax.scatter(normal[x_col],    normal[y_col],    alpha=0.4, s=40,  color="steelblue", label="Normal")
    ax.scatter(anomalies[x_col], anomalies[y_col], alpha=0.9, s=120, color="crimson",
               edgecolors="black", linewidths=0.8, label=f"Anomaly ({len(anomalies)})")

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title("K-Means Anomaly Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig
