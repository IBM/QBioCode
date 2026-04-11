import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# Set publication-quality defaults for scientific journals
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.minor.width"] = 0.8
plt.rcParams["ytick.minor.width"] = 0.8
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.05
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.5


def compute_results_correlation(results_df, correlation="spearman", thresh=0.7):
    """This function takes in as input a Pandas Dataframe containing the results and data evaluations for
    a given dataset.  It then produces a spearman correlation between the data evaluation characteristics (features)
    and instances where an F1 score was observed above a certain threshold (thresh).
    The function returns the input DataFrame with additional columns for datatype and model_embed_datatype,
    as well as a new DataFrame containing the computed correlations between metrics and features.
    The correlation is computed for each model-embedding-dataset combination, and the results are aggregated.
    The features considered for correlation include various data characteristics such as 'Feature_Samples_ratio', 'Intrinsic_Dimension', etc.
    The metrics considered for correlation include 'accuracy', 'f1_score', 'time', and 'auc'.
    The function also calculates the median metric value and the fraction of instances above the specified threshold for each combination.
    The resulting DataFrame contains the model-embedding-dataset, metric, feature, median metric value, fraction above threshold, and the computed correlation.
    This function is useful for understanding how different data characteristics relate to model performance metrics, particularly in the context of machine learning models applied to datasets.

    Args:
        results_df (pd.DataFrame): A DataFrame containing the results and data evaluations.
        correlation (str): The type of correlation to compute, default is 'spearman'.
        thresh (float): The threshold for F1 score to consider, default is 0.7.

    Returns:
        results_df (pd.DataFrame): The input DataFrame with additional columns for datatype and model_embed_datatype.
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.

    """

    # Refining datasrame
    results_df["datatype"] = [
        re.sub(r"\.csv", "", re.sub(r"-.*", "", x)) for x in results_df["Dataset"]
    ]
    results_df["model_embed_datatype"] = [
        "_".join([str(row.model), str(row.embeddings), str(row.datatype)])
        for idx, row in results_df.iterrows()
    ]

    correlations = []
    features = [
        "Feature_Samples_ratio",
        "Intrinsic_Dimension",
        "Condition number",
        "Fisher Discriminant Ratio",
        "Total Correlations",
        "Mutual information",
        "# Non-zero entries",
        "# Low variance features",
        "Variation",
        "std_var",
        "Coefficient of Variation %",
        "std_co_of_v",
        "Skewness",
        "std_skew",
        "Kurtosis",
        "std_kurt",
        "Mean Log Kernel Density",
        "Isomap Reconstruction Error",
        "Fractal dimension",
        "Entropy",
        "std_entropy",
    ]
    metrics = ["accuracy", "f1_score", "time", "auc"]

    keys = list(set(results_df["model_embed_datatype"]))
    for m in keys:
        dat_temp_m = results_df[results_df["model_embed_datatype"] == m]
        if len(dat_temp_m) > 0:
            for s in metrics:
                for f in features:
                    if f in dat_temp_m.columns:
                        if correlation == "spearman":
                            correlations.append(
                                [
                                    m,
                                    s,
                                    f,
                                    np.median(dat_temp_m[s]),
                                    sum(dat_temp_m[s] > thresh) / len(dat_temp_m[s]),
                                    spearmanr(dat_temp_m[s], dat_temp_m[f])[0],
                                ]
                            )

    correlations_df = pd.DataFrame(
        correlations,
        columns=[
            "model_embed_datatype",
            "metric",
            "feature",
            "median_metric",
            "frac_gt_thresh",
            "correlation",
        ],
    )

    return results_df, correlations_df


def plot_results_correlation(
    correlations_df,
    metric="f1_score",
    title="",
    correlation_type="Spearman ρ",
    figsize=(6.5, 10),
    save_file_path="",
    size="median_metric",
    xticks=True,
    key="model_embed_datatype",
    legend_offset=1.0,
    show_plots=True,
    colorbar_label="Correlation coefficient",
    size_label="Median metric value",
):
    """This function plots publication-quality correlation dot plots using the previously generated correlations_df dataframe.
    The larger the circle, the higher the metric value for that particular data set. The circle colors correspond to the
    correlations between the data characteristics (evaluations) and the metric. Red corresponds to a positive
    correlation, while blue indicates an anti-correlation. The strength of either type of correlation is represented by
    the shade of coloring -- the darker the circle, the more correlated/anticorrelated that particular characteristic is
    to the model's performance.

    Args:
        correlations_df (pd.DataFrame): A DataFrame containing the computed correlations between metrics and features.
        metric (str): The metric to plot, default is 'f1_score'.
        title (str): The title of the plot, default is an empty string.
        correlation_type (str): The type of correlation to display in the legend, default is 'Spearman ρ'.
        figsize (tuple): The size of the figure, default is (14, 9).
        save_file_path (str): The file path to save the plot, default is an empty string.
        size (str): The column name to use for the size of the dots, default is 'median_metric'.
        show_plots (bool): Whether to display plots, default is True.
        colorbar_label (str): Label for the colorbar, default is 'Correlation coefficient'.
        size_label (str): Label for the size legend, default is 'Median metric value'.

    Returns:
        None: Displays the plot and saves it to the specified file path if provided.
    """

    # Use enhanced professional diverging colormap
    from matplotlib.colors import LinearSegmentedColormap

    colors_custom = [
        "#053061",
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
        "#67001f",
    ]
    cmap_custom = LinearSegmentedColormap.from_list("custom_diverging", colors_custom, N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    # Sample data
    data = correlations_df[correlations_df["metric"] == metric].copy()
    data["feature"] = [
        re.sub(
            "std",
            "Std. dev. of",
            re.sub(
                "co of v",
                "coefficient of variation",
                re.sub(
                    "kurt$",
                    "kurtosis",
                    re.sub(
                        "skew$",
                        "skewness",
                        re.sub("var$", "variation", re.sub("%", "", re.sub("_", " ", x))),
                    ),
                ),
            ),
        )
        for x in data["feature"]
    ]

    if key == "model_datatype":
        data["datatype"] = ["_".join(x.split("_")[1:]) for x in data[key]]
        key_column = "Model / Dataset"
    else:
        data["datatype"] = ["_".join(x.split("_")[2:]) for x in data[key]]
        key_column = "Model / Embedding / Dataset"

    data = data.sort_values(["feature", "datatype"], ascending=False)
    data["model"] = [re.sub("_.*", "", x) for x in data[key]]
    data["model"] = [x.upper() for x in data["model"]]
    data = pd.concat(
        [
            data[~data["model"].isin(["QSVC", "QNN", "VQC", "PQK"])],
            data[data["model"].isin(["QSVC", "QNN", "VQC", "PQK"])],
        ]
    )
    fm = dict(zip(list(set(data["feature"])), range(len(set(data["feature"])))))
    data["feature_map"] = [fm[x] for x in data["feature"]]

    # Fill NaN values before scaling to avoid errors
    data = data.fillna(0)

    # Scale dot size based on actual data range for meaningful representation
    # Reduced sizes to minimize overlap
    epsilon = 25

    # Get actual min/max from the data to scale appropriately
    min_val = data[size].min()
    max_val = data[size].max()

    # Normalize to 0-1 based on actual data range, then scale to pixel sizes
    if max_val > min_val:
        normalized_values = (data[size] - min_val) / (max_val - min_val)
    else:
        normalized_values = np.ones_like(data[size]) * 0.5

    # Size formula: normalized value in [0,1] → size in [epsilon, 150+epsilon] (reduced from 200)
    data["norm_size"] = (normalized_values * 150 + epsilon).astype(float)

    data[key] = [re.sub("_", " / ", x) for x in data[key]]

    # Create figure with very compact design
    fig, ax = plt.subplots(figsize=figsize, facecolor="white", dpi=100)
    ax.set_facecolor("white")

    # Create scatter plot with enhanced professional styling
    scatter = ax.scatter(
        data[key],
        data["feature"],
        s=data["norm_size"],
        c=data["correlation"],
        cmap=cmap_custom,
        norm=norm,
        alpha=0.92,
        edgecolors="#34495E",
        linewidths=1.2,
        zorder=3,
    )

    # Add colorbar with enhanced professional styling
    cbar = plt.colorbar(scatter, ax=ax, pad=0.018, aspect=28, shrink=0.88)
    cbar.set_label(colorbar_label, rotation=270, labelpad=22, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.3, length=5, pad=4)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_edgecolor("#34495E")

    # Set labels with clean formatting
    ax.set_xlabel(key_column, fontweight="bold", fontsize=13, labelpad=10)
    ax.set_ylabel("Data Feature", fontweight="bold", fontsize=13, labelpad=10)

    # Add title if provided
    if title:
        ax.set_title(title, fontweight="bold", pad=20, fontsize=14)

    # Rotate x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha="right", va="top", fontsize=10)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=10)

    # Add professional grid for better readability
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.8, color="#95A5A6", zorder=0)
    ax.set_axisbelow(True)

    # Proper margins to prevent cropping while keeping columns close
    ax.margins(x=0.025, y=0.035)

    # Clean tick parameters
    ax.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=5)

    # Remove top and right spines for cleaner look
    sns.despine(ax=ax)

    # Create size legend with 4 dots showing ACTUAL median metric values from data
    handles_size, labels_size = scatter.legend_elements(
        prop="sizes", alpha=0.75, num=4, markeredgecolor="#34495E", markeredgewidth=1.2
    )

    # Use REAL median metric values from the data
    smin = np.min(data[size])
    smax = np.max(data[size])
    labels_size = [f"{x:.2f}" for x in np.linspace(smin, smax, 4)]

    # Position legend on the right side, well below the colorbar with proper spacing
    legend = ax.legend(
        handles_size,
        labels_size,
        title=size_label,
        loc="upper left",
        bbox_to_anchor=(1.15, -0.05),
        ncol=1,
        frameon=True,
        fancybox=False,
        title_fontsize=9,
        fontsize=8,
        edgecolor="#34495E",
        framealpha=0.98,
        labelspacing=0.8,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_facecolor("white")
    legend.get_title().set_fontweight("bold")

    # Adjust layout with reduced horizontal spacing between subplots
    plt.tight_layout(pad=0.8, w_pad=1.8)

    if save_file_path != "":
        plt.savefig(
            save_file_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="pdf" if save_file_path.endswith(".pdf") else None,
        )
        print(f"Scatter plot saved to: {save_file_path}")

    if show_plots:
        plt.show()
    plt.close()

    model_qml = ["QNN", "PQK", "VQC", "QSVC"]

    data[key_column] = data[key]
    data["Data feature"] = data["feature"]
    to_plot = data.pivot_table(columns=key_column, index="Data feature", values="correlation")

    # Define professional color scheme for model types
    ccolors = [
        "#7B68EE" if re.sub(" .*", "", x) in model_qml else "#FF8C00" for x in to_plot.columns
    ]

    # Create custom diverging colormap
    from matplotlib.colors import LinearSegmentedColormap

    colors_heatmap = [
        "#2166ac",
        "#4393c3",
        "#92c5de",
        "#d1e5f0",
        "#f7f7f7",
        "#fddbc7",
        "#f4a582",
        "#d6604d",
        "#b2182b",
    ]
    cmap_heatmap = LinearSegmentedColormap.from_list("custom_heatmap", colors_heatmap, N=256)

    # Create professional heatmap with better proportions
    heatmap_height = figsize[1] * 0.95  # Much taller to reduce space above colorbar
    heatmap_width = min(figsize[0] * 0.9, 10)  # Narrower columns

    g = sns.clustermap(
        to_plot.fillna(0),
        figsize=(heatmap_width, heatmap_height),
        col_colors=ccolors,
        cmap=cmap_heatmap,
        method="average",
        metric="euclidean",
        center=0,
        xticklabels=xticks,
        yticklabels=True,
        cbar_kws={"label": colorbar_label, "orientation": "horizontal"},
        linewidths=1.0,
        linecolor="white",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=0.05,
        cbar_pos=(0.55, 0.01, 0.4, 0.015),
    )

    # Hide dendrograms for cleaner appearance
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)

    # Improve axis labels with better styling
    g.ax_heatmap.set_xlabel(
        key_column, fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )
    g.ax_heatmap.set_ylabel(
        "Data Feature", fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )

    # Rotate x-labels 45 degrees for readability
    plt.setp(
        g.ax_heatmap.xaxis.get_majorticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
        color="#2C3E50",
    )
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=9, color="#2C3E50")

    # Improve tick parameters with better styling
    g.ax_heatmap.tick_params(
        axis="both", which="major", width=1.2, length=5, pad=4, colors="#2C3E50"
    )

    # Style heatmap spines
    for spine in g.ax_heatmap.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#34495E")

    # Enhance horizontal colorbar styling at bottom
    if g.cax is not None:
        g.cax.set_xlabel(
            colorbar_label, fontsize=10, fontweight="bold", labelpad=10, color="#2C3E50"
        )
        g.cax.tick_params(labelsize=9, width=1.2, length=4, colors="#2C3E50")
        for spine in g.cax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#34495E")

    if save_file_path != "":
        heatmap_path = re.sub(".pdf", "_heatmap.pdf", save_file_path)
        plt.savefig(
            heatmap_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="pdf" if heatmap_path.endswith(".pdf") else None,
        )
        print(f"Clustered heatmap saved to: {heatmap_path}")

    if show_plots:
        plt.show()
    plt.close()

    # Create non-clustered heatmap with quantum models first
    qml_col = [x for x in to_plot.columns if re.sub(" .*", "", x) in model_qml]
    cml_col = [x for x in to_plot.columns if re.sub(" .*", "", x) not in model_qml]
    to_plot_ordered = to_plot.loc[:, qml_col + cml_col]
    ccolors_ordered = [
        "#7B68EE" if re.sub(" .*", "", x) in model_qml else "#FF8C00"
        for x in to_plot_ordered.columns
    ]

    g2 = sns.clustermap(
        to_plot_ordered.fillna(0),
        figsize=(heatmap_width, heatmap_height),
        col_colors=ccolors_ordered,
        col_cluster=False,
        row_cluster=True,
        cmap=cmap_heatmap,
        center=0,
        xticklabels=xticks,
        yticklabels=True,
        cbar_kws={"label": colorbar_label, "orientation": "horizontal"},
        linewidths=1.0,
        linecolor="white",
        vmin=-1,
        vmax=1,
        dendrogram_ratio=0.05,
        cbar_pos=(0.55, 0.01, 0.4, 0.015),
        method="average",
        metric="euclidean",
    )

    # Improve axis labels with better styling
    g2.ax_heatmap.set_xlabel(
        key_column, fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )
    g2.ax_heatmap.set_ylabel(
        "Data Feature", fontweight="bold", fontsize=11, labelpad=12, color="#2C3E50"
    )

    # Rotate x-labels 45 degrees for readability
    plt.setp(
        g2.ax_heatmap.xaxis.get_majorticklabels(),
        rotation=45,
        ha="right",
        fontsize=9,
        color="#2C3E50",
    )
    plt.setp(g2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=9, color="#2C3E50")

    # Improve tick parameters with better styling
    g2.ax_heatmap.tick_params(
        axis="both", which="major", width=1.2, length=5, pad=4, colors="#2C3E50"
    )

    # Style heatmap spines
    for spine in g2.ax_heatmap.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor("#34495E")

    # Enhance horizontal colorbar styling at bottom
    if g2.cax is not None:
        g2.cax.set_xlabel(
            colorbar_label, fontsize=10, fontweight="bold", labelpad=10, color="#2C3E50"
        )
        g2.cax.tick_params(labelsize=9, width=1.2, length=4, colors="#2C3E50")
        for spine in g2.cax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#34495E")

    if save_file_path != "":
        noncluster_path = re.sub(".pdf", "_noncluster_heatmap.pdf", save_file_path)
        plt.savefig(
            noncluster_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="pdf" if noncluster_path.endswith(".pdf") else None,
        )
        print(f"Non-clustered heatmap saved to: {noncluster_path}")

    if show_plots:
        plt.show()
    plt.close()
