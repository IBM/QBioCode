# ====== Base class imports ======
import warnings

import hfda
import numpy as np
import pandas as pd
from scipy.linalg import eigvals, inv, norm
from scipy.spatial import ConvexHull as CH

# ====== Scipy imports ======
from scipy.stats import entropy
from skdim import id
from skdim.id import lPCA

# ====== Scikit-learn imports ======
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.manifold import Isomap
from sklearn.neighbors import KernelDensity


# df = pd.DataFrame(X)
def get_dimensions(df):
    """Get the number of features, samples, and feature-to-sample ratio from a DataFrame.
    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
    Returns:
        tuple: (num_features, num_samples, ratio)
            - num_features (int): Number of features in the DataFrame
            - num_samples (int): Number of samples in the DataFrame
            - ratio (float): Feature-to-sample ratio
    """
    # number of features
    num_features = df.shape[1]
    # of samples
    num_samples = df.shape[0]
    # feature-to-sample ratio
    ratio = num_features / num_samples

    return num_features, num_samples, ratio


def get_intrinsic_dim(df):
    """Get intrinsic dimension of the data using lPCA from skdim.
    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
    Returns:
        float: Intrinsic dimension of the data
    """
    # Intrinsic dimension, calculated via scikit-dimension's PCA method
    pca = id.lPCA()  # Initialize the PCA estimator from skdim
    pca.fit(df)  # Fit the estimator to your data
    return pca.dimension_


def get_condition_number(df):
    """Get the condition number of a matrix.

    A high condition number indicates that the matrix is ill-conditioned and
    can produce large output errors even for small input perturbations. A low
    condition number indicates a more stable matrix.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: Condition number of the matrix represented in ``df``.
    """
    # In general,
    # meaning that it can produce large errors in its output even with small errors in its input.
    # Conversely, a function with a low condition number is well-conditioned and more stable in terms of its output.
    return np.linalg.cond(df)


def get_fdr(df, y):
    """Calculate Fisher Discriminant Ratio for a given dataset.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        y (int): supervised binary class label

    Returns:
        float: Fisher Discriminant ratio
    """
    X = df.values
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    FDR = 0

    if n_classes != 2:
        warnings.warn("WARNING: Fisher Discriminant Ratio is only defined for binary classes. ")
    else:
        mean1 = np.mean(X[y == class_labels[0]], axis=0)  # mean for class1
        mean2 = np.mean(X[y == class_labels[1]], axis=0)  # mean for class2

        # calculate within-class scatter matrices
        scatter_within = np.zeros((X.shape[1], X.shape[1]))
        for label in class_labels:
            X_class = X[y == label]
            scatter_within += np.cov(X_class.T)

        # calculate between-class scatter matrix
        scatter_between = np.outer(mean1 - mean2, mean1 - mean2)

        # compute FDR
        FDR = np.trace(scatter_between) / np.trace(scatter_within)

    return FDR


def get_total_correlation(df):
    """Calculate Total Correlation

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: Total correlation
    """
    corr_matrix = df.corr()  # correlation matrix
    # total correlation by subtracting diagonal values to remove self-correlation
    total_correlation = corr_matrix.abs().sum().sum() - len(df.columns)

    return total_correlation


def get_mutual_information(df, y):
    """Calculate mutual information via sklearn

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        y (int): supervised binary class label

    Returns:
        float: Mutual information
    """
    mutual_info = np.mean(mutual_info_classif(df, y))

    return mutual_info


def get_variance(df):
    """Get variance

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        avg_var (float): Mean variance
        std_var (float): Standard deviation of variance
    """
    variations = round(df.var(), 2)
    avg_var = variations.mean()
    std_var = variations.std()

    return avg_var, std_var


def get_coefficient_var(df):
    """Get coefficient of variance

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        avg_co_of_v (float): Mean coefficient of variance
        std_var (float): Standard deviation of coefficient of variance
    """
    co_of_v = (df.std() / df.mean()) * 100
    avg_co_of_v = co_of_v.mean()
    std_co_of_v = co_of_v.std()

    return avg_co_of_v, std_co_of_v


def get_nnz(df):
    """Calculate nonzero values in the data

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        int: nonzero count
    """
    return np.count_nonzero(df.values)


def get_low_var_features(df, num_features):
    """Calculate get count of low variance features

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        num_features (int): number of features in the dataset

    Raises:
        ValueError: If no feature is strong enough to keep

    Returns:
        int: count of features with low variance
    """

    threshold = np.percentile(df.var(), 25)

    try:
        low_var_features = num_features - VarianceThreshold(threshold).fit(df).get_support().sum()
    except ValueError:
        print("No feature is strong enough to keep")
        low_var_features = None

    return low_var_features


def get_log_density(df):
    """Calculate the mean log density of the data

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        float: mean log kernel density
    """
    kde = KernelDensity(bandwidth=0.2, kernel="gaussian").fit(
        df
    )  # Create a KernelDensity estimator and fit the estimator to the data
    log_density = kde.score_samples(df)

    return log_density.mean()


def get_fractal_dim(df, k_max):
    """Calculate the fractal dimension of the data using Higuchi's method

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        k_max (int): Maximum number of k values to use in the calculation

    Returns:
        float: Fractal dimension of the data
    """
    FD = hfda.measure(df, k_max)

    return FD


def get_moments(df):
    """Compute third and fourth order moments of the data

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        avg_skew (float): Mean skewness
        std_skew (float): Standard deviation of skewness
        avg_kurt (float): Mean kurtosis
        std_kurt (float): Standard deviation of kurtosis
    """
    # Skewness
    skew = df.skew()
    avg_skew = skew.mean()
    std_skew = skew.std()
    # Kurtosis
    kurt = df.kurtosis()
    avg_kurt = kurt.mean()
    std_kurt = kurt.std()

    return avg_skew, std_skew, avg_kurt, std_kurt


def get_entropy(y):
    """Calculate entropy of the target variable

    Args:
        y (int): supervised binary class label

    Returns:
        avg_y_entropy (float): mean entropy
        std_y_entropy (flat): standard deviation of entropy
    """
    y_entropy = entropy(np.bincount(y), base=2)  # Compute the entropy of the target variable (y)
    avg_y_entropy = y_entropy.mean()
    std_y_entropy = y_entropy.std()

    return avg_y_entropy, std_y_entropy


def get_volume(df):
    """Get volume of the data from Convex Hull

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns

    Returns:
        volume (float): Volume of the space spanned by the features of the data
    """

    vol = 0
    if df.shape[0] <= df.shape[1]:
        warnings.warn("Convex Hull requires number of observations > number of features")
    else:
        vol = CH(df, qhull_options="QJ").volume

    return vol


def get_complexity(df, n_neighbors=10, n_components=2):
    """Measure the manifold complexity by fitting Isomap and analyzing the geodesic vs. Euclidean distances.
    This function computes the reconstruction error of the Isomap algorithm, which serves as an indicator of the complexity of the manifold represented by the data.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        n_neighbors: Number of neighbors for the Isomap algorithm. Default value 10
        n_components: Number of components (dimensions) for Isomap projection.  Default value 2

    Returns:
        - reconstruction_error: float
            The reconstruction error of the Isomap model, which indicates the complexity of the manifold.
        - reconstruction_error: The residual error of geodesic distances
    """

    isomap = Isomap(n_neighbors=10, n_components=2)
    isomap.fit(df.values)

    # reconstruction error - an indicator of complexity
    reconstruction_error = isomap.reconstruction_error()

    return reconstruction_error


def evaluate(df, y, file):
    """This function evaluates a dataset and returns a transposed summary DataFrame with various statistical measures, derived from the dataset.
    Using the functions defined above, it computes intrinsic dimension, condition number, Fisher Discriminant Ratio, total correlation, mutual information, variance, coefficient of variation,
    data sparsity, low variance features, data density, fractal dimension, data distributions (skewness and kurtosis), entropy of the target variable, and manifold complexity.
    The summary DataFrame is transposed for easier readability and contains the dataset name, number of features, number of samples, feature-to-sample ratio, and various statistical measures.
    This function is useful for quickly summarizing the characteristics of a dataset, especially in the context of machine learning and data analysis, allowing you to correlate the dataset's
    properties with its performance in predictive modeling tasks.

    Args:
        df (pandas.DataFrame): Dataset in pandas with observation in rows, features in columns
        y (int): supervised binary class label
        file (str): Name of the dataset file for identification in the summary DataFrame

    Returns:
        transposed (pandas.DataFrame): Summary DataFrame containing various statistical measures of the dataset
    """
    # Select only numeric columns from the DataFrame
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate statistical measures
    n_features, n_samples, feature_sample_ratio = get_dimensions(df_numeric)

    # get intrinsic dimension
    intrinsic_dim = get_intrinsic_dim(df_numeric)

    # Condition number
    condition_number = get_condition_number(df_numeric)

    # Class imbalance ratio via Fischer Discriminant
    fdr = get_fdr(df_numeric, y)

    # Total correlation
    total_correlation = get_total_correlation(df_numeric)

    # Mutual information
    mutual_info = get_mutual_information(df_numeric, y)

    # Variance
    avg_var, std_var = get_variance(df_numeric)

    # Coefficient of variance
    avg_co_of_v, std_co_of_v = get_coefficient_var(df_numeric)

    # Data sparsity
    count_nonzero = get_nnz(df)

    # Get the number of low variance features
    num_low_variance_features = get_low_var_features(df_numeric, n_features)

    # Data density
    mean_log_density = get_log_density(df_numeric)

    # Fractal Dimension
    k_max = 5
    fractal_dim = get_fractal_dim(df_numeric, k_max)

    # Data distributions
    avg_skew, std_skew, avg_kurt, std_kurt = get_moments(df_numeric)

    # entropy
    avg_y_entropy, std_y_entropy = get_entropy(y)

    # volume of data
    # volume = get_volume(df_numeric)

    # manifold complexity
    complexity = get_complexity(df_numeric)

    # Create summary DataFrame
    summary_df = pd.DataFrame.from_dict(
        {
            # Data set
            "Dataset": file,
            # Dimensions
            "# Features": n_features,
            "# Samples": n_samples,
            "Feature_Samples_ratio": feature_sample_ratio,
            # Intrinsic dimension
            "Intrinsic_Dimension": intrinsic_dim,
            # Condition number
            "Condition number": condition_number,
            # Class imbalance ratio
            "Fisher Discriminant Ratio": fdr,
            # Feature Correlations
            "Total Correlations": total_correlation,  # Total Correlations
            "Mutual information": mutual_info,  # Mutual information
            # Data sparsity
            "# Non-zero entries": count_nonzero,
            "# Low variance features": num_low_variance_features,
            #'Variation': variations,
            "Variation": avg_var,
            "std_var": std_var,
            #'Coefficient of Variation %': co_of_v,
            "Coefficient of Variation %": avg_co_of_v,
            "std_co_of_v": std_co_of_v,
            # Data distributions
            #'Skewness': skew,
            "Skewness": avg_skew,
            "std_skew": std_skew,
            #'Kurtosis': kurt,
            "Kurtosis": avg_kurt,
            "std_kurt": std_kurt,
            # Data density
            "Mean Log Kernel Density": mean_log_density,
            # volume of feature space
            #'Volume': volume,
            # Manifold complexity
            "Isomap Reconstruction Error": complexity,
            # Fractal dimension
            "Fractal dimension": fractal_dim,  # calculated via Higuchi Dimension
            #'Entropy': y_entropy,
            "Entropy": avg_y_entropy,
            "std_entropy": std_y_entropy,
        },
        orient="index",
    )

    transposed = summary_df.T
    # transposed.to_csv('DataSetEvaluation.csv', sep='\t', index=False)
    # print(transposed)
    return transposed


# evaluate(df,y)
