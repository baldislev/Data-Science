import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.feature_selection import SelectKBest, f_regression


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda
        self.weights_ = None

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        y_pred = None
        y_pred = X @ self.weights_.reshape((-1, 1))

        return y_pred.reshape(-1)

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        w_opt = None
        N, n_features = X.shape
        I = np.eye(n_features)
        I[0, 0] = 0
        w_opt = np.linalg.inv(X.T @ X + N * self.reg_lambda * I) @ (X.T @ y)

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    y = df[target_name]
    if feature_names is None:
        X = df.drop(target_name, axis=1).to_numpy()
    else:
        X = df[feature_names].to_numpy()
    y_pred = model.fit_predict(X, y)
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        xb = None
        ones = np.ones((X.shape[0], 1), dtype=int)
        xb = np.concatenate((ones, X), axis=1)

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2, k='all'):
        self.degree = degree
        self.k = k
        self.indices = None
        self.poly = None

    def fit(self, X, y=None):
        X_copy = np.copy(X)
        poly = PolynomialFeatures(self.degree, include_bias=False)
        self.poly = poly.fit(X_copy)

        if y is not None:
            np.seterr(divide='ignore', invalid='ignore')
            X_temp = poly.fit_transform(X_copy)
            k = 'all'
            if isinstance(self.k, int) and self.k < X_temp.shape[1]:
                k = self.k
            select = SelectKBest(k=k, score_func=f_regression)
            select.fit(X_temp, y)
            indices = select.get_support(indices=True)
            self.indices = indices
            np.seterr(divide='log', invalid='log')
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        X_transformed = None
        X_transformed = self.poly.transform(X)
        if self.indices is not None:
            X_transformed = X_transformed[:, self.indices]

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """
    cor_matrix = df.corr()
    cor_target = abs(cor_matrix[target_feature])
    relevant_features = cor_target.sort_values(ascending=False)[1:n+1]
    top_n_features = list(relevant_features.keys())
    top_n_corr = list(relevant_features)

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    diff = np.subtract(y, y_pred)
    squared_array = np.square(diff)
    mse = squared_array.mean()
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    data_mean = np.mean(y)
    residual = y - y_pred
    r2 = 1 - np.sum(np.square(residual))/np.sum(np.square(y - data_mean))
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """
    k_range = np.arange(50, 100, 5)
    all_scores = {}
    for d in degree_range:
        for l in lambda_range:
            for k in k_range:
                model.set_params(**{
                    'bostonfeaturestransformer__degree': d,
                    'linearregressor__reg_lambda': l,
                    'bostonfeaturestransformer__k': k
                })
                cv = sklearn.model_selection.ShuffleSplit(n_splits=k_folds, test_size=0.3, random_state=0)
                scores = sklearn.model_selection.cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
                all_scores[f'{(d, l, k)}'] = scores.mean()

    all_scores_sorted = dict(sorted(all_scores.items(), key=lambda item: item[1]))
    best_item = list(all_scores_sorted.keys())[-1]
    deg, lam, k_param = eval(best_item)
    best_params = {
        'bostonfeaturestransformer__degree': deg,
        'linearregressor__reg_lambda': lam,
        'bostonfeaturestransformer__k': k_param
    }
    return best_params
