import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from numpy import ravel
le = preprocessing.LabelEncoder()
imp_mean = SimpleImputer(missing_values=np.NaN, strategy='mean')


class Convert(BaseEstimator, TransformerMixin):
    """ Convert float type that marked as 'object' type to float """
    def fit(self, X, y=None):
        for c in X:
            if X[c].dtype.name == 'object':
                X[c] = X[c].astype(float, errors='raise')
        self.X = X
        return self

    def transform(self, X):
        return self.X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.X

    def get_feature_names_out(self, lst=None):
        return self.X.columns


class FillNan(BaseEstimator, TransformerMixin):
    """
    class filling nan values using TransformerMixin with mean function
    """

    def fit(self, X, y=None):
        self.X = X
        if X.isnull().values.any():
            X_df = pd.DataFrame(imp_mean.fit_transform(X))
            X_df.columns = X.columns
            X_df.index = X.index
            self.X = X_df
        return self

    def transform(self, X):
        return self.X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.X

    def get_feature_names_out(self, lst=None):
        return self.X.columns


def y_to_categorical(y):
    """
    y column (target) can contains a string datatypes, hence convert to int (also nan)
    :param y: target column
    :return: y encoded to int
    """
    return le.fit_transform(ravel(y))
