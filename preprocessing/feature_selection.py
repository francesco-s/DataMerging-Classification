import pandas as pd
import numpy as np


def feature_to_drop(df):
    """ Identify highly correlated features to drop """
    cor = df.corr().abs()
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool_))
    to_drop_numerical = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop_numerical
