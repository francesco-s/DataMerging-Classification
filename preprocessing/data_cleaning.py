import numpy as np


def clean_data(df, int_features, float_features, cat_features):
    """
    Clean the individual dataframe by removing duplicates, filling missing values,
    and converting categorical features to dummy variables.

    Parameters:
    - df: DataFrame to clean
    - int_features: List of integer feature names
    - float_features: List of float feature names
    - cat_features: List of categorical feature names

    Returns:
    - Cleaned DataFrame
    """

    # Ensure that we are only processing columns that exist in the DataFrame
    int_features = [col for col in int_features if col in df.columns]
    float_features = [col for col in float_features if col in df.columns]
    cat_features = [col for col in cat_features if col in df.columns]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Replace blank slots with NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Fill missing values for numerical features
    for column in int_features + float_features:
        df[column] = df[column].fillna(value=df[column].mean())

    # Fill missing values for categorical features
    for column in cat_features:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df


def merge_data(df1, df2, merge_columns):
    """
    Merge two dataframes on specified columns.

    Parameters:
    - df1: First DataFrame
    - df2: Second DataFrame
    - merge_columns: List of column names to merge on

    Returns:
    - Merged DataFrame
    """
    return df1.merge(df2, on=merge_columns, how='outer')
