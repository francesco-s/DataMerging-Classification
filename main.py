import time

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from preprocessing.data_cleaning import clean_data, merge_data
from models.logistic_regression import log_regression_cv
from models.random_forest import random_forest_cv
from models.svm_classifier import svm_classifier_cv
from evaluation.evaluation_metrics import plot_f1_scores

if __name__ == '__main__':
    # Load data
    df1 = pd.read_csv('data/Training_part1.csv', sep=";")
    df2 = pd.read_csv('data/Training_part2.csv', sep=";")

    # Define feature groups
    int_features = ['BIB', 'WET']
    float_features = ['FAN', 'LUK', 'NUS', 'SIS', 'UIN']
    cat_features = ['COD', 'ERG', 'GJAH', 'MYR', 'PKD', 'RAS', 'TOK', 'VOL', 'KAT', 'XIN']

    # Step 1: Clean individual dataframes
    df1_cleaned = clean_data(df1, int_features, float_features, cat_features)
    df2_cleaned = clean_data(df2, int_features, float_features, cat_features)

    # Step 2: Merge the cleaned dataframes
    df3 = merge_data(df1_cleaned, df2_cleaned, merge_columns=["id"])

    # Step 3: Drop unnecessary columns and adjust feature lists
    df3.drop(['RAS', 'id'], axis=1, inplace=True)
    cat_features.remove('RAS')

    # Step 4: Convert categorical features to dummy variables
    df3 = pd.get_dummies(df3, columns=cat_features, drop_first=True)

    # Step 5: Prepare features (X) and target (y)
    X = df3.loc[:, df3.columns != 'Class'].values
    y = df3.loc[:, 'Class'].values

    # Step 6: Perform cross-validation for each model on the entire dataset
    cv_folds = 5

    print("Running Logistic Regression with Cross-Validation...")
    start_time = time.time()
    lr_pred = log_regression_cv(X, y, cv=cv_folds)
    cv_time = time.time() - start_time
    print(f"Time taken: {cv_time:.2f} seconds")
    print("Accuracy:", accuracy_score(y, lr_pred))
    print("=== Classification Report (Logistic Regression) ===")
    print(classification_report(y, lr_pred))

    print("Running Random Forest Classifier with Cross-Validation...")
    start_time = time.time()
    rf_pred = random_forest_cv(X, y, cv=cv_folds)
    cv_time = time.time() - start_time
    print(f"Time taken: {cv_time:.2f} seconds")
    print("Accuracy:", accuracy_score(y, rf_pred))
    print("=== Classification Report (Random Forest) ===")
    print(classification_report(y, rf_pred))

    print("Running Support Vector Machine with Cross-Validation...")
    start_time = time.time()
    svm_pred = svm_classifier_cv(X, y, cv=cv_folds)
    cv_time = time.time() - start_time
    print(f"Time taken: {cv_time:.2f} seconds")
    print("Accuracy:", accuracy_score(y, svm_pred))
    print("=== Classification Report (SVM) ===")
    print(classification_report(y, svm_pred))

    # Step 7: Plot F1-Scores for comparison
    print("Plotting F1-Scores...")
    plot_f1_scores(y,
                   {'Logistic Regression': lr_pred,
                    'Random Forest': rf_pred,
                    'SVM': svm_pred})