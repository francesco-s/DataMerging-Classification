import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt


def log_regression(X_train, y_train, X_test, y_test):
    """ Logistic Regression Model """
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr', max_iter=4000)
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)

    print("Accuracy (logistic regression):", accuracy_score(y_test, lr_predict))
    print("=== Classification Report (logistic regression) ===")
    print(classification_report(y_test, lr_predict))
    print("=== Confusion Matrix (logistic regression) ===")
    print(confusion_matrix(y_test, lr_predict))

    return lr_predict


def random_forest(X_train, y_train, X_test, y_test):
    """ Random Forest Classifier """
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)

    print("Accuracy (random forest) :", accuracy_score(y_test, rfc_predict))
    print("=== Classification Report (random forest) ===")
    print(classification_report(y_test, rfc_predict))
    print("=== Confusion Matrix (random forest) ===")
    print(confusion_matrix(y_test, rfc_predict))

    return rfc_predict


def linear_svm(X_train, y_train, X_test, y_test):
    """ Linear Support Vector Machine """
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    svm_predict = clf.predict(X_test)

    print("Accuracy (linear svm):", accuracy_score(y_test, svm_predict))
    print("=== Classification Report (linear svm) ===")
    print(classification_report(y_test, svm_predict))
    print("=== Confusion Matrix (linear svm) ===")
    print(confusion_matrix(y_test, svm_predict))

    return svm_predict


def feature_todrop(df):
    """ Identify highly correlated features to drop """
    cor = df.corr().abs()
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool_))
    to_drop_numerical = [column for column in upper.columns if any(upper[column] > 0.95)]
    return to_drop_numerical


if __name__ == '__main__':
    # Load data
    df1 = pd.read_csv('data/Training_part1.csv', sep=";")
    df2 = pd.read_csv('data/Training_part2.csv', sep=";")

    # Preprocessing
    df1.drop_duplicates(inplace=True)
    df2.drop_duplicates(inplace=True)
    df3 = df1.merge(df2, on=["id"], how='outer')

    int_feature = ['BIB', 'WET']
    float_feature = ['FAN', 'LUK', 'NUS', 'SIS', 'UIN']
    cat_features = ['COD', 'ERG', 'GJAH', 'MYR', 'PKD', 'RAS', 'TOK', 'VOL', 'KAT', 'XIN']

    df3 = df3.replace(r'^\s*$', np.NaN, regex=True)  # replace all blank slots with NaN
    df3.drop(['RAS', 'id'], axis=1, inplace=True)
    cat_features.remove('RAS')

    for column in int_feature + float_feature:
        df3[column] = df3[column].fillna(value=df3[column].mean())
    for column in cat_features:
        df3[column] = df3[column].fillna(df3[column].mode()[0])

    # Convert categorical features to dummy variables
    df3 = pd.get_dummies(df3, columns=cat_features, drop_first=True)

    # Split features and target
    X = df3.loc[:, df3.columns != 'Class'].values
    y = df3.loc[:, 'Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Feature selection
    X_train_df = pd.DataFrame(X_train, columns=df3.columns.drop('Class'))
    to_drop = feature_todrop(X_train_df)
    X_train_df.drop(to_drop, axis=1, inplace=True)

    X_test_df = pd.DataFrame(X_test, columns=df3.columns.drop('Class'))
    X_test_df.drop(to_drop, axis=1, inplace=True)

    X_train = X_train_df.values
    X_test = X_test_df.values

    # Model training and evaluation
    lr_pred = log_regression(X_train, y_train, X_test, y_test)
    print()
    rf_pred = random_forest(X_train, y_train, X_test, y_test)
    print()
    svm_pred = linear_svm(X_train, y_train, X_test, y_test)

    # F1-score plotting
    df_res = pd.DataFrame({
        'logistic regression': [f1_score(y_test, lr_pred, pos_label="n"), f1_score(y_test, lr_pred, pos_label="y")],
        'random forest': [f1_score(y_test, rf_pred, pos_label="n"), f1_score(y_test, rf_pred, pos_label="y")],
        'linear svm': [f1_score(y_test, svm_pred, pos_label="n"), f1_score(y_test, svm_pred, pos_label="y")]
    }, index=['n', 'y'])

    ax = df_res.plot.bar()
    ax.set_xlabel("class")
    ax.set_ylabel("f1-score")
    plt.show()
