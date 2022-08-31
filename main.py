import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns


def log_regression(X_train, y_train, X_test, y_test):
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # build a logistic regression model (param max_iter fine tuned)
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr', max_iter=4000)

    lr.fit(X_train, y_train)

    lr_predict = lr.predict(X_test)

    print("Accuracy (logistic regression):", accuracy_score(y_test, lr_predict))

    # Calculate precision, recall and F1 metrics (useful for unbalanced classification problem like this)
    print("=== Classification Report (logistic regression) ===")
    print(classification_report(y_test, lr_predict))

    # confusion matrix
    print("=== Confusion Matrix (logistic regression) ===")
    print(confusion_matrix(y_test, lr_predict))

    return lr_predict


def random_forest(X_train, y_train, X_test, y_test):
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # predictions
    rfc_predict = rfc.predict(X_test)

    print("Accuracy (random forest) :", accuracy_score(y_test, rfc_predict))

    # Calculate precision, recall and F1 metrics (useful for unbalanced classification problem like this)
    print("=== Classification Report (random forest) ===")
    print(classification_report(y_test, rfc_predict))

    # confusion matrix
    print("=== Confusion Matrix (random forest) ===")
    print(confusion_matrix(y_test, rfc_predict))

    return rfc_predict



def linear_svm(X_train, y_train, X_test, y_test):
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

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
    # Using Pearson Correlation (only numeric feature)

    # Pair-wise correlation
    cor = df.corr()

    corr_matrix = cor.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    # Find features with correlation greater than 0.95
    to_drop_numerical = [column for column in upper.columns if any(upper[column] > 0.95)]

    # In the previous version I was going to calculate the correlation between
    # categorical features using a chi square test and contingency matrix.
    # Later I noticed that it was convenient to convert categorical features into a larger set of numeric features.
    # I leave the code anyway
    '''
    cat_features = ['COD', 'ERG', 'GJAH', 'MYR', 'PKD', 'RAS', 'TOK', 'VOL', 'KAT', 'XIN']
    p_values = dict()

    for f in cat_features:
        for f2 in cat_features:
            if f != f2 and (f, f2) not in p_values.keys() and (f2, f) not in p_values.keys():
                contigency = pd.crosstab(df[f], df[f2])

                print(contigency)
                plt.figure(figsize=(12, 8))
                sns.heatmap(contigency, annot=True, cmap=plt.cm.Blues)
                plt.show()

                # Chi-square test of independence.
                c, p, dof, expected = chi2_contingency(contigency)
                p_values[(f, f2)] = p

    p_values_sorted = dict(sorted(p_values.items(), key=lambda item: item[1]))

    alpha = 0.95
    to_drop_categorical = []

    for key, value in p_values_sorted.items():
        if value > alpha:
            to_drop_categorical.append(
                key[1])  # all the feature are independent each other (according to the chi square test
            # with alpha=0.95)

    # print("Categorical feature to drop: ", to_drop_categorical)'''

    return to_drop_numerical


if __name__ == '__main__':
    df1 = pd.read_csv('./Training_part1.csv', sep=";")
    df2 = pd.read_csv('./Training_part2.csv', sep=";")

    # print(df1.sort_index(axis=0))
    print("Before preprocessing ----->")
    print("Training_part1 shape: {}".format(df1.shape))
    print("Training_part2 shape: {}".format(df2.shape))

    print("After deleting duplicates ----->")
    df1.drop_duplicates(inplace=True)
    df2.drop_duplicates(inplace=True)

    pd.set_option('display.max_columns', None)

    print("Training_part1 shape: {}".format(df1.shape))
    print("Training_part2 shape: {}".format(df2.shape))

    df3 = df1.merge(df2, on=["id"], how='outer')
    df3.to_csv("final.csv", index=False)

    int_feature = ['BIB', 'WET']
    float_feature = ['FAN', 'LUK', 'NUS', 'SIS', 'UIN']
    cat_features = ['COD', 'ERG', 'GJAH', 'MYR', 'PKD', 'RAS', 'TOK', 'VOL', 'KAT', 'XIN']

    # working with null values
    df3 = df3.replace(r'^\s*$', np.NaN, regex=True)  # replace all blank slot with null
    print('Null count for each feature ----->')
    print(df3.isnull().sum())  # RAS is the feature with more nulls (2145/3700 null values -> 58% null values).
    print("Null count:", df3.isnull().sum().sum())  # 2683 null values (2145 RAS, FAN 100, NUS 100...)
    # removing RAS and id features
    df3.drop(['RAS', 'id'], axis=1, inplace=True)
    cat_features.remove('RAS')

    # fill with mean
    for column in int_feature + float_feature:
        df3[column] = df3[column].fillna(value=df3[column].mean())

    # fill with majority value
    for column in cat_features:
        df3[column] = df3[column].fillna(df3[column].mode()[0])

    print("Null count after filling:", df3.isnull().sum().sum())  # no NaN

    # covert categorical feature into a numeric representation with more dimensions
    cod = pd.get_dummies(df3['COD'], drop_first=True)
    erg = pd.get_dummies(df3['ERG'], drop_first=True)
    gjah = pd.get_dummies(df3['GJAH'], drop_first=True)
    myr = pd.get_dummies(df3['MYR'], drop_first=True)
    pkd = pd.get_dummies(df3['PKD'], drop_first=True)
    tok = pd.get_dummies(df3['TOK'], drop_first=True)
    vol = pd.get_dummies(df3['VOL'], drop_first=True)
    kat = pd.get_dummies(df3['KAT'], drop_first=True)
    xin = pd.get_dummies(df3['XIN'], drop_first=True)

    # dropping original categorical features
    df3.drop(['COD', 'ERG', 'GJAH', 'MYR', 'PKD', 'TOK', 'VOL', 'KAT', 'XIN'], axis=1, inplace=True)
    # add new features
    df3 = pd.concat([df3, cod, erg, gjah, myr, pkd, tok, vol, kat, xin], axis=1)

    # Separating out the features
    X = df3.loc[:, df3.columns != 'Class'].values
    # Separating out the target
    y = df3.loc[:, ['Class']].values
    # train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # summarize
    print('Data shape:', df3.shape)
    print('X_Train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
    print('X_test shape: {}, y_test shape: {}'.format(X_test.shape, y_test.shape))

    # create a dataframe with X_train
    X_train_df = pd.DataFrame(X_train)
    # set columns name
    all_feature = df3.columns.tolist()
    all_feature.remove('Class')
    X_train_df.set_axis(all_feature, axis=1, inplace=True)

    # casting objects to float32 and int32
    X_train_df[int_feature] = X_train_df[int_feature].astype('int32')
    X_train_df[float_feature] = X_train_df[float_feature].astype('float32')
    # print(X_train_df.info())

    # feature selection on train set
    to_drop = feature_todrop(X_train_df)
    print("Features to drop:", to_drop)
    X_train_df.drop(to_drop, axis=1, inplace=True)
    print("X_train new shape:", X_train_df.shape)

    # processing X_test
    X_test_df = pd.DataFrame(X_test)
    X_test_df.set_axis(all_feature, axis=1, inplace=True)
    X_test_df[int_feature] = X_test_df[int_feature].astype('int32')
    X_test_df[float_feature] = X_test_df[float_feature].astype('float32')
    X_test_df.drop(to_drop, axis=1, inplace=True)
    print("X_test new shape:", X_test_df.shape)

    X_train = X_train_df.loc[:, :].values
    X_test = X_test_df.loc[:, :].values

    # In this study I will use logistic regression, random forest and a liner support vector machine.
    # However, other approaches such as clustering or neural networks could be used.
    lr_pred = log_regression(X_train, y_train, X_test, y_test)
    print()
    rf_pred = random_forest(X_train, y_train, X_test, y_test)
    print()
    svm_pred = linear_svm(X_train, y_train, X_test, y_test)  # long training time (about 10 minute on my machine)

    df_res = pd.DataFrame({'logistic regression': [f1_score(y_test, lr_pred, pos_label="n"),
                                                   f1_score(y_test, lr_pred, pos_label="y")],
                           'random forest': [f1_score(y_test, rf_pred, pos_label="n"),
                                             f1_score(y_test, rf_pred, pos_label="y")],
                           'linear svm': [f1_score(y_test, svm_pred, pos_label="n"),
                                          f1_score(y_test, svm_pred, pos_label="y")]}, index=['n', 'y'])
    ax = df_res.plot.bar()
    ax.set_xlabel("class")
    ax.set_ylabel("f1-score")
    plt.show()

    # OUTPUT WILL BE LIKE THE FOLLOWING
    '''
        Accuracy (logistic regression): 0.918918918918919
        === Classification Report (logistic regression) ===
                      precision    recall  f1-score   support
        
                   n       0.37      0.13      0.19        55
                   y       0.93      0.98      0.96       685
        
            accuracy                           0.92       740
           macro avg       0.65      0.55      0.57       740
        weighted avg       0.89      0.92      0.90       740
        
        === Confusion Matrix (logistic regression) ===
        [[  7  48]
         [ 12 673]]

 
        Accuracy (random forest) : 0.9837837837837838
        === Classification Report (random forest) ===
                      precision    recall  f1-score   support
        
                   n       1.00      0.78      0.88        55
                   y       0.98      1.00      0.99       685
        
            accuracy                           0.98       740
           macro avg       0.99      0.89      0.93       740
        weighted avg       0.98      0.98      0.98       740
        
        === Confusion Matrix (random forest) ===
        [[ 43  12]
         [  0 685]]
         
         
        Accuracy (linear svm): 0.9256756756756757
        === Classification Report (linear svm) ===
                      precision    recall  f1-score   support
        
                   n       0.50      0.09      0.15        55
                   y       0.93      0.99      0.96       685
        
            accuracy                           0.93       740
           macro avg       0.72      0.54      0.56       740
        weighted avg       0.90      0.93      0.90       740
        
        === Confusion Matrix (linear svm) ===
        [[  5  50]
         [  5 680]]    
    '''
    # Considering f1-score in both class (n and y), random forest is easily the best model
