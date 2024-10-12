from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


def log_regression(X_train, y_train, X_test, y_test):
    """ Logistic Regression Model """
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr', max_iter=4000)
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)

    print("\n Accuracy (logistic regression):", accuracy_score(y_test, lr_predict))
    print("=== Classification Report (logistic regression) ===")
    print(classification_report(y_test, lr_predict))
    print("=== Confusion Matrix (logistic regression) ===")
    print(confusion_matrix(y_test, lr_predict))

    return lr_predict


def log_regression_cv(X, y, cv):
    """ Logistic Regression with Cross-Validation """

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', max_iter=4000)

    lr_predict = cross_val_predict(lr, X_scaled, y, cv=cv)
    return lr_predict
