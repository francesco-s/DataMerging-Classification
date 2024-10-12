from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict


def random_forest(X_train, y_train, X_test, y_test):
    """ Random Forest Classifier """
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)

    print("\n Accuracy (random forest):", accuracy_score(y_test, rfc_predict))
    print("=== Classification Report (random forest) ===")
    print(classification_report(y_test, rfc_predict))
    print("=== Confusion Matrix (random forest) ===")
    print(confusion_matrix(y_test, rfc_predict))

    return rfc_predict


def random_forest_cv(X, y, cv):
    """ Random Forest with Cross-Validation """
    rfc = RandomForestClassifier()
    rfc_predict = cross_val_predict(rfc, X, y, cv=cv)
    return rfc_predict
