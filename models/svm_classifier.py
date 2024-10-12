from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict


def svm_classifier(X_train, y_train, X_test, y_test):
    """ Linear Support Vector Machine """
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    svm_predict = clf.predict(X_test)

    print("\nAccuracy (linear svm):", accuracy_score(y_test, svm_predict))
    print("=== Classification Report (linear svm) ===")
    print(classification_report(y_test, svm_predict))
    print("=== Confusion Matrix (linear svm) ===")
    print(confusion_matrix(y_test, svm_predict))

    return svm_predict


def svm_classifier_cv(X, y, cv):
    """ Linear SVM with Cross-Validation """
    clf = svm.SVC(kernel='linear')
    svm_predict = cross_val_predict(clf, X, y, cv=cv)
    return svm_predict
