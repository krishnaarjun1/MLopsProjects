from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train, save_path="models/model.pkl"):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, save_path)
    return clf
