from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model(model, X_test, y_test, output_path="outputs/evaluation.txt"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    with open(output_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}")
    
    print(f"Accuracy: {acc:.4f}")
    print(report)
