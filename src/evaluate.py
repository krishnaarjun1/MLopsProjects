import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Ensure folder exists
    os.makedirs("evaluation_results", exist_ok=True)

    # Save to JSON
    with open("evaluation_results/metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation metrics saved to evaluation_results/metrics.json")

    return results
