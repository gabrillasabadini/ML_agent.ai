import pandas as pd
from model_agent import train_selected_model

def analyze_and_decide(df, target_col, target_type):
    """
    Decide which model to use and return performance report.
    """
    # Basic checks
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Features & Target split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Decision Logic
    num_features = len(X.select_dtypes(include=['number']).columns)
    num_samples = len(df)

    decision = {}
    report = {}

    if "Categorical" in target_type:
        if num_features > 20 or num_samples > 5000:
            model_name = "NeuralNetwork_Classifier"
            reason = "Dataset is large or high-dimensional → Neural Network chosen."
        else:
            model_name = "RandomForestClassifier"
            reason = "Moderate dataset size and categorical target → Random Forest chosen."
    else:  # Regression
        if num_samples > 5000:
            model_name = "NeuralNetwork_Regressor"
            reason = "Large numeric dataset → Neural Network chosen."
        else:
            model_name = "LinearRegression"
            reason = "Small to medium numeric dataset → Linear Regression chosen."

    decision["model_name"] = model_name
    decision["reason"] = reason

    # Train and evaluate
    metrics, sample_predictions, nn_summary = train_selected_model(model_name, X, y, target_type)

    report["metrics"] = metrics
    report["sample_predictions"] = sample_predictions
    report["nn_summary"] = nn_summary

    return decision, report
