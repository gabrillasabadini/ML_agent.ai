import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


def encode_categorical_columns(df):
    """
    Automatically detect and encode categorical columns.
    - Converts all non-numeric columns to numeric using LabelEncoder.
    - Returns the encoded DataFrame and the encoders used.
    """
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or str(df_encoded[col].dtype).startswith('category'):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    return df_encoded, label_encoders


def train_selected_model(model_name, X, y, target_type):
    nn_summary_lines = []

    # Encode all categorical (non-numeric) columns in X
    X_encoded, encoders = encode_categorical_columns(X)

    # Encode categorical target variable (for classification)
    if "Categorical" in target_type and (y.dtype == 'object' or str(y.dtype).startswith('category')):
        y = y.astype('category').cat.codes

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # ------------------- Traditional ML Models -------------------
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, preds), 3),
            "Precision": round(precision_score(y_test, preds, average='weighted'), 3),
            "Recall": round(recall_score(y_test, preds, average='weighted'), 3)
        }

    elif model_name == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "MAE": round(mean_absolute_error(y_test, preds), 3),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 3),
            "R2": round(r2_score(y_test, preds), 3)
        }

    # ------------------- Neural Network Models -------------------
    elif model_name == "NeuralNetwork_Classifier":
        num_classes = len(np.unique(y))
        y_train_enc = to_categorical(y_train, num_classes)
        y_test_enc = to_categorical(y_test, num_classes)

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train_enc, epochs=10, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test, y_test_enc, verbose=0)

        metrics = {"Accuracy": round(acc, 3), "Loss": round(loss, 3)}
        model.summary(print_fn=lambda x: nn_summary_lines.append(x))

    elif model_name == "NeuralNetwork_Regressor":
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)

        metrics = {"MAE": round(mae, 3), "MSE": round(loss, 3)}

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ------------------- Prepare Output -------------------
    preds_display = preds[:5] if 'preds' in locals() else [np.nan]*5
    sample_predictions = pd.DataFrame({
        "Actual": y_test[:5].values,
        "Predicted": preds_display
    })

    nn_summary = "\n".join(nn_summary_lines) if nn_summary_lines else None

    return metrics, sample_predictions, nn_summary
