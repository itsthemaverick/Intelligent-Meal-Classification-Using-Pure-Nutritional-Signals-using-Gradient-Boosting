import joblib
import pandas as pd

def load_model():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

def predict_category(model, scaler, input_data, feature_names):
    df = pd.DataFrame([input_data], columns=feature_names)
    df_scaled = scaler.transform(df)
    return model.predict(df_scaled)[0]
