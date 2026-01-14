import os
import joblib
from src.data_loader import load_data
from src.features import prepare_features
from src.model import get_model
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import (
    plot_class_distribution,
    plot_feature_correlation,
    plot_feature_importance
)

os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

df = load_data("data/processed/clean_values.csv")

X_train, X_test, y_train, y_test = prepare_features(df)

model = get_model()
model = train_model(model, X_train, y_train)

accuracy, report = evaluate_model(model, X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")
print(report)

plot_class_distribution(df, "visualizations")
plot_feature_correlation(df, "visualizations")
plot_feature_importance(
    model,
    X_test,
    y_test,
    df.drop(columns=["Category"]).columns,
    "visualizations"
)

joblib.dump(model, "models/best_model.pkl")
