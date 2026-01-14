# NutriClass --- Meal Category Classification Using Nutritional Data

## Overview

NutriClass is a production-grade machine learning project that
classifies meals into meaningful food categories using **only
nutritional values**.\
Unlike menu-name or text-based classifiers, this system relies purely on
quantitative nutrition signals such as protein, fats, carbohydrates,
sugars, sodium, and calories.

The project follows **real ML engineering practices**: modular codebase,
no notebooks, reproducible pipeline, model persistence, visualization,
and prediction support.

------------------------------------------------------------------------

## Problem Statement

Given the nutritional composition of a meal, predict its category: -
Breakfast - Beverages - Tea & Coffee - Smoothies & Shakes - Desserts -
Snacks & Sides - Salads - Chicken & Fish - Beef & Pork

This mirrors real-world use cases such as: - Diet recommendation
systems - Food logging applications - Health analytics platforms -
Automated menu tagging

------------------------------------------------------------------------

## Key Features

-   Nutrition-only classification (no text leakage)
-   High-accuracy boosted tree model
-   Stratified train-test split
-   Persistent feature scaling
-   Model-agnostic interpretability (Permutation Importance)
-   Clean, modular, production-ready structure
-   Batch-ready prediction pipeline

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas / NumPy
-   scikit-learn
-   Matplotlib / Seaborn
-   Joblib

------------------------------------------------------------------------

## Project Structure

    meal-classification-ml/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── models/
    │   ├── best_model.pkl
    │   └── scaler.pkl
    │
    ├── src/
    │   ├── data_loader.py
    │   ├── features.py
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── visualize.py
    │   └── predict.py
    │
    ├── visualizations/
    ├── main.py
    ├── config.py
    └── requirements.txt

------------------------------------------------------------------------

## Model

The system uses **HistGradientBoostingClassifier**, chosen for: - Strong
performance on tabular data - Ability to model non-linear nutrition
patterns - Excellent bias--variance tradeoff

accuracy: **80%**, depending on class balance.

------------------------------------------------------------------------

## Visual Diagnostics

Generated automatically: - Class distribution - Feature correlation
heatmap - Permutation-based feature importance

These plots help validate data quality and model reasoning.

------------------------------------------------------------------------

## How to Run

    pip install -r requirements.txt
    python main.py

------------------------------------------------------------------------

## Prediction Example

Provide nutritional values in the trained feature order:

    [Calories, Total Fat, Saturated Fat, Trans Fat, Cholesterol,
     Sodium, Carbohydrates, Dietary Fiber, Sugars, Protein]

The system outputs the predicted meal category.

------------------------------------------------------------------------

## Why This Project Stands Out

-   No notebooks --- real engineering workflow
-   Nutrition-driven logic (domain-aware ML)
-   Interpretability-first design
-   Easily extensible to APIs or production deployment

------------------------------------------------------------------------

## Future Enhancements

-   XGBoost / LightGBM integration
-   SHAP-based explanations
-   REST API deployment
-   Real-time nutrition input interface

------------------------------------------------------------------------

## Author : itsthemaverick

Built with an engineering-first mindset for high-accuracy, interpretable
machine learning.
