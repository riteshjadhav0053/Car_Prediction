# 🚗 Car Price Prediction

An end-to-end Machine Learning project that predicts the resale price of a used car based on its specifications, usage, ownership details, and other vehicle characteristics.

The project uses **XGBoost Regression** with preprocessing and hyperparameter tuning, and provides an interactive **Streamlit web application** for making predictions.

---

## 📌 Project Overview

Used-car prices depend on many factors such as:

- Brand and model
- Manufacturing year
- Kilometers driven
- Fuel type
- Transmission
- Engine specifications
- Power and torque
- Vehicle dimensions
- Location
- Ownership history
- Seller type
- Drivetrain

This project builds a regression model to estimate the market price of a used car from these features.

---

## 🎯 Objectives

- Perform exploratory data analysis
- Clean and preprocess the dataset
- Engineer useful features
- Compare multiple regression algorithms
- Tune the best-performing model
- Evaluate model performance using regression metrics
- Save the trained model and preprocessing pipeline
- Build an interactive Streamlit application
- Deploy a complete end-to-end ML prediction workflow

---

## 🗂️ Project Structure

```text
Car-Price-Prediction/
│
├── data/
│   └── dataset.csv
│
├── models/
│   ├── best_xgb_model.pkl
│   ├── preprocessor.pkl
│   ├── make_model_map.pkl
│   └── car_specs.pkl
│
├── notebook/
│   └── car_price_prediction.ipynb
│
├── src/
│   ├── components/
│   └── pipeline/
│       └── predict_pipeline.py
│
├── artifacts/
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md