# Car Price Predictor
## Project Overview
A web application that predicts car prices based on technical specifications using Machine Learning.
## Live Demo
Run locally: `python app.py` then open `http://localhost:5000`
## Tech Stack
- Python, Flask (Backend)
- HTML, CSS (Frontend)
- Scikit-learn, Random Forest Regressor (ML)
## Features
- Clean web interface
- 5 input features: HP, MPG, Volume, Weight, Cylinders
- Instant price prediction
- 88.59% R² Score accuracy
## Model Performance
| Metric | Value |
|--------|-------|
| R² Score | 88.59% |
| Mean Absolute Error (MAE) | $2,592.11 |
| Root Mean Squared Error (RMSE) | $3,025.48 |
| Algorithm | Random Forest Regressor |
| Training Samples | 800 |
| Testing Samples | 200 |
## How to Run
```bash
pip install -r requirements.txt
python train_model.py
python app.py
