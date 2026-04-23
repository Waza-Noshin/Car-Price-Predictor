import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

os.makedirs('models', exist_ok=True)

print("=" * 50)
print("CAR PRICE PREDICTION - TRAINING")
print("=" * 50)

# Create sample car dataset directly
print("\n📥 Creating car dataset...")
np.random.seed(42)
n_samples = 1000

# Create realistic car features
df = pd.DataFrame({
    'HP': np.random.randint(50, 400, n_samples),
    'MPG': np.random.uniform(10, 50, n_samples),
    'Volume': np.random.uniform(100, 500, n_samples),
    'Weight': np.random.randint(1500, 6000, n_samples),
    'Cylinders': np.random.choice([3, 4, 5, 6, 8], n_samples)
})

# Calculate price based on features (realistic relationship)
# Higher HP = higher price
# Higher MPG = slightly higher price
# Lower Weight = higher price (sports cars)
# More Cylinders = higher price
df['Price'] = (
    df['HP'] * 80 +
    df['MPG'] * 50 +
    (5000 - df['Weight'] / 10) +
    df['Cylinders'] * 1500 +
    np.random.randint(-5000, 5000, n_samples)
)

# Ensure price is positive and reasonable
df['Price'] = df['Price'].clip(3000, 120000).astype(int)

print(f"✅ Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")

# Display first few rows
print("\n📊 First 5 rows of dataset:")
print(df.head())

print(f"\n📋 Columns: {list(df.columns)}")

# Features and target
feature_columns = ['HP', 'MPG', 'Volume', 'Weight', 'Cylinders']
target_column = 'Price'

X = df[feature_columns]
y = df[target_column]

print(f"\n✅ Using features: {feature_columns}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
print("\n🔄 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"   R² Score: {r2:.4f} ({r2 * 100:.2f}%)")
print(f"   Mean Absolute Error: ${mae:,.2f}")
print(f"   Root Mean Squared Error: ${rmse:,.2f}")

# Save model and scaler
print("\n💾 Saving model...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model saved to 'models/' folder")
print("=" * 50)

# Save dataset for reference
df.to_csv('car_dataset.csv', index=False)
print("📁 Dataset saved as 'car_dataset.csv'")