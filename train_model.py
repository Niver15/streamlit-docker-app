#MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Prepare X and y
X = data[npi_cols].values
y = data['target'].values

print(f"X shape: {X.shape}, y shape: {y.shape}")  # Should match in rows

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
# Save model to disk in folder 'model'
import os
import joblib
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/my_model.pkl')
print("Model saved!")