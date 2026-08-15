import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---- 1. Load cleaned data ----
df = pd.read_csv("data/cleaned_dataset.csv")

# ---- 2. Define our inputs (features) and output (target) ----
# We only use the 20 recipe/constituent columns as inputs — not Biomaterial text, not Notes.
feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "PAC_g", "Lime_g"
]
target_column = "Density_ppg"

# ---- 3. Keep only rows where the target actually has a value ----
model_data = df[feature_columns + [target_column]].dropna(subset=[target_column])
print(f"Rows available for {target_column}: {len(model_data)}")

X = model_data[feature_columns]   # inputs
y = model_data[target_column]     # what we're trying to predict

# ---- 4. Split into training data and testing data ----
# We train the model on 80% of rows, then test it on the 20% it never saw,
# to check if it actually learned something real (not just memorized).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

# ---- 5. Train a Random Forest model ----
# A Random Forest builds many decision trees and averages their answers —
# it's robust to small/noisy data and doesn't need feature scaling.
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---- 6. Test how good it is ----
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n=== Model Performance on {target_column} ===")
print(f"Mean Absolute Error: {mae:.3f}  (average size of prediction mistake)")
print(f"R² Score: {r2:.3f}  (1.0 = perfect, 0 = no better than guessing average)")

# ---- 7. Which inputs matter most? ----
importance = pd.Series(model.feature_importances_, index=feature_columns)
importance = importance.sort_values(ascending=False)
print("\n=== Feature Importance (what drives Density the most) ===")
print(importance.head(10))