import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "PAC_g", "Lime_g"
]
target_column = "PV_cP"

model_data = df[feature_columns + [target_column] + ["Author", "Entry_ID", "Variation"]].dropna(subset=[target_column])

print("=== PV_cP distribution ===")
print(model_data[target_column].describe())

X = model_data[feature_columns]
y = model_data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Show exactly which rows were tested and how badly each was missed
test_info = model_data.loc[y_test.index, ["Author", "Entry_ID", "Variation"]].copy()
test_info["Actual_PV"] = y_test.values
test_info["Predicted_PV"] = predictions
test_info["Error"] = test_info["Predicted_PV"] - test_info["Actual_PV"]

print("\n=== Test set: actual vs predicted ===")
print(test_info.to_string(index=False))