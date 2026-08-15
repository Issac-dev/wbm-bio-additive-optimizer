import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_conc", "B_conc", "G_conc", "O_conc", "CS_conc", "CH_conc",
    "Be_conc", "Ba_conc", "PAC_conc", "Lime_conc"
]

def diagnose(target_column, n_splits=5, top_n=10):
    model_data = df[feature_columns + [target_column] + ["Author", "Entry_ID", "Variation"]].dropna(subset=[target_column])
    X = model_data[feature_columns]
    y = model_data[target_column]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=kf)

    result = model_data[["Author", "Entry_ID", "Variation"]].copy()
    result["Actual"] = y.values
    result["Predicted"] = predictions
    result["Abs_Error"] = (result["Predicted"] - result["Actual"]).abs()
    result = result.sort_values("Abs_Error", ascending=False)

    print(f"\n=== Worst {top_n} predictions for {target_column} ===")
    print(result.head(top_n).to_string(index=False))

diagnose("Density_ppg")
diagnose("YP_lb100ft2")