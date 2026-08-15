import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_conc", "B_conc", "G_conc", "O_conc", "CS_conc", "CH_conc",
    "Be_conc", "Ba_conc", "PAC_conc", "Lime_conc"
]

def train_and_evaluate(target_column, n_splits=5):
    model_data = df[feature_columns + [target_column]].dropna(subset=[target_column])
    print(f"\n{'='*50}")
    print(f"Target: {target_column}  |  Rows available: {len(model_data)}")

    X = model_data[feature_columns]
    y = model_data[target_column]

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    # k-fold cross-validation: every row gets predicted exactly once,
    # using a model trained on the OTHER folds. Much more stable than
    # one random 80/20 split when you only have ~40-60 rows.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, X, y, cv=kf)

    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"MAE: {mae:.3f}  |  R² (cross-validated): {r2:.3f}")

    # Fit on full data to get final feature importance
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=feature_columns)
    importance = importance.sort_values(ascending=False)
    print("Top features:")
    print(importance.head(6))

    return model, mae, r2, predictions, y

targets_to_model = [
    "Density_ppg", "pH", "PV_cP", "YP_lb100ft2",
    "Gel_10s", "Viscosity_100rpm"
]

results = {}
for target in targets_to_model:
    model, mae, r2, preds, actual = train_and_evaluate(target)
    results[target] = {"model": model, "mae": mae, "r2": r2}

print(f"\n{'='*50}")
print("=== Summary across all targets (cross-validated) ===")
for target, res in results.items():
    print(f"{target:20s}  MAE={res['mae']:.3f}   R²={res['r2']:.3f}")