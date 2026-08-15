import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]

# pH removed — it's driven by NaOH/Lime/PAC, not bio-additives, so it's
# outside the scope of what this project is actually testing
targets = [
    "Density_ppg", "PV_cP", "YP_lb100ft2",
    "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"
]

os.makedirs("models", exist_ok=True)

for target in targets:
    model_data = df[feature_columns + [target]].dropna(subset=[target])
    X = model_data[feature_columns]
    y = model_data[target]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    joblib.dump(model, f"models/{target}_model.pkl")
    print(f"Saved model for {target} ({len(model_data)} rows used)")

print("\nAll models saved to the 'models/' folder.")