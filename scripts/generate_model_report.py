import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]

targets = [
    "Density_ppg", "PV_cP", "YP_lb100ft2",
    "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"
]

report_rows = []
for target in targets:
    model_data = df[feature_columns + [target]].dropna(subset=[target])
    X = model_data[feature_columns]
    y = model_data[target]
    n_rows = len(model_data)

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    # 5-fold cross-validation: split data 5 different ways, train/test each way,
    # average the result — more reliable than a single train/test split on small data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

# Confidence now reflects BOTH sample size AND actual model performance (R²)
    mean_r2 = r2_scores.mean()
    if mean_r2 < 0.3:
        confidence = "Low (poor fit)"
    elif mean_r2 < 0.5:
        confidence = "Moderate"
    elif n_rows >= 40:
        confidence = "High"
    else:
        confidence = "Moderate"

    report_rows.append({
        "Target": target,
        "Rows used": n_rows,
        "R² (mean)": round(r2_scores.mean(), 3),
        "R² (std dev)": round(r2_scores.std(), 3),
        "MAE (mean)": round(mae_scores.mean(), 3),
        "Confidence": confidence
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv("data/model_performance_report.csv", index=False)
print(report_df.to_string(index=False))
print("\nSaved to data/model_performance_report.csv")