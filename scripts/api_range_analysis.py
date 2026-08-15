import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/cleaned_dataset.csv")

feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]

# API RP 13B-1 typical operating ranges (edit these to match YOUR program's exact spec sheet)
INDUSTRY_RANGES = {
    "Density_ppg": (8.65, 19.0),
    "PV_cP": (1, 65),          # broad allowable field range
    "YP_lb100ft2": (3, 60),
    "Fluid_Loss_mL": (0, 15),  # API max recommended fluid loss
}

targets = ["Density_ppg", "PV_cP", "YP_lb100ft2", "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"]

report_rows = []
for target in targets:
    model_data = df[feature_columns + [target]].dropna(subset=[target])

    if target in INDUSTRY_RANGES:
        low, high = INDUSTRY_RANGES[target]
        in_range = model_data[(model_data[target] >= low) & (model_data[target] <= high)]
        out_range = model_data[(model_data[target] < low) | (model_data[target] > high)]
        method = "API RP 13B-1 range"
    else:
        # No formal API spec for this property — use statistical range instead (mean ± 2 std)
        # This is a transparent, defensible substitute where no industry number applies.
        mean, std = model_data[target].mean(), model_data[target].std()
        low, high = mean - 2*std, mean + 2*std
        in_range = model_data[(model_data[target] >= low) & (model_data[target] <= high)]
        out_range = model_data[(model_data[target] < low) | (model_data[target] > high)]
        method = "Statistical range (mean ± 2 std)"

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)

    if len(in_range) >= 10:
        r2_in = cross_val_score(model, in_range[feature_columns], in_range[target], cv=kf, scoring="r2").mean()
    else:
        r2_in = None

    report_rows.append({
        "Target": target,
        "Method": method,
        "Range (low-high)": f"{low:.2f} - {high:.2f}",
        "In-range rows": len(in_range),
        "Out-of-range rows": len(out_range),
        "In-range R²": round(r2_in, 3) if r2_in is not None else "insufficient data",
        "Full-dataset R² (reference)": None  # fill from your model_performance_report.csv
    })

report_df = pd.DataFrame(report_rows)
report_df.to_csv("data/api_range_stratification_report.csv", index=False)
print(report_df.to_string(index=False))
print("\nSaved to data/api_range_stratification_report.csv")