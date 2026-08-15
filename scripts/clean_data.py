import pandas as pd
# ---- 1. Load the raw file ----
raw = pd.read_excel(
    "data/Project_Dataset.xlsx",
    sheet_name="Dataset",
    header=None,
    skiprows=3
)
# ---- 2. Give every column a clean, simple name ----
column_names = [
    "Entry_ID", "Author", "Source", "Variation",
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g",
    "Biomaterial", "Temperature_F",
    "PV_cP", "AV_cP", "YP_lb100ft2",
    "Gel_10s", "Gel_10min", "API_Filtration_mL", "Fluid_Loss_mL",
    "Mud_Cake_mm", "Density_ppg", "pH", "Viscosity_100rpm",
    "SG", "Notes"
]
raw.columns = column_names
# ---- 3. Convert grams to concentration (g per 100 mL water) ----
# This makes recipes from different papers comparable, since batch sizes
# (total water volume) vary a lot across papers (300mL vs 1000mL etc.)
conc_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]
for col in conc_columns:
    new_col = col.replace("_g", "_conc")
    raw[new_col] = (raw[col] / raw["Water_mL"]) * 100  # g per 100 mL water
# ---- 4. Save the cleaned file ----
raw.to_csv("data/cleaned_dataset.csv", index=False)

print("Done! Cleaned data saved to data/cleaned_dataset.csv")
print(f"Total rows: {len(raw)}")
print(raw.head(10))