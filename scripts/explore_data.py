import pandas as pd

# Load the cleaned data we just made
df = pd.read_csv("data/cleaned_dataset.csv")

# ---- 1. How many rows have a real (non-blank) value for each target? ----
target_columns = [
    "PV_cP", "AV_cP", "YP_lb100ft2", "Gel_10s", "Gel_10min",
    "API_Filtration_mL", "Fluid_Loss_mL", "Mud_Cake_mm",
    "Density_ppg", "pH", "Viscosity_100rpm", "SG"
]

print("=== How many usable rows per target property ===")
for col in target_columns:
    count = df[col].notna().sum()   # notna() = "is this NOT blank?"
    print(f"{col:25s}: {count} rows")

# ---- 2. How many rows per biomaterial ----
print("\n=== Rows per Biomaterial ===")
print(df["Biomaterial"].value_counts())

# ---- 3. Basic stats on the recipe/input columns ----
input_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]
print("\n=== Input column summary (min / max / how many non-zero) ===")
for col in input_columns:
    nonzero = (df[col] > 0).sum()
    print(f"{col:10s}: min={df[col].min():<8} max={df[col].max():<8} non-zero rows={nonzero}")