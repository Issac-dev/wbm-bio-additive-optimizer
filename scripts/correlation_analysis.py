import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_dataset.csv")

biomaterial_cols = ["E_g", "B_g", "G_g", "O_g", "CS_g"]
biomaterial_labels = {
    "E_g": "Eggshell", "B_g": "Banana Peel", "G_g": "Groundnut Shell",
    "O_g": "Okra", "CS_g": "Cornstarch"
}

# ---- 1. Correlation matrix among the 5 biomaterials ----
corr_matrix = df[biomaterial_cols].corr(method="pearson")
corr_matrix = corr_matrix.rename(index=biomaterial_labels, columns=biomaterial_labels)

print("=== Pearson Correlation Matrix: Biomaterials ===")
print(corr_matrix.round(3))
corr_matrix.to_csv("data/biomaterial_correlation_matrix.csv")

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", square=True)
plt.title("Correlation Between Biomaterial Usage Across Dataset")
plt.tight_layout()
plt.savefig("plots/biomaterial_correlation_heatmap.png", dpi=150)
print("\nSaved: plots/biomaterial_correlation_heatmap.png")

# ---- 2. Correlation between biomaterials AND target properties ----
targets = ["Density_ppg", "PV_cP", "YP_lb100ft2", "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"]
combined = df[biomaterial_cols + targets]
combined = combined.rename(columns=biomaterial_labels)

full_corr = combined.corr(method="pearson")
biomaterial_vs_targets = full_corr.loc[list(biomaterial_labels.values()), targets]

print("\n=== Correlation: Biomaterials vs Target Properties ===")
print(biomaterial_vs_targets.round(3))
biomaterial_vs_targets.to_csv("data/biomaterial_vs_target_correlation.csv")

plt.figure(figsize=(10, 5))
sns.heatmap(biomaterial_vs_targets, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation: Biomaterial Amount vs Mud Properties")
plt.tight_layout()
plt.savefig("plots/biomaterial_vs_target_heatmap.png", dpi=150)
print("Saved: plots/biomaterial_vs_target_heatmap.png")