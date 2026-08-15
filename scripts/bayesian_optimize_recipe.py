import pandas as pd
import numpy as np
import joblib
import optuna
import optuna.visualization.matplotlib
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# ---- 1. Load the trained models ----
targets = [
    "Density_ppg", "PV_cP", "YP_lb100ft2",
    "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"
]
models = {t: joblib.load(f"models/{t}_model.pkl") for t in targets}

feature_columns = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]

# ---- 2. Set your target values ----
target_values = {
    "PV_cP": 15,
    "YP_lb100ft2": 20,
    "Density_ppg": 9.5,
    "Gel_10s": 10,
    "Gel_10min": 15,
    "Fluid_Loss_mL": 8,
    "Mud_Cake_mm": 2.5,
}

# ---- 3. Define the objective function ----
def objective(trial):
    recipe = {
        "E_g": trial.suggest_float("E_g", 0, 30),
        "B_g": trial.suggest_float("B_g", 0, 10),
        "G_g": trial.suggest_float("G_g", 0, 10),
        "O_g": trial.suggest_float("O_g", 0, 4),
        "CS_g": trial.suggest_float("CS_g", 0, 10),
        "Water_mL": trial.suggest_float("Water_mL", 300, 400),
        "Be_g": trial.suggest_float("Be_g", 0, 25),
        "Ba_g": trial.suggest_float("Ba_g", 0, 100),
    }
    locked_at_zero = ["CH_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
                       "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"]
    for col in locked_at_zero:
        recipe[col] = 0

    X = pd.DataFrame([recipe])[feature_columns]

    total_error = 0
    for target, desired_value in target_values.items():
        prediction = models[target].predict(X)[0]
        error = abs(prediction - desired_value) / (abs(desired_value) + 1e-6)
        total_error += error

    return total_error

# ---- 4. Run the Bayesian optimization search (seeded for reproducibility) ----
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=500, show_progress_bar=True)

# ---- 5. Build the best recipe dataframe ----
best_recipe = study.best_params
for col in ["CH_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
            "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"]:
    best_recipe[col] = 0

X_best = pd.DataFrame([best_recipe])[feature_columns]

# ---- 6. Clearly state the single best result ----
print("\n" + "="*60)
print("BEST OVERALL RECIPE (Bayesian Optimization)")
print("="*60)

print("\nRecipe (grams unless noted):")
for col in ["E_g", "B_g", "G_g", "O_g", "CS_g", "Water_mL", "Be_g", "Ba_g"]:
    print(f"  {col:12s}: {best_recipe[col]:.2f}")

print("\nPredicted vs Target properties:")
for target, desired in target_values.items():
    predicted = models[target].predict(X_best)[0]
    diff = predicted - desired
    pct_diff = (diff / desired) * 100
    print(f"  {target:18s}: predicted={predicted:.2f}  target={desired:.2f}  "
          f"diff={diff:+.2f} ({pct_diff:+.1f}%)")

print(f"\nOverall total_error score: {study.best_value:.4f}")
print("(Lower is better — 0 would mean a perfect match to every target)")
print("="*60)

# ---- 7. Plot: Optuna's convergence history ----
fig = optuna.visualization.matplotlib.plot_optimization_history(study)
fig.figure.savefig("plots/bayesian_convergence.png", dpi=150)
print("\nSaved: plots/bayesian_convergence.png")

# ---- 8. Plot: parameter importance ----
fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
fig2.figure.savefig("plots/bayesian_param_importance.png", dpi=150)
print("Saved: plots/bayesian_param_importance.png")

# ---- 9. Plot: predicted vs target bar chart ----
predicted_values = [models[t].predict(X_best)[0] for t in target_values]
target_list = list(target_values.values())
labels = list(target_values.keys())

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, predicted_values, width, label="Predicted", color="steelblue")
plt.bar(x + width/2, target_list, width, label="Target", color="indianred")
plt.xticks(x, labels, rotation=30, ha="right")
plt.ylabel("Value")
plt.title("Bayesian Optimization: Best Recipe vs Targets")
plt.legend()
plt.tight_layout()
plt.savefig("plots/bayesian_best_vs_target.png", dpi=150)
print("Saved: plots/bayesian_best_vs_target.png")