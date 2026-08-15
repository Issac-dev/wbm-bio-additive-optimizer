import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

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

# ---- 2. Define the search space — bio-additives + bentonite + barite + water only ----
search_ranges = {
    "E_g":  (0, 30),      # Eggshell
    "B_g":  (0, 10),      # Banana Peel
    "G_g":  (0, 10),      # Groundnut Shell
    "O_g":  (0, 4),       # Okra
    "CS_g": (0, 10),      # Cornstarch
    "Water_mL": (300, 400),
    "Be_g": (0, 25),      # Bentonite
    "Ba_g": (0, 100),     # Barite
}

def generate_candidates(n=20000):
    rng = np.random.default_rng(42)
    candidates = pd.DataFrame({
        col: rng.uniform(low, high, n)
        for col, (low, high) in search_ranges.items()
    })
    locked_at_zero = ["CH_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
                       "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"]
    for col in locked_at_zero:
        candidates[col] = 0
    return candidates[feature_columns]

def score_candidates(candidates, target_values, weights=None):
    if weights is None:
        weights = {t: 1.0 for t in target_values}

    X = candidates[feature_columns].copy()

    total_error = np.zeros(len(candidates))
    results = candidates.copy()

    for target, desired_value in target_values.items():
        predictions = models[target].predict(X)
        error = np.abs(predictions - desired_value) / (abs(desired_value) + 1e-6)
        total_error += weights.get(target, 1.0) * error
        results[f"pred_{target}"] = predictions

    results["total_error"] = total_error
    return results

# ---- 3. Set your target values ----
target_values = {
    "PV_cP": 15,
    "YP_lb100ft2": 20,
    "Density_ppg": 9.5,
    "Gel_10s": 10,
    "Gel_10min": 15,
    "Fluid_Loss_mL": 8,
    "Mud_Cake_mm": 2.5,
}

candidates = generate_candidates(n=20000)
scored = score_candidates(candidates, target_values)

# ---- 4. Show the top 5 recipes ----
best = scored.sort_values("total_error").head(5)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)
print("=== Top 5 recommended recipes ===")
print(best.round(2))

# ---- 5. Clearly state the single best result ----
best_recipe = best.iloc[0]

print("\n" + "="*60)
print("BEST OVERALL RECIPE (Random Search)")
print("="*60)
print("\nRecipe (grams unless noted):")
for col in ["E_g", "B_g", "G_g", "O_g", "CS_g", "Water_mL", "Be_g", "Ba_g"]:
    print(f"  {col:12s}: {best_recipe[col]:.2f}")

print("\nPredicted vs Target properties:")
for target, desired in target_values.items():
    predicted = best_recipe[f"pred_{target}"]
    diff = predicted - desired
    pct_diff = (diff / desired) * 100
    print(f"  {target:18s}: predicted={predicted:.2f}  target={desired:.2f}  "
          f"diff={diff:+.2f} ({pct_diff:+.1f}%)")

print(f"\nOverall total_error score: {best_recipe['total_error']:.4f}")
print("(Lower is better — 0 would mean a perfect match to every target)")
print("="*60)

# ---- 6. Plot: convergence over candidates ----
sorted_errors = scored["total_error"].sort_values().reset_index(drop=True)
# ---- Plot 1: How error improves as candidates are evaluated (in original random order) ----
# Do NOT sort first — we want to see the running best AS the random search actually explored,
# in the order candidates were generated.
running_best = scored["total_error"].cummin()

plt.figure(figsize=(8, 5))
plt.plot(range(len(running_best)), running_best)
plt.xlabel("Number of candidates evaluated")
plt.ylabel("Best total_error found so far")
plt.title("Random Search: Convergence over 20,000 candidates")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/random_search_convergence.png", dpi=150)

# ---- 7. Plot: predicted vs target for top 5 candidates ----
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, target in enumerate(target_values):
    ax = axes[i]
    predicted = best[f"pred_{target}"].values
    target_val = target_values[target]
    ax.bar(range(len(predicted)), predicted, alpha=0.7, label="Predicted")
    ax.axhline(target_val, color="red", linestyle="--", label="Target")
    ax.set_title(target)
    ax.set_xlabel("Top 5 candidates (rank)")
    ax.legend(fontsize=8)

for j in range(len(target_values), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("plots/random_search_top5_vs_target.png", dpi=150)
print("Saved: plots/random_search_top5_vs_target.png")