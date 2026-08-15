import pandas as pd
import numpy as np
import joblib
import optuna
TARGETS = [
    "Density_ppg", "PV_cP", "YP_lb100ft2",
    "Gel_10s", "Gel_10min", "Fluid_Loss_mL", "Mud_Cake_mm"
]
FEATURE_COLUMNS = [
    "E_g", "B_g", "G_g", "O_g", "CS_g", "CH_g", "Water_mL",
    "Be_g", "Ba_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
    "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"
]
# The 5 toggleable biomaterials, and their default max search value
BIOMATERIAL_RANGES = {
    "E_g":  (0, 30),   # Eggshell
    "B_g":  (0, 10),   # Banana Peel
    "G_g":  (0, 10),   # Groundnut Shell
    "O_g":  (0, 4),    # Okra
    "CS_g": (0, 10),   # Cornstarch
}
# Always-available process variables (never toggled off)
FIXED_RANGES = {
    "Water_mL": (300, 400),
    "Be_g": (0, 25),   # Bentonite
    "Ba_g": (0, 100),  # Barite
}
# Everything else in the dataset that this project scope doesn't search over
ALWAYS_LOCKED_AT_ZERO = ["CH_g", "CP_g", "Xan_g", "CaCO3_g", "PAC_g",
                          "Lime_g", "NaOH_g", "PS_g", "PHU_g", "SS_g", "D_g", "KCl_g"]
def load_models():
    return {t: joblib.load(f"models/{t}_model.pkl") for t in TARGETS}
def build_search_ranges(enabled_biomaterials):
    """
    enabled_biomaterials: list of column names to allow, e.g. ["E_g", "CS_g"]
    Any biomaterial NOT in this list gets locked to 0 (excluded from the search).
    """
    ranges = dict(FIXED_RANGES)  # always include bentonite/barite/water
    for col in BIOMATERIAL_RANGES:
        if col in enabled_biomaterials:
            ranges[col] = BIOMATERIAL_RANGES[col]
        # if not enabled, we simply don't add it to ranges — it gets locked to 0 later
    return ranges
def get_locked_columns(search_ranges):
    """Every feature column NOT in search_ranges gets fixed at 0."""
    return [col for col in FEATURE_COLUMNS if col not in search_ranges] 
def predict_all(models, recipe_dict):
    X = pd.DataFrame([recipe_dict])[FEATURE_COLUMNS]
    return {t: models[t].predict(X)[0] for t in TARGETS}
# ---------------- Random Search ----------------
def generate_candidates(search_ranges, n=20000, seed=42):
    rng = np.random.default_rng(seed)
    candidates = pd.DataFrame({
        col: rng.uniform(low, high, n) for col, (low, high) in search_ranges.items()
    })
    locked_cols = get_locked_columns(search_ranges)
    for col in locked_cols:
        candidates[col] = 0
    return candidates[FEATURE_COLUMNS]
def run_random_search(models, target_values, enabled_biomaterials, n=20000):
    search_ranges = build_search_ranges(enabled_biomaterials)
    candidates = generate_candidates(search_ranges, n=n)
    X = candidates[FEATURE_COLUMNS].copy()
    results = candidates.copy()
    total_error = np.zeros(len(candidates))
    for target, desired in target_values.items():
        preds = models[target].predict(X)
        error = np.abs(preds - desired) / (abs(desired) + 1e-6)
        total_error += error
        results[f"pred_{target}"] = preds
    results["total_error"] = total_error
    results = results.sort_values("total_error").reset_index(drop=True)
    return results
# ---------------- Bayesian Optimization ----------------
def run_bayesian(models, target_values, enabled_biomaterials, n_trials=500, seed=42):
    search_ranges = build_search_ranges(enabled_biomaterials)
    locked_cols = get_locked_columns(search_ranges)
    def objective(trial):
        recipe = {
            col: trial.suggest_float(col, low, high)
            for col, (low, high) in search_ranges.items()
        }
        for col in locked_cols:
            recipe[col] = 0
        X = pd.DataFrame([recipe])[FEATURE_COLUMNS]
        total_error = 0
        for target, desired in target_values.items():
            pred = models[target].predict(X)[0]
            total_error += abs(pred - desired) / (abs(desired) + 1e-6)
        return total_error
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best_recipe = study.best_params.copy()
    for col in locked_cols:
        best_recipe[col] = 0
    return study, best_recipe
def compute_concentrations(recipe_dict):
    """
    Converts a recipe (grams + water mL) into weight percent (wt%) concentrations.
    Assumes water density = 1 g/mL, so water_mL grams == water_mL.
    """
    solid_columns = [c for c in FEATURE_COLUMNS if c != "Water_mL"]
    total_solid_mass = sum(recipe_dict.get(c, 0) for c in solid_columns)
    water_mass = recipe_dict.get("Water_mL", 0)  # 1 g/mL assumption
    total_mass = total_solid_mass + water_mass
    if total_mass == 0:
        return {c: 0.0 for c in FEATURE_COLUMNS}
    concentrations = {
        c: (recipe_dict.get(c, 0) / total_mass) * 100
        for c in FEATURE_COLUMNS
    }
    return concentrations