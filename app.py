import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib
import sys
import os

sys.path.append("scripts")
from optimizer_core import (
    load_models, run_random_search, run_bayesian, predict_all,
    compute_concentrations
)

st.set_page_config(page_title="WBM Bio-Additive Recipe Optimizer", layout="wide")

st.title("WBM Bio-Additive Recipe Optimizer")
st.caption("Predict optimal bio-additive recipes for target mud properties.")

# ---- Load models (cached so it doesn't reload every interaction) ----
@st.cache_resource
def get_models():
    return load_models()

models = get_models()

# ---- Load model confidence report (if it exists) ----
@st.cache_data
def get_confidence_flags():
    path = "data/model_performance_report.csv"
    if os.path.exists(path):
        report = pd.read_csv(path)
        return dict(zip(report["Target"], report["Confidence"]))
    return {}

confidence_flags = get_confidence_flags()

# ---- Sidebar: biomaterial toggles ----
st.sidebar.header("Biomaterials to Include")
biomaterial_labels = {
    "E_g": "Eggshell Powder",
    "B_g": "Banana Peel Powder",
    "G_g": "Groundnut Shell",
    "O_g": "Okra",
    "CS_g": "Cornstarch",
}

enabled_biomaterials = []
for col, label in biomaterial_labels.items():
    if st.sidebar.checkbox(label, value=True, key=f"bio_{col}"):
        enabled_biomaterials.append(col)

st.sidebar.divider()

# ---- Sidebar: target inputs ----
st.sidebar.header("Target Properties")

default_targets = {
    "PV_cP": 15.0,
    "YP_lb100ft2": 20.0,
    "Density_ppg": 9.5,
    "Gel_10s": 10.0,
    "Gel_10min": 15.0,
    "Fluid_Loss_mL": 8.0,
    "Mud_Cake_mm": 2.5,
}

target_values = {}
for target, default in default_targets.items():
    include = st.sidebar.checkbox(f"Include {target}", value=True, key=f"chk_{target}")
    if include:
        val = st.sidebar.number_input(target, value=default, key=f"val_{target}")
        target_values[target] = val

st.sidebar.divider()
n_random = st.sidebar.number_input("Random Search candidates", value=20000, step=1000)
n_trials = st.sidebar.number_input("Bayesian trials", value=500, step=50)

run_button = st.sidebar.button("Run Optimization", type="primary")

# ---- Ingredient display setup ----
ingredient_names = {
    "E_g": "Eggshell", "B_g": "Banana Peel", "G_g": "Groundnut Shell",
    "O_g": "Okra", "CS_g": "Cornstarch",
    "Water_mL": "Water", "Be_g": "Bentonite", "Ba_g": "Barite"
}

def build_recipe_table(recipe_source, ingredient_cols, concentrations):
    """recipe_source can be a pandas Series (Random Search row) or a dict (Bayesian best_recipe)."""
    rows = []
    for c in ingredient_cols:
        amount = recipe_source[c]
        unit = "mL" if c == "Water_mL" else "g"
        rows.append({
            "Ingredient": ingredient_names[c],
            "Amount": f"{amount:.2f} {unit}",
            "Concentration (wt%)": f"{concentrations[c]:.2f}%"
        })
    return pd.DataFrame(rows)

# ---- Main panel ----
if run_button:
    if not target_values:
        st.error("Select at least one target property before running.")
        st.stop()
    if not enabled_biomaterials:
        st.error("Select at least one biomaterial before running.")
        st.stop()

    with st.spinner("Running Random Search..."):
        rs_results = run_random_search(models, target_values, enabled_biomaterials, n=int(n_random))
        rs_best = rs_results.iloc[0]

    with st.spinner("Running Bayesian Optimization..."):
        study, bo_best_recipe = run_bayesian(models, target_values, enabled_biomaterials, n_trials=int(n_trials))
        bo_predictions = predict_all(models, bo_best_recipe)

    st.success(f"Optimized using: {', '.join(biomaterial_labels[c] for c in enabled_biomaterials)}")

    ingredient_cols = [c for c in ["E_g", "B_g", "G_g", "O_g", "CS_g"] if c in enabled_biomaterials]
    ingredient_cols += ["Water_mL", "Be_g", "Ba_g"]

    # Compute wt% concentrations for both recipes
    rs_concentrations = compute_concentrations(rs_best.to_dict())
    bo_concentrations = compute_concentrations(bo_best_recipe)

    col1, col2 = st.columns(2)

    # ---------------- Random Search results ----------------
    with col1:
        st.subheader("Random Search—Recipe")
        recipe_table = build_recipe_table(rs_best, ingredient_cols, rs_concentrations)
        st.table(recipe_table)

        st.markdown("**Predicted vs Target**")
        comparison = pd.DataFrame({
            "Target": list(target_values.keys()),
            "Predicted": [rs_best[f"pred_{t}"] for t in target_values],
            "Desired": list(target_values.values()),
        })
        comparison["Diff %"] = ((comparison["Predicted"] - comparison["Desired"]) / comparison["Desired"] * 100).round(1)
        if confidence_flags:
            comparison["Confidence"] = comparison["Target"].map(confidence_flags)
        st.table(comparison.round(2))

        st.metric("Total Error Score", f"{rs_best['total_error']:.4f}")

        sorted_errors = rs_results["total_error"].sort_values().reset_index(drop=True)
        running_best = sorted_errors.cummin()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(len(running_best)), running_best)
        ax.set_xlabel("Candidates evaluated")
        ax.set_ylabel("Best error so far")
        ax.set_title("Random Search Convergence")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

    # ---------------- Bayesian results ----------------
    with col2:
        st.subheader("Bayesian Optimization—Recipe")
        recipe_table_bo = build_recipe_table(bo_best_recipe, ingredient_cols, bo_concentrations)
        st.table(recipe_table_bo)

        st.markdown("**Predicted vs Target**")
        comparison_bo = pd.DataFrame({
            "Target": list(target_values.keys()),
            "Predicted": [bo_predictions[t] for t in target_values],
            "Desired": list(target_values.values()),
        })
        comparison_bo["Diff %"] = ((comparison_bo["Predicted"] - comparison_bo["Desired"]) / comparison_bo["Desired"] * 100).round(1)
        if confidence_flags:
            comparison_bo["Confidence"] = comparison_bo["Target"].map(confidence_flags)
        st.table(comparison_bo.round(2))

        st.metric("Total Error Score", f"{study.best_value:.4f}")

        fig2 = optuna.visualization.matplotlib.plot_optimization_history(study)
        st.pyplot(fig2.figure)

    # ---------------- Side-by-side comparison ----------------
    st.divider()
    st.subheader("Recipe Comparison (Concentration, wt%)")
    side_by_side = pd.DataFrame({
        "Ingredient": [ingredient_names[c] for c in ingredient_cols],
        "Random Search (wt%)": [f"{rs_concentrations[c]:.2f}%" for c in ingredient_cols],
        "Bayesian Optimization (wt%)": [f"{bo_concentrations[c]:.2f}%" for c in ingredient_cols],
    })
    st.table(side_by_side)

    # ---------------- CSV export ----------------
    st.divider()
    export_df = pd.DataFrame({
        "Ingredient": [ingredient_names[c] for c in ingredient_cols],
        "Random_Search_g_or_mL": [rs_best[c] for c in ingredient_cols],
        "Random_Search_wt%": [rs_concentrations[c] for c in ingredient_cols],
        "Bayesian_g_or_mL": [bo_best_recipe[c] for c in ingredient_cols],
        "Bayesian_wt%": [bo_concentrations[c] for c in ingredient_cols],
    })
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Recipe Comparison (CSV)", csv_data, "recipe_comparison.csv", "text/csv")

else:
    st.info("Select biomaterials and target properties in the sidebar, then click 'Run Optimization'.")