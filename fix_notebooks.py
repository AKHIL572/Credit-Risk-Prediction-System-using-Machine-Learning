"""
fix_notebooks.py
Patches all known bugs in the project notebooks (JSON manipulation).
Run once from the project root: python fix_notebooks.py
"""
# force UTF-8 output on Windows
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
from pathlib import Path

NB_DIR = Path("notebooks")

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def load(name):
    with open(NB_DIR / name, encoding="utf-8") as f:
        return json.load(f)

def save(name, nb):
    with open(NB_DIR / name, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [SAVED] {name}")

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    lines = text.splitlines(keepends=True)
    cell["source"] = lines
    cell["outputs"] = []
    cell["execution_count"] = None

def clear_outputs(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3_data_preprocessing.ipynb
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Fixing 3_data_preprocessing.ipynb --")
nb3 = load("3_data_preprocessing.ipynb")

for cell in nb3["cells"]:
    s = src(cell)

    # BUG 1: typo "Libreries"
    if "Import Libreries" in s:
        set_src(cell, s.replace("Import Libreries", "Import Libraries"))
        print("  [FIXED] Typo: 'Libreries' -> 'Libraries'")

    # BUG 2: redundant duplicate save to final_model_data.csv
    if "final_model_data.csv" in s and "processed_data.csv" not in s:
        new_src = (
            "# NOTE: final_model_data.csv was a redundant duplicate of processed_data.csv.\n"
            "# Removed to avoid wasting ~282 MB of disk space.\n"
            "# Use data/processed/processed_data.csv as input for 4_modeling.ipynb.\n"
            "print('Redundant save skipped. Use processed_data.csv instead.')\n"
        )
        set_src(cell, new_src)
        print("  [FIXED] Removed redundant final_model_data.csv save")

save("3_data_preprocessing.ipynb", nb3)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 4_modeling.ipynb
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Fixing 4_modeling.ipynb --")
nb4 = load("4_modeling.ipynb")

for cell in nb4["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = src(cell)

    # BUG 3: stale FileNotFoundError output in data-load cell
    if "processed_data.csv" in s and cell.get("outputs"):
        has_error = any(o.get("output_type") == "error" for o in cell["outputs"])
        if has_error:
            cell["outputs"] = []
            cell["execution_count"] = None
            print("  [FIXED] Cleared stale FileNotFoundError output in data-load cell")

    # BUG 4: model saved as credit_risk_model_business.joblib but app.py
    #         loads credit_risk_model.joblib — fix the save paths to match.
    if "credit_risk_model_business.joblib" in s:
        new_src = (s
            .replace(
                'joblib.dump(final_model, "models/credit_risk_model_business.joblib")',
                'joblib.dump(final_model, "models/credit_risk_model.joblib")'
            )
            .replace(
                'joblib.dump(FULL_FEATURES, "models/expected_features_full.joblib")',
                'joblib.dump(FULL_FEATURES, "models/expected_features.joblib")'
            )
            .replace(
                '"  models/credit_risk_model_business.joblib"',
                '"  models/credit_risk_model.joblib"'
            )
            .replace(
                '"  models/expected_features_full.joblib"',
                '"  models/expected_features.joblib"'
            )
        )
        set_src(cell, new_src)
        print("  [FIXED] Model save paths now match app.py expectations")

save("4_modeling.ipynb", nb4)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 5_business_insights.ipynb
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- Fixing 5_business_insights.ipynb --")
nb5 = load("5_business_insights.ipynb")

for cell in nb5["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = src(cell)
    changed = False

    # BUG 5: wrong dataset read path ("Dataset" instead of "data/raw")
    if 'PROJECT_ROOT / "Dataset" / "dataset.csv"' in s:
        s = s.replace(
            'DATASET_PATH = PROJECT_ROOT / "Dataset" / "dataset.csv"',
            'DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"'
        )
        changed = True
        print("  [FIXED] Dataset read path: 'Dataset/' -> 'data/raw/'")

    # BUG 6: wrong output path for dashboard_data.csv
    if 'PROJECT_ROOT / "Dataset" / "dashboard_data.csv"' in s:
        s = s.replace(
            'CLEAN_PATH = PROJECT_ROOT / "Dataset" / "dashboard_data.csv"',
            'CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard_data.csv"'
        )
        changed = True
        print("  [FIXED] dashboard_data.csv output: 'Dataset/' -> 'data/processed/'")

    # BUG 7: wrong output dir for risk tables
    if 'DATASET_DIR = PROJECT_ROOT / "Dataset"' in s:
        s = s.replace(
            'DATASET_DIR = PROJECT_ROOT / "Dataset"',
            'DATASET_DIR = PROJECT_ROOT / "data" / "processed"'
        )
        changed = True
        print("  [FIXED] Risk table output dir: 'Dataset/' -> 'data/processed/'")

    # BUG 8: loan_amnt assertion fails when NaN present (NaN > 0 == False)
    if 'df["loan_amnt"].gt(0).all()' in s:
        s = s.replace(
            'assert df["loan_amnt"].gt(0).all(), "loan_amnt contains non-positive values"',
            'assert df["loan_amnt"].dropna().gt(0).all(), "loan_amnt contains non-positive values"'
        )
        changed = True
        print("  [FIXED] loan_amnt assertion now skips NaN values")

    # BUG 9: groupby on income_bracket without observed=True -> FutureWarning
    if 'groupby("income_bracket")' in s and 'observed=True' not in s:
        s = s.replace(
            'df.groupby("income_bracket")["is_default"]',
            'df.groupby("income_bracket", observed=True)["is_default"]'
        ).replace(
            'df_resolved.groupby("income_bracket")["is_default"]',
            'df_resolved.groupby("income_bracket", observed=True)["is_default"]'
        )
        changed = True
        print("  [FIXED] Added observed=True to income_bracket groupby")

    if changed:
        set_src(cell, s)

# Clear all stale outputs for a clean re-run
clear_outputs(nb5)
print("  [FIXED] Cleared all stale outputs for fresh re-run")

save("5_business_insights.ipynb", nb5)

print("\n[DONE] All notebook fixes applied successfully.\n")
