"""
revert_model_paths.py  – fixes the incorrect model-path patch applied earlier.
The notebook trains on ENGINEERED features; the app uses RAW features.
They must remain separate models to avoid breaking the app.
"""
import json
from pathlib import Path

nb_path = Path("notebooks/4_modeling.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    s = "".join(cell["source"])

    if "credit_risk_model_pure.joblib" in s and "credit_risk_model_full" not in s:
        # This is the save cell – revert the primary model name to *_full
        # so it doesn't overwrite the app's model (which uses raw features)
        new_s = s
        new_s = new_s.replace(
            'joblib.dump(final_model, "models/credit_risk_model.joblib")',
            'joblib.dump(final_model, "models/credit_risk_model_full.joblib")',
        )
        new_s = new_s.replace(
            'joblib.dump(FULL_FEATURES, "models/expected_features.joblib")',
            'joblib.dump(FULL_FEATURES, "models/expected_features_full.joblib")',
        )
        # fix the print lines too
        new_s = new_s.replace(
            '"  models/credit_risk_model.joblib"',
            '"  models/credit_risk_model_full.joblib"',
        )
        new_s = new_s.replace(
            '"  models/expected_features.joblib"',
            '"  models/expected_features_full.joblib"',
        )

        if new_s != s:
            cell["source"] = new_s.splitlines(keepends=True)
            cell["outputs"] = []
            cell["execution_count"] = None
            print("[REVERTED] Model save paths kept as _full/_pure (separate from app model)")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("[SAVED] 4_modeling.ipynb")
