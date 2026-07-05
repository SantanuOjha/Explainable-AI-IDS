"""Data loading, feature schema and model deserialization helpers.

The notebooks trained on the CICIDS2017 combined dataset after:
    - stripping whitespace from column names
    - replacing +/-inf with NaN and dropping NaN rows
    - LabelEncoder + binarization (0 = BENIGN, 1 = ATTACK)
    - StandardScaler fit on the train split

The persisted CSVs under Data/Sampled_Data/ are already in *scaled*
feature space with integer column headers, so this module re-attaches
the human-readable CICIDS2017 feature names.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "Sampled_Data"

CLASS_NAMES = ["BENIGN", "ATTACK"]

# CICIDS2017 flow features, in training order (header of the combined CSV,
# whitespace stripped, Label column removed).
FEATURE_NAMES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Display name -> pickle filename, as produced by notebooks 03/04.
MODEL_REGISTRY = {
    "Random Forest (baseline)": "rf.pkl",
    "Random Forest (SMOTE)": "rf_smote.pkl",
    "Random Forest (Borderline-SMOTE)": "rf_smote_bl.pkl",
    "XGBoost (SMOTE)": "XG_smote.pkl",
    "XGBoost (Borderline-SMOTE)": "XG_smote_BL.pkl",
    "XGBoost (KMeans-SMOTE)": "XG_smote_kmeans.pkl",
}


def available_models() -> dict[str, Path]:
    """Registry entries whose pickle actually exists on disk."""
    found = {}
    for name, filename in MODEL_REGISTRY.items():
        path = DATA_DIR / filename
        if path.exists():
            found[name] = path
    return found


def load_model(path: str | Path):
    """Deserialize a trained model (joblib format, as saved by the notebooks)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run the training notebooks first "
            "or mount the Data/ directory."
        )
    return joblib.load(path)


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a feature frame to the numeric matrix the models expect.

    NIDS exports commonly contain stray categorical/object columns and
    +/-inf values (e.g. Flow Bytes/s on zero-duration flows). Any value
    that cannot be parsed as a number becomes 0, mirroring the notebook's
    inf->NaN->drop policy without silently discarding live traffic rows.
    """
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_test_data(n_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Load the held-out test split with real feature names attached.

    Returns (X, y) where X columns follow FEATURE_NAMES and
    y is 0 = BENIGN, 1 = ATTACK.
    """
    x = pd.read_csv(DATA_DIR / "X_test.csv", nrows=n_rows)
    y = pd.read_csv(DATA_DIR / "y_test.csv", nrows=n_rows).squeeze("columns")
    x.columns = FEATURE_NAMES
    # y_test.csv stores the multiclass LabelEncoder output (BENIGN sorts
    # first alphabetically -> 0). The models were trained binarized, so
    # collapse every attack class to 1, matching notebook 03's y_train_bin.
    return sanitize_features(x), (y.astype(int) != 0).astype(int)
