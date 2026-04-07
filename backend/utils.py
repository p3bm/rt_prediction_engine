import json
import joblib
import os
import numpy as np
import pandas as pd
from datetime import datetime
import random

def create_run_dir(base="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def save_model(path, model_dict):
    joblib.dump(model_dict, path)

def clean_params(params):
    """Ensure model objects inside params are converted to strings."""
    cleaned = {}
    for k, v in params.items():
        if hasattr(v, "__class__"):
            cleaned[k] = str(v)
        else:
            cleaned[k] = v
    return {k: v for k, v in cleaned.items() if k != "model"}

def make_json_serializable(obj):
    """
    Recursively convert objects into JSON-serializable types.
    Handles numpy, pandas, datetime, etc.
    """

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(v) for v in obj)

    elif isinstance(obj, (np.integer,)):
        return int(obj)

    elif isinstance(obj, (np.floating,)):
        return float(obj)

    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    elif isinstance(obj, (pd.Series,)):
        return obj.tolist()

    elif isinstance(obj, (pd.Index,)):
        return obj.tolist()

    elif isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict()

    elif isinstance(obj, datetime):
        return obj.isoformat()

    else:
        return obj

def save_log(path: str, log_dict: dict):
    """
    Save experiment log as JSON with full compatibility.
    """

    # Add timestamp
    log_dict["timestamp"] = datetime.now()

    # Convert to JSON-safe structure
    safe_log = make_json_serializable(log_dict)

    with open(path, "w") as f:
        json.dump(safe_log, f, indent=2)
