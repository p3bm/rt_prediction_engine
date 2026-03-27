import json
import joblib
import os
from datetime import datetime

def create_run_dir(base="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, timestamp)
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed):
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)

def save_model(path, model_dict):
    joblib.dump(model_dict, path)

def save_log(path, log_dict):
    log_dict["timestamp"] = str(datetime.now())
    with open(path, "w") as f:
        json.dump(log_dict, f, indent=2)
