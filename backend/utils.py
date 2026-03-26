import json
import joblib
from datetime import datetime

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
