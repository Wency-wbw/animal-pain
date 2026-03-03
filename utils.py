
import os, json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

def macro_pr(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy":float(acc), "precision":float(p), "recall":float(r), "f1":float(f1)}

def save_json(obj, path):
    with open(path, "w") as f: json.dump(obj, f, indent=2)
