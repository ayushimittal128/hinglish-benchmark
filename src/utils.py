# src/utils.py
import json, os, time
import pandas as pd

def load_csv(path, text_col="text", label_col="label", sample=None):
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV at {path} must contain columns: {text_col}, {label_col}")
    df = df[[text_col, label_col]].dropna()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42)
    return df.rename(columns={text_col: "text", label_col: "label"})

def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("-", " ").replace("/", " ")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def dump_jsonl(records, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def now_ms():
    return int(time.time()*1000)
