# src/benchmark.py
import os, time
import pandas as pd
from tqdm import tqdm
from .models import GeminiClient
from .evaluation import compute_metrics
from .utils import load_csv, normalize_label, ensure_dir, dump_jsonl

SENTIMENT_LABELS = ["positive", "negative", "neutral"]
HATE_LABELS = ["hate", "not hate"]

class HinglishBenchmark:
    def __init__(self, data_dir="data", results_dir="results"):
        self.data_dir = data_dir
        self.results_dir = results_dir
        ensure_dir(self.results_dir)
        self.client = GeminiClient()

    def load_datasets(self, sentiment_file="sentiment_data.csv", hate_file="hate_speech_data.csv",
                      sample_sentiment=None, sample_hate=None):
        self.df_sent = load_csv(os.path.join(self.data_dir, sentiment_file), "text", "label", sample_sentiment)
        self.df_hate = load_csv(os.path.join(self.data_dir, hate_file), "text", "label", sample_hate)
        # normalize gt labels
        self.df_sent["label"] = self.df_sent["label"].map(normalize_label)
        self.df_hate["label"] = self.df_hate["label"].map(normalize_label)

    def _run_task(self, df: pd.DataFrame, task_name: str, allowed_labels: list[str]):
        records = []
        preds, gts = [], []
        errors = 0
        t0 = time.time()
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{task_name}"):
            text = str(row["text"])
            gt = normalize_label(row["label"])
            out = self.client.classify(text, task=task_name, labels=allowed_labels)
            pred = out["label"]
            if pred is None:
                errors += 1
                # fallback to first label to keep vector sizes aligned
                pred = allowed_labels[0]
            preds.append(pred)
            gts.append(gt)
            records.append({
                "task": task_name,
                "text": text,
                "gt": gt,
                "pred": pred,
                "raw": out.get("raw"),
                "latency_ms": out.get("latency_ms"),
                "error": out.get("error")
            })
        elapsed = time.time() - t0
        metrics = compute_metrics(gts, preds, labels=allowed_labels)
        metrics["errors"] = errors
        metrics["elapsed_sec"] = round(elapsed, 2)
        return records, metrics

    def run_all_experiments(self, which: str | None = None):
        results = []
        outputs_all = []

        if which in (None, "sentiment"):
            recs, m = self._run_task(self.df_sent, "sentiment analysis", SENTIMENT_LABELS)
            outputs_all.extend(recs)
            m["task"] = "sentiment"
            results.append(m)

        if which in (None, "hate_speech"):
            recs, m = self._run_task(self.df_hate, "hate speech detection", HATE_LABELS)
            outputs_all.extend(recs)
            m["task"] = "hate_speech"
            results.append(m)

        return outputs_all, pd.DataFrame(results)[["task","accuracy","f1_macro","precision","recall","total_samples","errors","elapsed_sec"]]

    def save_results(self, outputs, metrics_df):
        dump_jsonl(outputs, os.path.join(self.results_dir, "outputs.jsonl"))
        metrics_path = os.path.join(self.results_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved: {metrics_path}")

    def generate_report(self):
        # ultra-simple report (you can expand later)
        print("Report generation: open results/metrics.csv and plot if needed.")
