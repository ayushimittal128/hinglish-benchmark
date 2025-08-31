# run_benchmark.py
import argparse, os
from dotenv import load_dotenv
from src.benchmark import HinglishBenchmark

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sentiment","hate_speech"], help="Run a single task")
    p.add_argument("--sample", type=int, default=None, help="Sample N rows per selected task")
    p.add_argument("--all", action="store_true", help="Run both tasks")
    return p.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    bench = HinglishBenchmark()

    # sampling per task
    sample_sent = args.sample if (args.task == "sentiment" or args.all) else (args.sample if args.task == "sentiment" else None)
    sample_hate = args.sample if (args.task == "hate_speech" or args.all) else (args.sample if args.task == "hate_speech" else None)

    # load data with optional sampling
    bench.load_datasets(sample_sentiment=sample_sent if args.task in (None,"sentiment") or args.all else None,
                        sample_hate=sample_hate if args.task in (None,"hate_speech") or args.all else None)

    if args.all or args.task is None:
        outputs, metrics_df = bench.run_all_experiments()
    else:
        which = "sentiment" if args.task == "sentiment" else "hate_speech"
        outputs, metrics_df = bench.run_all_experiments(which=which)

    bench.save_results(outputs, metrics_df)
    bench.generate_report()
    print(metrics_df)

if __name__ == "__main__":
    main()
