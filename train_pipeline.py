import argparse
from pathlib import Path

import pandas as pd

from sia_core import load_tickets, train_model


def main():
    parser = argparse.ArgumentParser(description="Train the Support Integrity Auditor classifier.")
    parser.add_argument("--data", default="data/sample_tickets.csv", help="Path to a CSV of support tickets.")
    parser.add_argument("--model-dir", default="models", help="Directory where model artifacts will be saved.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}. Put the Kaggle CSV there or use data/sample_tickets.csv.")

    tickets = load_tickets(data_path)
    model, metrics, labeled = train_model(tickets, args.model_dir)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    labeled.to_csv(reports_dir / "pseudo_labeled_tickets.csv", index=False)

    print("Training complete.")
    print(f"Rows used: {metrics['row_count']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Per-class recall: {metrics['per_class_recall']}")
    print(f"Pseudo-label signal agreement: {metrics['pseudo_label_signal_agreement']:.3f}")
    print(f"Saved model to {Path(args.model_dir) / 'sia_model.pkl'}")


if __name__ == "__main__":
    main()
