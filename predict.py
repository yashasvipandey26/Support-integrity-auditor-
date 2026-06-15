import argparse
import json
from pathlib import Path

import pandas as pd

from sia_core import load_model, predict_with_dossiers


def main():
    parser = argparse.ArgumentParser(description="Predict priority mismatches and generate evidence dossiers.")
    parser.add_argument("--input", required=True, help="CSV file containing tickets to audit.")
    parser.add_argument("--model", default="models/sia_model.pkl", help="Trained model path.")
    parser.add_argument("--output", default="reports/predictions.csv", help="Output CSV path.")
    parser.add_argument("--dossiers", default="reports/dossiers.json", help="Output JSON dossier path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run: python train_pipeline.py --data data/sample_tickets.csv")

    raw = pd.read_csv(args.input)
    model = load_model(model_path)
    audited = predict_with_dossiers(raw, model)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audited.drop(columns=["dossier"]).to_csv(output_path, index=False)

    dossier_path = Path(args.dossiers)
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    flagged = audited[audited["predicted_mismatch"] == 1]["dossier"].tolist()
    with open(dossier_path, "w", encoding="utf-8") as handle:
        json.dump(flagged, handle, indent=2)

    print(f"Wrote predictions to {output_path}")
    print(f"Wrote {len(flagged)} mismatch dossiers to {dossier_path}")


if __name__ == "__main__":
    main()
