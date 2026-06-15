import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
SEVERITY_TO_NUM = {name: idx for idx, name in enumerate(SEVERITY_ORDER)}
NUM_TO_SEVERITY = {idx: name for name, idx in SEVERITY_TO_NUM.items()}

CANONICAL_COLUMNS = {
    "ticket_id": ["Ticket ID", "Ticket Id", "ticket_id", "id"],
    "subject": ["Ticket Subject", "Subject", "ticket_subject"],
    "description": ["Ticket Description", "Description", "ticket_description"],
    "priority": ["Ticket Priority", "Priority", "ticket_priority"],
    "channel": ["Ticket Channel", "Channel", "ticket_channel"],
    "resolution_time": ["Resolution Time", "Resolution Time (hours)", "Resolution Time Hours", "Resolution_Time_Hours", "resolution_time"],
    "ticket_type": ["Ticket Type", "Type", "ticket_type"],
    "product": ["Product Purchased", "Product", "product_purchased"],
    "email": ["Customer Email", "Email", "customer_email"],
}

CRITICAL_PATTERNS = [
    r"\bdown\b",
    r"\boutage\b",
    r"\bcannot access\b",
    r"\bdata loss\b",
    r"\bsecurity\b",
    r"\bbreach\b",
    r"\bproduction\b",
    r"\bpayment(s)? failing\b",
    r"\bblocked\b",
    r"\bescalat(e|ion|ed)\b",
]

HIGH_PATTERNS = [
    r"\berror\b",
    r"\bfail(ed|ing)?\b",
    r"\bunable\b",
    r"\burgent\b",
    r"\bdeadline\b",
    r"\bcrash(ed|ing)?\b",
    r"\btimeout\b",
]

LOW_PATTERNS = [
    r"\bquestion\b",
    r"\bhow do i\b",
    r"\bfeature request\b",
    r"\bcosmetic\b",
    r"\btypo\b",
    r"\bminor\b",
]

NEGATION_PATTERN = re.compile(r"\b(not urgent|no outage|not critical|minor only|works now)\b", re.I)


def _first_existing_column(df, names):
    lower_to_actual = {str(col).strip().lower(): col for col in df.columns}
    for name in names:
        actual = lower_to_actual.get(name.lower())
        if actual is not None:
            return actual
    return None


def load_tickets(path):
    df = pd.read_csv(path)
    return normalize_columns(df)


def normalize_columns(df):
    out = pd.DataFrame(index=df.index)
    for canonical, possible_names in CANONICAL_COLUMNS.items():
        existing = _first_existing_column(df, possible_names)
        if existing is None:
            out[canonical] = ""
        else:
            out[canonical] = df[existing]

    out["ticket_id"] = out["ticket_id"].replace("", np.nan)
    out["ticket_id"] = out["ticket_id"].fillna(pd.Series([f"T-{i + 1:05d}" for i in range(len(out))], index=out.index))
    out["subject"] = out["subject"].fillna("")
    out["description"] = out["description"].fillna("")
    out["channel"] = out["channel"].fillna("Unknown")
    out["ticket_type"] = out["ticket_type"].fillna("Unknown")
    out["product"] = out["product"].fillna("Unknown")
    out["email"] = out["email"].fillna("")
    out["priority"] = out["priority"].fillna("Medium").map(clean_priority).fillna("Medium")
    out["resolution_time"] = out["resolution_time"].apply(parse_resolution_hours)
    out["combined_text"] = (out["subject"].astype(str) + " " + out["description"].astype(str)).str.strip()
    out["domain_tier"] = out["email"].astype(str).apply(domain_tier)
    return out


def clean_priority(value):
    text = str(value).strip().lower()
    for priority in SEVERITY_ORDER:
        if text == priority.lower():
            return priority
    if "crit" in text:
        return "Critical"
    if "high" in text:
        return "High"
    if "med" in text:
        return "Medium"
    if "low" in text:
        return "Low"
    return np.nan


def parse_resolution_hours(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    text = str(value).lower()
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return np.nan
    number = float(numbers[0])
    if "day" in text:
        return number * 24
    if "min" in text:
        return number / 60
    return number


def domain_tier(email):
    text = str(email).lower()
    if any(domain in text for domain in ["enterprise", "corp", "bank", "hospital", "gov"]):
        return "Enterprise"
    if text.endswith("@gmail.com") or text.endswith("@yahoo.com") or text.endswith("@outlook.com"):
        return "Consumer"
    if "@" in text:
        return "Business"
    return "Unknown"


def count_matches(text, patterns):
    return sum(1 for pattern in patterns if re.search(pattern, text, re.I))


def text_severity_score(row):
    text = f"{row.get('subject', '')} {row.get('description', '')}"
    critical = count_matches(text, CRITICAL_PATTERNS)
    high = count_matches(text, HIGH_PATTERNS)
    low = count_matches(text, LOW_PATTERNS)
    score = 1.0 + critical * 0.9 + high * 0.45 - low * 0.35
    if NEGATION_PATTERN.search(text):
        score -= 0.6
    if str(row.get("channel", "")).lower() in {"phone", "chat"} and critical > 0:
        score += 0.2
    return float(np.clip(score, 0, 3))


def resolution_severity_score(hours, quantiles=None):
    if pd.isna(hours):
        return 1.0
    if quantiles is None:
        if hours <= 4:
            return 0.0
        if hours <= 24:
            return 1.0
        if hours <= 72:
            return 2.0
        return 3.0
    q25, q50, q75 = quantiles
    if hours <= q25:
        return 0.0
    if hours <= q50:
        return 1.0
    if hours <= q75:
        return 2.0
    return 3.0


def add_pseudo_labels(df):
    df = df.copy()
    quantiles = df["resolution_time"].dropna().quantile([0.25, 0.5, 0.75]).tolist()
    if len(quantiles) != 3:
        quantiles = None

    df["text_signal_score"] = df.apply(text_severity_score, axis=1)
    df["resolution_signal_score"] = df["resolution_time"].apply(lambda value: resolution_severity_score(value, quantiles))
    df["inferred_score"] = (0.65 * df["text_signal_score"]) + (0.35 * df["resolution_signal_score"])
    df["inferred_severity_num"] = df["inferred_score"].round().clip(0, 3).astype(int)
    df["inferred_severity"] = df["inferred_severity_num"].map(NUM_TO_SEVERITY)
    df["assigned_priority_num"] = df["priority"].map(SEVERITY_TO_NUM).fillna(1).astype(int)
    df["severity_delta"] = df["inferred_severity_num"] - df["assigned_priority_num"]
    df["is_mismatch"] = (df["severity_delta"].abs() >= 1).astype(int)
    df["mismatch_type"] = np.where(
        df["severity_delta"] > 0,
        "Hidden Crisis",
        np.where(df["severity_delta"] < 0, "False Alarm", "Consistent"),
    )
    return df


def build_training_frame(raw_df):
    return add_pseudo_labels(normalize_columns(raw_df))


def feature_columns():
    return ["combined_text", "channel", "ticket_type", "product", "domain_tier", "priority", "resolution_time"]


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1), "combined_text"),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["channel", "ticket_type", "product", "domain_tier", "priority"],
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                ["resolution_time"],
            ),
        ]
    )
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    return Pipeline([("features", preprocessor), ("classifier", classifier)])


def train_model(df, model_dir="models", test_size=0.2):
    labeled = add_pseudo_labels(df)
    X = labeled[feature_columns()]
    y = labeled["is_mismatch"]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "per_class_recall": {
            "consistent": float(recall_score(y_test, predictions, pos_label=0, zero_division=0)),
            "mismatched": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        },
        "classification_report": classification_report(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "pseudo_label_signal_agreement": float(
            np.mean(labeled["text_signal_score"].round().clip(0, 3) == labeled["resolution_signal_score"].round().clip(0, 3))
        ),
        "row_count": int(len(labeled)),
        "mismatch_rate": float(labeled["is_mismatch"].mean()),
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "sia_model.pkl", "wb") as handle:
        pickle.dump(model, handle)
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return model, metrics, labeled


def load_model(model_path):
    with open(model_path, "rb") as handle:
        return pickle.load(handle)


def predict_with_dossiers(df, model):
    prepared = add_pseudo_labels(normalize_columns(df))
    probabilities = model.predict_proba(prepared[feature_columns()])[:, 1]
    predictions = model.predict(prepared[feature_columns()])
    prepared["predicted_mismatch"] = predictions.astype(int)
    prepared["model_confidence"] = np.where(predictions == 1, probabilities, 1 - probabilities)
    prepared["dossier"] = prepared.apply(make_dossier, axis=1)
    return prepared


def make_dossier(row):
    text = f"{row.get('subject', '')} {row.get('description', '')}"
    found_keywords = []
    for pattern in CRITICAL_PATTERNS + HIGH_PATTERNS + LOW_PATTERNS:
        match = re.search(pattern, text, re.I)
        if match:
            found_keywords.append(match.group(0))
    keyword_value = ", ".join(dict.fromkeys(found_keywords[:5])) or "No strong severity keyword found"

    evidence = [
        {
            "signal": "keyword",
            "value": keyword_value,
            "weight": f"text_signal_score={row['text_signal_score']:.2f}",
            "source_field": "Ticket Subject + Ticket Description",
        },
        {
            "signal": "resolution_time",
            "value": "" if pd.isna(row["resolution_time"]) else f"{row['resolution_time']:.2f} hours",
            "interpretation": f"resolution_signal_score={row['resolution_signal_score']:.2f}",
            "source_field": "Resolution Time",
        },
        {
            "signal": "metadata",
            "value": f"channel={row.get('channel', 'Unknown')}; type={row.get('ticket_type', 'Unknown')}",
            "weight": "used as classifier metadata",
            "source_field": "Ticket Channel + Ticket Type",
        },
    ]

    if row["severity_delta"] > 0:
        explanation = (
            "The inferred severity is higher than the assigned priority. "
            "The ticket should be reviewed because the text or resolution-time signals indicate stronger impact than the label."
        )
    elif row["severity_delta"] < 0:
        explanation = (
            "The inferred severity is lower than the assigned priority. "
            "The ticket may be over-prioritized because the grounded signals do not support the assigned urgency."
        )
    else:
        explanation = (
            "The inferred severity matches the assigned priority. "
            "No priority mismatch is indicated by the available ticket fields."
        )

    return {
        "ticket_id": str(row["ticket_id"]),
        "assigned_priority": str(row["priority"]),
        "inferred_severity": str(row["inferred_severity"]),
        "mismatch_type": str(row["mismatch_type"]),
        "severity_delta": int(row["severity_delta"]),
        "feature_evidence": evidence,
        "constraint_analysis": explanation,
        "confidence": f"{row.get('model_confidence', 0.0):.2f}",
    }
