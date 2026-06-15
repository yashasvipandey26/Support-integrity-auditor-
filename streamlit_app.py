import json
from pathlib import Path

import pandas as pd
import streamlit as st

from sia_core import load_model, normalize_columns, predict_with_dossiers, train_model

st.set_page_config(page_title="Support Integrity Auditor", layout="wide")
st.title("Support Integrity Auditor")

MODEL_PATH = Path("models/sia_model.pkl")
# Fixed: Point to your verified repository file path
SAMPLE_PATH = Path("data/adversarial_tickets.csv")

@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    
    # Fallback to keep app running if model weights are missing
    if SAMPLE_PATH.exists():
        sample = pd.read_csv(SAMPLE_PATH)
        model, _, _ = train_model(normalize_columns(sample), "models")
        return model
    else:
        st.error(f"Critical Error: {SAMPLE_PATH} not found. Please upload a CSV batch.")
        st.stop()

def audit_dataframe(df):
    model = get_model()
    return predict_with_dossiers(df, model)

tab_single, tab_batch, tab_dashboard = st.tabs(["Single Ticket", "Batch CSV", "Dashboard"])

with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Ticket Subject", "Payments failing for checkout customers")
        description = st.text_area(
            "Ticket Description",
            "Production checkout is down and customers cannot complete payments.",
            height=120,
        )
        priority = st.selectbox("Ticket Priority", ["Low", "Medium", "High", "Critical"], index=0)
    with col2:
        channel = st.selectbox("Ticket Channel", ["email", "chat", "phone", "social media"], index=1)
        ticket_type = st.text_input("Ticket Type", "Billing")
        product = st.text_input("Product Purchased", "Enterprise CRM")
        email = st.text_input("Customer Email", "ops@enterprise-corp.com")
        resolution_time = st.number_input("Resolution Time", min_value=0.0, value=36.0, step=1.0)

    if st.button("Audit Ticket", type="primary"):
        single = pd.DataFrame(
            [
                {
                    "Ticket Subject": subject,
                    "Ticket Description": description,
                    "Ticket Priority": priority,
                    "Ticket Channel": channel,
                    "Ticket Type": ticket_type,
                    "Product Purchased": product,
                    "Customer Email": email,
                    "Resolution Time": resolution_time,
                }
            ]
        )
        result = audit_dataframe(single)
        dossier = result.iloc[0]["dossier"]
        verdict = "Mismatch" if result.iloc[0]["predicted_mismatch"] == 1 else "Consistent"
        st.metric("Binary Judgment", verdict)
        st.json(dossier)

with tab_batch:
    uploaded = st.file_uploader("Upload ticket CSV", type=["csv"])
    if uploaded:
        batch = pd.read_csv(uploaded)
    else:
        batch = pd.read_csv(SAMPLE_PATH)
        st.caption("Using sample tickets until a CSV is uploaded.")

    audited = audit_dataframe(batch)
    st.dataframe(
        audited[
            [
                "ticket_id",
                "priority",
                "inferred_severity",
                "severity_delta",
                "mismatch_type",
                "predicted_mismatch",
                "model_confidence",
            ]
        ],
        use_container_width=True,
    )
    st.download_button(
        "Download Dossiers JSON",
        json.dumps(audited[audited["predicted_mismatch"] == 1]["dossier"].tolist(), indent=2),
        file_name="sia_dossiers.json",
        mime="application/json",
    )

with tab_dashboard:
    # Safely building default dashboard view metrics
    df = audit_dataframe(pd.read_csv(SAMPLE_PATH))
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets Audited", len(df))
    c2.metric("Flagged Mismatches", int(df["predicted_mismatch"].sum()))
    c3.metric("Mismatch Rate", f"{df['predicted_mismatch'].mean() * 100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Mismatch Types")
        if "mismatch_type" in df.columns and not df["mismatch_type"].isna().all():
            st.bar_chart(df["mismatch_type"].value_counts())
        else:
            st.info("No active mismatches loaded to segment by distribution type.")
    with right:
        st.subheader("Top Signal Scores")
        # Checking columns dynamically depending on source labels
        available_signals = [col for col in ["text_signal_score", "resolution_signal_score", "severity_delta"] if col in df.columns]
        st.bar_chart(df[available_signals])

    st.subheader("Severity Delta Heatmap")
    heatmap_data = df.pivot_table(
        index="ticket_type",
        columns="channel",
        values="severity_delta",
        aggfunc="mean",
        fill_value=0,
    )
    st.dataframe(heatmap_data.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)
