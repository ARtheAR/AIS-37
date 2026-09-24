import streamlit as st
import json


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Midwest Pest Risk Dashboard",
    page_icon="🌾",
    layout="wide"
)


# --------------------------------------------------
# Load risk scores
# --------------------------------------------------

with open("risk_scores.json", "r") as f:
    risk_data = json.load(f)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🌾 Midwest Pest Risk Dashboard")
st.write("Current pest risk assessment and treatment proposals.")


# --------------------------------------------------
# Risk score descriptions
# --------------------------------------------------

risk_labels = {
    1: "No Imminent Risk",
    2: "Minimal Risk",
    3: "Moderate Risk",
    4: "High Risk",
    5: "Severe Risk"
}


# --------------------------------------------------
# Display pests
# --------------------------------------------------

st.subheader("Pest Risk")

cols = st.columns(len(risk_data))

for col, (pest, data) in zip(cols, risk_data.items()):

    score = data["score"]
    proposal = data["proposal"]

    with col:

        st.metric(
            label=pest,
            value=f"{score}/5"
        )

        st.write(
            f"**{risk_labels.get(score, 'Unknown Risk')}**"
        )

        st.info(proposal)

