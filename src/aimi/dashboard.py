from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from aimi.generator import GeneratorConfig, SyntheticBatchGenerator, parse_profile
from aimi.service import ServiceContainer

st.set_page_config(page_title="Manufacturing Intelligence", layout="wide")
st.title("AI-Driven Manufacturing Intelligence")

svc = ServiceContainer()

source = st.radio("Data source", ["Synthetic", "Upload CSV"])
if source == "Synthetic":
    batches = st.slider("Number of batches", 50, 500, 150, 10)
    bdf, pdf = SyntheticBatchGenerator(GeneratorConfig(n_batches=batches)).generate()
else:
    uploaded = st.file_uploader("Upload batch CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    bdf = pd.read_csv(io.BytesIO(uploaded.getvalue()))
    pdf = svc.profile_df

row = svc.train_df.sample(1, random_state=42).iloc[0].to_dict()
pred = svc.predict(row)
opt = svc.optimize(40)
profile = parse_profile(pdf.iloc[0]["energy_profile"])
anomaly_score = float(svc.epi.score([profile])[0])

left, right = st.columns(2)
with left:
    st.subheader("Prediction")
    st.json(pred)
    st.metric("Anomaly score", f"{anomaly_score:.3f}")
with right:
    st.subheader("Adaptive carbon target")
    st.metric("Target (kg CO2e)", f"{opt['carbon_target']:.2f}")

st.subheader("Energy profile")
st.plotly_chart(px.line(x=list(range(len(profile))), y=profile, labels={"x": "t", "y": "kW"}), use_container_width=True)

st.subheader("Pareto candidates")
pareto_df = pd.DataFrame(opt["pareto_candidates"])
st.dataframe(pareto_df)

st.subheader("Golden signature")
latest = svc.store.latest()
if latest:
    st.write(latest)
else:
    st.info("No golden signature stored yet. Use API POST /golden-signature to add one.")
