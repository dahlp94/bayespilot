from __future__ import annotations

import pandas as pd
import streamlit as st
import arviz as az
import matplotlib.pyplot as plt

import os
import sys

# Add project root (parent of app/) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from bayespilot.profiling.dataset_analyzer import analyze_dataset
from bayespilot.planning.intent_parser import parse_intent
from bayespilot.planning.model_selector import infer_target_type, select_model_type
from bayespilot.planning.model_spec import ModelSpec
from bayespilot.inference.run_inference import run_bayesian_inference, summarize_posterior


st.set_page_config(page_title="BayesPilot", layout="wide")

st.title("BayesPilot")
st.subheader("An AI Copilot for Bayesian Data Analysis")

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("## Dataset Preview")
    st.dataframe(df.head())

    profile = analyze_dataset(df)

    with st.expander("Dataset Profile", expanded=True):
        st.write(f"Rows: {profile.n_rows}")
        st.write(f"Columns: {profile.n_cols}")
        st.write("Numeric columns:", profile.numeric_columns)
        st.write("Categorical columns:", profile.categorical_columns)
        st.write("Missing counts:", profile.missing_counts)

    target = st.selectbox("Select target variable", options=profile.candidate_targets)

    predictor_options = [col for col in df.columns if col != target]
    predictors = st.multiselect("Select predictor variables", options=predictor_options, default=predictor_options[:3])

    question = st.text_area(
        "Ask an analytical question",
        value=f"Which variables most influence {target}?"
    )

    if st.button("Run Bayesian Analysis"):
        if not predictors:
            st.error("Please select at least one predictor.")
        else:
            parsed_intent = parse_intent(question)
            target_type = infer_target_type(df[target])
            model_type = select_model_type(df[target])

            spec = ModelSpec(
                target=target,
                predictors=predictors,
                model_type=model_type,
                target_type=target_type,
                question=question,
            )

            st.write("## Analysis Plan")
            st.json(
                {
                    "target": spec.target,
                    "predictors": spec.predictors,
                    "model_type": spec.model_type,
                    "target_type": spec.target_type,
                    "intent": parsed_intent.intent,
                    "intent_explanation": parsed_intent.explanation,
                }
            )

            with st.spinner("Running MCMC inference..."):
                results = run_bayesian_inference(df, spec, draws=1000, tune=1000)

            idata = results["idata"]
            feature_names = results["feature_names"]

            st.write("## Posterior Summary")
            coef_summary = summarize_posterior(idata, feature_names)
            st.dataframe(coef_summary)

            st.write("## Diagnostics")
            diagnostics_df = az.summary(idata, var_names=["intercept", "beta"], round_to=3)
            st.dataframe(diagnostics_df)

            st.write("## Trace Plot")
            fig = az.plot_trace(idata, var_names=["intercept", "beta"])
            st.pyplot(plt.gcf())
            plt.clf()