import streamlit as st
import pandas as pd
from analysis import analyze_and_decide

st.set_page_config(page_title="🤖 AutoML Decision Agent", layout="wide")

st.title("🤖 Smart AutoML Decision Agent")
st.write("Upload your dataset and let the agent automatically choose and train the best model!")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("🎯 Select Target Column", df.columns)
        target_type = st.radio(
            "Select Target Type",
            ("Categorical (Classification)", "Numeric (Regression)")
        )

        if st.button("🚀 Run Agent"):
            with st.spinner("Analyzing data and training the best model..."):
                decision, report = analyze_and_decide(df, target_col, target_type)

            st.success("✅ Model Decision Completed!")

            st.subheader("🤖 Model Decision Summary")
            st.write(f"**Model Selected:** {decision.get('model_name', 'N/A')}")
            st.write(f"**Reason for Selection:** {decision.get('reason', 'N/A')}")

            st.markdown("---")

            st.subheader("📈 Model Performance Metrics")
            if "metrics" in report:
                st.json(report["metrics"])

            if "sample_predictions" in report:
                st.subheader("🔍 Sample Predictions")
                st.dataframe(report["sample_predictions"])

            if "nn_summary" in report and report["nn_summary"]:
                st.subheader("🧠 Neural Network Summary")
                st.text(report["nn_summary"])

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file appears to be empty. Please upload a valid CSV.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
