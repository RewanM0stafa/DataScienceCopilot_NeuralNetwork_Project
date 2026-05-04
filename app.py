import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer

from data_cleaner import clean_data
from model_selector import select_best_model
from code_generator import generate_training_code

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="Data Science Copilot", layout="wide")
st.title("🤖 Data Science Copilot")
st.caption("Upload a CSV — the system cleans it, picks the best model, and generates ready-to-use code.")

# ============================================================
# Sidebar: data source
# ============================================================
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Choose input",
    ["Upload CSV", "Use Breast Cancer dataset (demo)"]
)

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

else:
    data       = load_breast_cancer()
    df         = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    st.sidebar.success("Breast Cancer dataset loaded (569 rows x 31 columns)")

# ============================================================
# Step 1: Data preview
# ============================================================
if df is not None:
    st.subheader("Step 1 — Data Preview")
    st.dataframe(df.head())
    st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")

    target_col = st.selectbox("Select the target column", df.columns,
                              index=len(df.columns) - 1)

    st.divider()

    # ============================================================
    # Run pipeline
    # ============================================================
    if st.button("🚀 Run Automated Analysis", use_container_width=True):

        # Step 2: Cleaning
        st.subheader("Step 2 — Data Cleaning")
        with st.spinner("Cleaning data..."):
            df_clean = clean_data(df)
        st.success(f"Cleaning done: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")

        st.divider()

        # Step 3 & 4: Feature engineering + model selection
        st.subheader("Steps 3 & 4 — Feature Engineering + Model Selection")
        with st.spinner("Training models..."):
            model_info = select_best_model(df_clean, target_col)

        # Results table
        scores_df = pd.DataFrame(
            model_info['all_scores'].items(),
            columns=["Model", "CV Score"]
        ).sort_values("CV Score", ascending=False)

        st.dataframe(scores_df, use_container_width=True)
        st.success(f"Best model: **{model_info['best_name']}**")
        st.metric("Test Score", f"{model_info['test_score']:.4f}")

        st.divider()

        # Step 5: Code generation
        st.subheader("Step 5 — Generated Code")
        code = generate_training_code(model_info)
        st.code(code, language="python")

        st.download_button(
            label="📥 Download generated code",
            data=code,
            file_name="model_output.py",
            mime="text/x-python",
            use_container_width=True,
        )

else:
    st.info("Choose a data source from the sidebar to get started.")