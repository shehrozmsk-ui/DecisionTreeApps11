from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "decision_tree_model.joblib"
META_PATH = ROOT / "model_metadata.json"

st.set_page_config(
    page_title="Starbucks Nutrition & Decision Tree",
    page_icon="☕",
    layout="wide",
)

# Dark look similar to the reference screenshot
st.markdown("""
<style>
.stApp { background: #0e1117; color: #f5f5f7; }
[data-testid="stHeader"] { background: rgba(14,17,23,0.95); }
h1, h2, h3, p, label { color: #f5f5f7; }
div[data-testid="stMetric"] {
  background: #20212a; border: 1px solid #343540;
  padding: 14px; border-radius: 14px;
}
.stButton > button, .stFormSubmitButton > button {
  background: #e11d48; color: white; border: 0;
  border-radius: 10px; min-height: 3rem; font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  background: #be123c; color: white; border: 0;
}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
  background: #262731; border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("☕ Starbucks Nutrition Explorer")
st.write("Explore menu calories and nutrition, then try the Decision Tree machine-learning prediction.")

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, meta = get_model()

def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def find_calories(columns):
    mapping = {norm(c): c for c in columns}
    for candidate in ["calories", "calorie", "energykcal", "kcal"]:
        if candidate in mapping:
            return mapping[candidate]
    for c in columns:
        if any(x in norm(c) for x in ["calories", "calorie", "energykcal", "kcal"]):
            return c
    return None

st.sidebar.header("Starbucks dataset")
uploaded = st.sidebar.file_uploader("Upload Starbucks CSV", type=["csv"])
local_csvs = sorted(ROOT.glob("*.csv"))
df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")
elif local_csvs:
    try:
        df = pd.read_csv(local_csvs[0])
        st.sidebar.caption(f"Using: {local_csvs[0].name}")
    except Exception as exc:
        st.sidebar.error(f"Could not read CSV: {exc}")

tab1, tab2 = st.tabs(["☕ Menu & Calorie Analysis", "🌳 Decision Tree ML Prediction"])

with tab1:
    st.header("Starbucks menu and calories")
    if df is None:
        st.info("Upload your Starbucks CSV using the sidebar. The menu table and calorie charts will appear here.")
    else:
        cal_col = find_calories(df.columns)
        c1, c2, c3 = st.columns(3)
        c1.metric("Menu rows", f"{len(df):,}")
        c2.metric("Dataset columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        search = st.text_input("Search menu items", placeholder="Type a drink, food, or menu name…")
        shown = df.copy()
        if search.strip():
            mask = shown.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            shown = shown[mask]
        st.caption(f"Showing {len(shown):,} of {len(df):,} rows")
        st.dataframe(shown, use_container_width=True, hide_index=True)

        if cal_col:
            cal_data = pd.to_numeric(df[cal_col], errors="coerce").dropna()
            if not cal_data.empty:
                st.subheader("Calories overview")
                a, b, c = st.columns(3)
                a.metric("Average calories", f"{cal_data.mean():.1f}")
                b.metric("Lowest calories", f"{cal_data.min():.0f}")
                c.metric("Highest calories", f"{cal_data.max():.0f}")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(cal_data, bins=25, kde=True, ax=ax)
                ax.set_xlabel("Calories")
                ax.set_ylabel("Number of menu items")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Highest-calorie menu items")
                numeric_cal = df.copy()
                numeric_cal[cal_col] = pd.to_numeric(numeric_cal[cal_col], errors="coerce")
                st.dataframe(numeric_cal.sort_values(cal_col, ascending=False).head(10),
                             use_container_width=True, hide_index=True)

        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric) >= 2:
            with st.expander("Nutrition correlations and missing values"):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[numeric].corr(), cmap="coolwarm", annot=False, ax=ax)
                ax.set_title("Numeric nutrition correlation")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                report = pd.DataFrame({
                    "Data type": df.dtypes.astype(str),
                    "Missing count": df.isna().sum(),
                    "Missing %": (df.isna().mean() * 100).round(2)
                })
                st.dataframe(report, use_container_width=True)

with tab2:
    st.header("Decision Tree ML Prediction")
    st.write("Enter nutrition values below and press the button to get a model prediction.")

    tree_image = ROOT / "outputs" / "decision_tree.png"
    if tree_image.exists():
        with st.expander("View the actual trained Decision Tree diagram", expanded=True):
            st.caption("Each box is a decision rule. Follow the branches to understand how the model reaches a class.")
            st.image(str(tree_image), caption="Trained Starbucks Decision Tree", use_container_width=True)
    else:
        st.info("The tree diagram will appear here after you run `python train_model.py` and upload the generated `outputs/decision_tree.png` file to GitHub.")
    if model is None or meta is None:
        st.warning(
            "The trained model files are not available yet. First run `python train_model.py` "
            "with your Starbucks CSV, then upload `decision_tree_model.joblib` and "
            "`model_metadata.json` to the same GitHub repository."
        )
    else:
        st.caption(
            f"Prediction target: {meta['target_column']}  •  "
            f"Test accuracy: {meta['accuracy']:.1%}"
        )
        if meta.get("generated_calorie_target"):
            st.info("This model predicts Low / Medium / High calorie categories learned from your dataset.")

        numeric_features = meta.get("numeric_features", [])
        feature_columns = meta["feature_columns"]
        categorical_features = meta.get("categorical_features", [])

        with st.form("prediction_form"):
            st.subheader("Enter nutrition values")
            values = {}
            left, right = st.columns(2)
            for i, feature in enumerate(feature_columns):
                pane = left if i % 2 == 0 else right
                with pane:
                    if feature in numeric_features:
                        default = 0.0
                        if df is not None and feature in df.columns:
                            series = pd.to_numeric(df[feature], errors="coerce").dropna()
                            if not series.empty:
                                default = float(series.median())
                        values[feature] = st.number_input(
                            f"Enter {feature}", value=default, step=1.0, format="%.2f"
                        )
                    else:
                        options = []
                        if df is not None and feature in df.columns:
                            options = df[feature].dropna().astype(str).value_counts().head(100).index.tolist()
                        if options:
                            values[feature] = st.selectbox(f"Choose {feature}", options)
                        else:
                            values[feature] = st.text_input(f"Enter {feature}")
            submitted = st.form_submit_button("Predict calorie category", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([values], columns=feature_columns)
            for feature in numeric_features:
                input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce")
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted calorie category: **{prediction}**")
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(input_df)[0]
                        classifier = model.named_steps.get("classifier", model)
                        classes = classifier.classes_
                        prob_df = pd.DataFrame({
                            "Category": classes.astype(str),
                            "Probability": probs
                        }).sort_values("Probability", ascending=False)
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        st.bar_chart(prob_df.set_index("Category"))
                    except Exception:
                        pass

st.divider()
st.caption("Nutrition results depend on the CSV dataset you provide. ML predictions are educational estimates.")
