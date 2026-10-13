
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

# Styling
st.markdown("""
<style>
.stApp {
    background: #0e1117;
    color: #f5f5f7;
}
[data-testid="stHeader"] {
    background: rgba(14,17,23,0.95);
}
h1, h2, h3, p, label {
    color: #f5f5f7;
}
div[data-testid="stMetric"] {
    background: #20212a;
    border: 1px solid #343540;
    padding: 14px;
    border-radius: 14px;
}
.stButton > button, .stFormSubmitButton > button {
    background: #e11d48;
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 3rem;
    font-weight: 700;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #be123c;
    color: white;
    border: 0;
}
</style>
""", unsafe_allow_html=True)

st.title("☕ Starbucks Nutrition Explorer")
st.write(
    "Explore Starbucks menu nutrition, analyze calories, "
    "and try a Decision Tree machine-learning prediction."
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None

    try:
        model = joblib.load(MODEL_PATH)

        with open(META_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        return model, metadata

    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
        return None, None


model, meta = get_model()


def norm(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_calories(columns):
    mapping = {norm(column): column for column in columns}

    for candidate in ["calories", "calorie", "energykcal", "kcal"]:
        if candidate in mapping:
            return mapping[candidate]

    for column in columns:
        if any(
            term in norm(column)
            for term in ["calories", "calorie", "energykcal", "kcal"]
        ):
            return column

    return None


# Load the dataset
st.sidebar.header("Starbucks Dataset")
uploaded = st.sidebar.file_uploader(
    "Upload Starbucks CSV",
    type=["csv"]
)

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


# Application tabs
tab1, tab2 = st.tabs([
    "☕ Menu & Calorie Analysis",
    "🌳 Decision Tree ML Prediction"
])


# =========================================================
# TAB 1: EXPLORATORY DATA ANALYSIS
# =========================================================
with tab1:
    st.header("Starbucks Menu and Calories")

    if df is None:
        st.info(
            "Upload your Starbucks CSV using the sidebar "
            "to display the menu and nutrition charts."
        )

    else:
        cal_col = find_calories(df.columns)

        col1, col2, col3 = st.columns(3)

        col1.metric("Menu Rows", f"{len(df):,}")
        col2.metric("Dataset Columns", f"{len(df.columns):,}")
        col3.metric(
            "Missing Cells",
            f"{int(df.isna().sum().sum()):,}"
        )

        st.subheader("Menu Dataset")

        search = st.text_input(
            "Search menu items",
            placeholder="Enter a drink, food, or menu name..."
        )

        shown = df.copy()

        if search.strip():
            mask = shown.astype(str).apply(
                lambda column: column.str.contains(
                    search,
                    case=False,
                    na=False,
                    regex=False
                )
            ).any(axis=1)

            shown = shown[mask]

        st.caption(f"Showing {len(shown):,} of {len(df):,} rows")

        st.dataframe(
            shown,
            use_container_width=True,
            hide_index=True
        )

        # Calorie analysis
        if cal_col:
            cal_data = pd.to_numeric(
                df[cal_col],
                errors="coerce"
            ).dropna()

            if not cal_data.empty:
                st.subheader("Calories Overview")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Average Calories",
                    f"{cal_data.mean():.1f}"
                )
                col2.metric(
                    "Lowest Calories",
                    f"{cal_data.min():.0f}"
                )
                col3.metric(
                    "Highest Calories",
                    f"{cal_data.max():.0f}"
                )

                st.subheader("Calorie Distribution")

                fig, ax = plt.subplots(figsize=(10, 4))

                sns.histplot(
                    cal_data,
                    bins=25,
                    kde=True,
                    ax=ax
                )

                ax.set_xlabel("Calories")
                ax.set_ylabel("Number of Menu Items")
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Highest-Calorie Menu Items")

                numeric_cal = df.copy()
                numeric_cal[cal_col] = pd.to_numeric(
                    numeric_cal[cal_col],
                    errors="coerce"
                )

                st.dataframe(
                    numeric_cal
                    .sort_values(cal_col, ascending=False)
                    .head(10),
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.warning(
                "No calorie column was detected in this CSV."
            )

        # Correlation and missing-value analysis
        numeric = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        with st.expander("Nutrition Correlations and Missing Values"):
            report = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Count": df.isna().sum(),
                "Missing Percentage": (
                    df.isna().mean() * 100
                ).round(2)
            })

            st.subheader("Missing-Value Report")
            st.dataframe(
                report,
                use_container_width=True
            )

            if len(numeric) >= 2:
                st.subheader("Numeric Feature Correlation")

                fig, ax = plt.subplots(figsize=(10, 6))

                sns.heatmap(
                    df[numeric].corr(),
                    cmap="coolwarm",
                    annot=False,
                    ax=ax
                )

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info(
                    "At least two numeric columns are required "
                    "to display a correlation heatmap."
                )


# =========================================================
# TAB 2: DECISION TREE MACHINE-LEARNING PREDICTION
# =========================================================
with tab2:
    st.header("Decision Tree ML Prediction")

    st.write(
        "Enter nutrition values and submit the form "
        "to get a prediction from your trained model."
    )

    tree_image = ROOT / "outputs" / "decision_tree.png"

    if tree_image.exists():
        with st.expander(
            "View the Trained Decision Tree",
            expanded=True
        ):
            st.caption(
                "The diagram displays the decision rules "
                "used by the trained tree."
            )

            st.image(
                str(tree_image),
                caption="Trained Starbucks Decision Tree",
                use_container_width=True
            )

    else:
        st.info(
            "Decision Tree image not found. Run train_model.py "
            "and upload outputs/decision_tree.png if generated."
        )

    if model is None or meta is None:
        st.warning(
            "Trained model files are missing or could not be loaded. "
            "Run train_model.py with your Starbucks CSV, then upload "
            "decision_tree_model.joblib and model_metadata.json "
            "to the same GitHub repository as app.py."
        )

    else:
        target_column = meta.get("target_column", "Not specified")
        accuracy = meta.get("accuracy")

        if accuracy is not None:
            st.caption(
                f"Prediction Target: {target_column} | "
                f"Test Accuracy: {accuracy:.1%}"
            )
        else:
            st.caption(f"Prediction Target: {target_column}")

        if meta.get("generated_calorie_target"):
            st.info(
                "This model predicts calorie categories "
                "based on the training dataset."
            )

        numeric_features = meta.get("numeric_features", [])
        feature_columns = meta.get("feature_columns", [])

        if not feature_columns:
            st.error(
                "No feature_columns were found in model_metadata.json. "
                "Check the metadata generated by train_model.py."
            )

        else:
            with st.form("prediction_form"):
                st.subheader("Enter Nutrition Values")

                values = {}
                left, right = st.columns(2)

                for i, feature in enumerate(feature_columns):
                    pane = left if i % 2 == 0 else right

                    with pane:
                        if feature in numeric_features:
                            default = 0.0

                            if df is not None and feature in df.columns:
                                series = pd.to_numeric(
                                    df[feature],
                                    errors="coerce"
                                ).dropna()

                                if not series.empty:
                                    default = float(series.median())

                            values[feature] = st.number_input(
                                f"Enter {feature}",
                                value=default,
                                step=1.0,
                                format="%.2f"
                            )

                        else:
                            options = []

                            if df is not None and feature in df.columns:
                                options = (
                                    df[feature]
                                    .dropna()
                                    .astype(str)
                                    .value_counts()
                                    .head(100)
                                    .index.tolist()
                                )

                            if options:
                                values[feature] = st.selectbox(
                                    f"Choose {feature}",
                                    options
                                )
                            else:
                                values[feature] = st.text_input(
                                    f"Enter {feature}"
                                )

                submitted = st.form_submit_button(
                    "Predict Calorie Category",
                    use_container_width=True
                )

            # FIXED: outer try has a matching except block
            if submitted:
                input_df = pd.DataFrame(
                    [values],
                    columns=feature_columns
                )

                for feature in numeric_features:
                    input_df[feature] = pd.to_numeric(
                        input_df[feature],
                        errors="coerce"
                    )

                try:
                    prediction = model.predict(input_df)[0]

                    st.success(
                        f"Predicted category: {prediction}"
                    )

                    if hasattr(model, "predict_proba"):
                        try:
                            probs = model.predict_proba(input_df)[0]

                            if hasattr(model, "named_steps"):
                                classifier = model.named_steps.get(
                                    "classifier"
                                )
                            else:
                                classifier = model

                            if classifier is not None and hasattr(
                                classifier, "classes_"
                            ):
                                classes = classifier.classes_

                                prob_df = pd.DataFrame({
                                    "Category": classes.astype(str),
                                    "Probability": probs
                                }).sort_values(
                                    "Probability",
                                    ascending=False
                                )

                                st.subheader("Prediction Probabilities")

                                st.dataframe(
                                    prob_df,
                                    use_container_width=True,
                                    hide_index=True
                                )

                                st.bar_chart(
                                    prob_df.set_index("Category")
                                )

                        except Exception as exc:
                            st.warning(
                                "Prediction succeeded, but probabilities "
                                f"could not be displayed: {exc}"
                            )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")


# Footer
st.divider()

st.caption(
    "Nutrition results depend on the CSV dataset you provide. "
    "Machine-learning predictions are educational estimates."
)
