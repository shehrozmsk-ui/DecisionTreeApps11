"""
Starbucks Menu Nutrition: EDA + Decision Tree classification.

SETUP:
1. Put your Starbucks CSV in this folder (recommended), OR configure KaggleHub below.
2. Run: python train_model.py
3. The script creates decision_tree_model.joblib and model_metadata.json.
4. Run: streamlit run app.py

Target logic:
- If TARGET_COLUMN is set to an existing column, that column is predicted.
- Otherwise, if a Calories column is found, the script creates Calorie_Level
  (Low / Medium / High) from Calories and excludes Calories from model inputs
  to prevent target leakage.
- This is a classification template. For a continuous numeric prediction,
  use DecisionTreeRegressor instead.
"""

from pathlib import Path
import json
import warnings
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ======================== EDIT IF NEEDED ========================
# If the CSV is in this project folder, leave CSV_PATH blank for auto-detection.
CSV_PATH = ""
# Optional Kaggle identifier, such as "owner/dataset-slug".
# Used only when no local CSV is found.
KAGGLE_DATASET = ""
# Set to the exact target column if your dataset already has a useful target.
# Leave blank to automatically create Calorie_Level from a Calories column.
TARGET_COLUMN = ""
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_TREE_DEPTH = 5
# ===============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):
    lookup = {clean_key(c): c for c in columns}
    for candidate in candidates:
        if clean_key(candidate) in lookup:
            return lookup[clean_key(candidate)]
    # Accept columns containing the candidate term, e.g. "Total Calories".
    for c in columns:
        key = clean_key(c)
        for candidate in candidates:
            if clean_key(candidate) in key:
                return c
    return None


def load_data():
    if CSV_PATH.strip():
        path = ROOT / CSV_PATH
        if not path.exists():
            raise FileNotFoundError(f"CSV_PATH was set but file was not found: {path}")
        print(f"Loading local CSV: {path.name}")
        return pd.read_csv(path)

    local_csvs = sorted(
        p for p in ROOT.glob("*.csv")
        if p.name.lower() not in {"sample.csv"}
    )
    if local_csvs:
        print("CSV files found in project folder:")
        for i, path in enumerate(local_csvs):
            print(f"  [{i}] {path.name}")
        selected = local_csvs[0]
        print(f"Using: {selected.name}")
        return pd.read_csv(selected)

    if KAGGLE_DATASET.strip():
        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError("Install kagglehub: pip install kagglehub") from exc
        folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csvs = sorted(folder.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in downloaded folder: {folder}")
        print("Downloaded CSV files:")
        for i, p in enumerate(csvs):
            print(f"  [{i}] {p.name}")
        return pd.read_csv(csvs[0])

    raise FileNotFoundError(
        "No CSV file found. Put your Starbucks nutrition CSV in the same folder "
        "as train_model.py, or set CSV_PATH / KAGGLE_DATASET near the top."
    )


def save_eda(df):
    print("\n" + "=" * 60 + "\nSTARBUCKS MENU NUTRITION — EDA\n" + "=" * 60)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\nFirst 5 rows:\n", df.head())
    print("\nColumn names:\n", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nDataFrame info:")
    df.info()
    print("\nDescriptive statistics:\n", df.describe(include="all").T)

    missing = pd.DataFrame({
        "Missing_Count": df.isna().sum(),
        "Missing_Percent": (df.isna().mean() * 100).round(2)
    })
    print("\nMissing values:\n", missing[missing["Missing_Count"] > 0]
          if missing["Missing_Count"].any() else "No missing values found.")
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    missing.to_csv(OUTPUT_DIR / "missing_values.csv")
    df.describe(include="all").T.to_csv(OUTPUT_DIR / "descriptive_statistics.csv")

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric:
        df[numeric].hist(figsize=(15, 10), bins=25, edgecolor="black")
        plt.suptitle("Starbucks Nutrition: Numeric Distributions")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "numeric_distributions.png", dpi=150)
        plt.close()

        for col in numeric[:8]:
            plt.figure(figsize=(8, 3))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot: {col}")
            plt.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(col))
            plt.savefig(OUTPUT_DIR / f"boxplot_{safe}.png", dpi=150)
            plt.close()

    if len(numeric) >= 2:
        plt.figure(figsize=(12, 8))
        sns.heatmap(df[numeric].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Nutrition Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
        plt.close()

    for col in categorical[:5]:
        counts = df[col].astype(str).value_counts().head(12)
        if not counts.empty:
            plt.figure(figsize=(10, 4))
            counts.plot(kind="bar")
            plt.title(f"Top values: {col}")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(col))
            plt.savefig(OUTPUT_DIR / f"category_{safe}.png", dpi=150)
            plt.close()

    calorie_col = find_column(df.columns, ["Calories", "Calorie", "Energy (kcal)", "kcal"])
    if calorie_col and pd.api.types.is_numeric_dtype(df[calorie_col]):
        plt.figure(figsize=(9, 4))
        sns.histplot(df[calorie_col].dropna(), bins=25, kde=True)
        plt.title(f"Starbucks Menu Calorie Distribution ({calorie_col})")
        plt.xlabel("Calories")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "calorie_distribution.png", dpi=150)
        plt.close()

    print(f"\nEDA reports and charts saved to: {OUTPUT_DIR}")


def train_model(original_df):
    df = original_df.copy()
    target = TARGET_COLUMN.strip() if TARGET_COLUMN.strip() else None

    if target and target not in df.columns:
        raise ValueError(
            f"TARGET_COLUMN '{target}' was not found. Available columns: {df.columns.tolist()}"
        )

    calorie_col = find_column(df.columns, ["Calories", "Calorie", "Energy (kcal)", "kcal"])
    generated_target = False

    if target is None:
        if calorie_col is None:
            raise ValueError(
                "No TARGET_COLUMN was configured and no Calories column was detected. "
                "Set TARGET_COLUMN to an existing classification label, or adjust the "
                "calorie column detection in train_model.py."
            )
        df[calorie_col] = pd.to_numeric(df[calorie_col], errors="coerce")
        valid_calories = df[calorie_col].dropna()
        if valid_calories.nunique() < 3:
            raise ValueError("The detected calorie column needs at least 3 distinct numeric values.")
        # Use tertiles to create low/medium/high classes, robust to dataset scale.
        # Rank first so repeated calorie values do not make qcut fail when bin edges tie.
        df["Calorie_Level"] = pd.qcut(
            df[calorie_col].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"]
        )
        target = "Calorie_Level"
        generated_target = True
        print(f"\nCreated target '{target}' from '{calorie_col}' using calorie tertiles.")

    data = df.dropna(subset=[target]).drop_duplicates().copy()
    y = data[target].astype(str)
    X = data.drop(columns=[target]).copy()

    # Avoid direct target leakage for the generated calorie-level task.
    # If target is calorie-derived, raw calories must not be a feature.
    if generated_target and calorie_col in X.columns:
        X = X.drop(columns=[calorie_col])
        print(f"Excluded '{calorie_col}' from model features to prevent target leakage.")

    # Also remove identifiers / near-unique name fields that can overfit.
    identifier_cols = [
        c for c in X.columns
        if any(term in clean_key(c) for term in ["id", "productname", "beveragename", "itemname"])
        and X[c].nunique(dropna=True) > max(20, int(0.5 * len(X)))
    ]
    if identifier_cols:
        print("Dropping high-cardinality identifier/name columns:", identifier_cols)
        X = X.drop(columns=identifier_cols)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
        print("Dropped constant columns:", constant_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=MAX_TREE_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print("\n" + "=" * 60 + "\nDECISION TREE EVALUATION\n" + "=" * 60)
    print(f"Target: {target}")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

    labels = list(pipeline.named_steps["classifier"].classes_)
    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Starbucks Decision Tree — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save a readable visual of the actual trained Decision Tree.
    try:
        from sklearn.tree import plot_tree
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        fitted_tree = pipeline.named_steps["classifier"]
        try:
            tree_feature_names = fitted_preprocessor.get_feature_names_out()
        except Exception:
            tree_feature_names = [f"feature_{i}" for i in range(fitted_tree.n_features_in_)]
        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(
            fitted_tree,
            feature_names=[str(x) for x in tree_feature_names],
            class_names=[str(x) for x in fitted_tree.classes_],
            filled=True,
            rounded=True,
            proportion=False,
            impurity=False,
            fontsize=7,
            ax=ax
        )
        ax.set_title("Starbucks Nutrition — Trained Decision Tree")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "decision_tree.png", dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved decision tree diagram: {OUTPUT_DIR / 'decision_tree.png'}")
    except Exception as exc:
        print(f"Could not create decision tree diagram: {exc}")

    joblib.dump(pipeline, ROOT / "decision_tree_model.joblib")
    metadata = {
        "target_column": str(target),
        "generated_calorie_target": bool(generated_target),
        "calorie_column": str(calorie_col) if calorie_col else None,
        "feature_columns": [str(c) for c in X.columns],
        "numeric_features": [str(c) for c in numeric_features],
        "categorical_features": [str(c) for c in categorical_features],
        "class_labels": [str(c) for c in labels],
        "accuracy": float(accuracy),
        "rows_used": int(len(data)),
        "feature_count": int(X.shape[1]),
        "max_depth": MAX_TREE_DEPTH
    }
    with open(ROOT / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    test_results = X_test.copy()
    test_results["actual"] = y_test.values
    test_results["predicted"] = pred
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Target: {target}\nAccuracy: {accuracy:.4f}\n\n")
        f.write(classification_report(y_test, pred, zero_division=0))

    print(f"\nSaved model: {ROOT / 'decision_tree_model.joblib'}")
    print(f"Saved metadata: {ROOT / 'model_metadata.json'}")
    print(f"EDA/model outputs: {OUTPUT_DIR}")


def main():
    df = load_data()
    if df.empty:
        raise ValueError("The dataset is empty.")
    save_eda(df)
    train_model(df)
    print("\nSUCCESS: Starbucks EDA + Decision Tree completed.")


if __name__ == "__main__":
    main()
