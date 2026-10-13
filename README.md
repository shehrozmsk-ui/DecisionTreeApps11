# Starbucks Menu Nutrition: EDA + Decision Tree + Streamlit

This project analyzes a Starbucks drink nutrition CSV, creates exploratory charts, trains a Decision Tree classifier, and provides a Streamlit dashboard.

## Project files

```text
starbucks_menu_nutrition_project/
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── your_starbucks_dataset.csv       # Add your dataset here if appropriate
├── decision_tree_model.joblib       # Generated after training
├── model_metadata.json              # Generated after training
└── outputs/                         # Generated EDA charts and reports
```

## 1. Add your dataset

Place your Starbucks CSV in the same folder as `train_model.py`. If there are multiple CSV files in the folder, the script uses the first one alphabetically. Alternatively, set `CSV_PATH` in `train_model.py` to the exact filename.

If you want to download through KaggleHub instead, set `KAGGLE_DATASET = "owner/dataset-slug"` and make sure KaggleHub authentication and internet access are configured.

## 2. Target selection

By default, the script looks for a Calories column and creates a three-level classification target named `Calorie_Level` (Low, Medium, High). It excludes the original calorie column from the model features to prevent target leakage.

If your CSV already has a suitable classification target, set `TARGET_COLUMN` to its exact name. For example, you could predict a beverage category if it is a label in your data. The target must have at least two classes.

If you need to predict exact calorie numbers rather than calorie groups, a Decision Tree **regressor** is more appropriate than a classifier and the training code must be changed accordingly.

## 3. Install and train

From a terminal in this project folder:

```bash
pip install -r requirements.txt
python train_model.py
```

The script prints EDA information and saves charts/tables/reports in `outputs/`. It also creates `decision_tree_model.joblib` and `model_metadata.json`.

## 4. Run Streamlit locally

```bash
streamlit run app.py
```

The app has a CSV upload option for EDA, nutrition charts, and a prediction form. The model files must exist for predictions.

## 5. Upload to GitHub

Create a repository such as `starbucks-menu-nutrition`. Upload:

- `train_model.py`
- `app.py`
- `requirements.txt`
- `README.md`
- `.gitignore`
- `decision_tree_model.joblib`
- `model_metadata.json`

You may include the CSV if its license permits redistribution and it is not too large. If the dataset is private, licensed against redistribution, or large, do not commit it. You can still use the local CSV to train and include only the model files and application code in GitHub.

Never upload `kaggle.json`, API tokens, passwords, or other credentials.

## 6. Deploy on Streamlit Community Cloud

1. Push your project to GitHub.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app**.
4. Select your repository, branch, and `app.py` as the app file.
5. Click **Deploy**.

Streamlit runs `app.py`; it does not automatically run `train_model.py`. Train the model before deployment and commit the generated model and metadata files. The app will update when you push code changes to GitHub.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy


## Decision Tree diagram
After running `python train_model.py`, the project creates `outputs/decision_tree.png`, a visual diagram of the fitted Decision Tree. Upload the generated `outputs/decision_tree.png` file (and model files) to GitHub so the deployed Streamlit app can display it.
