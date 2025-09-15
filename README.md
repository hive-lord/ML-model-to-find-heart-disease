# ML-model-to-find-heart-disease
This is a logistic regression model used to find if people have heart disease or not, refer readme for more info
## 📖 Table of Contents
- [Project Overview]
- [Setup youtube link]
- [why I chose this dataset]
- [Dataset Description]
- [Technologies Used]
- [Project Structure]
- [Setup]
- [Notebook Walkthrough]
- [Model Performance]
---

## 📝 Project Overview

[cite_start]The primary objective of this project is to build a robust classifier capable of predicting whether a patient has heart disease based on 13 medical attributes  The process involves:
1.   **Data Cleaning & Preprocessing**: Loading the dataset and handling missing or inconsistent values found within the data 
2.  **Exploratory Data Analysis (EDA)**: Visualizing the data to understand feature distributions, correlations, and key patterns.
3.  **Model Training**: Implementing and training both a Logistic Regression model and a Random Forest Classifier on the preprocessed data.
4.  **Model Evaluation**: Using key classification metrics (Accuracy, Precision, Recall, F1-Score) to assess and compare the performance of the models on a held-out test set.

## heres youtube unlisted video 

---
## why i chose this dataset
there are a lot of people in india, my country who suffer from heart disease and many doctors are not skilled enough to find people have actually heart disease or the symptom is side effect of some other issue. Hence this model finds if the really have an underlying heart condition and what might be its cause.

---

## 📊 Dataset Description

[cite_start]This project utilizes the "Heart Disease Cleveland" dataset, provided in the `heart-disease-cleveland.csv` file   The dataset consists of 14 attributes collected from patients 

| Feature    | Description                                                                                                                              |
| :--------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| `age`      | Age of the patient in years                                                                                                           |
| `sex`      | Sex of the patient (1 = male; 0 = female)                                                                                            |
| `cp`       | Chest pain type (Values: 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)                     |
| `trestbps` | Resting blood pressure in mm Hg                                                                                                    |
| `chol`     | Serum cholesterol in mg/dl                                                                                                          |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                                                                              |
| `restecg`  | Resting electrocardiographic results (Values: 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)                   |
| `thalach`  | Maximum heart rate achieved                                                                                                      |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no)                                                                                              |
| `oldpeak`  | ST depression induced by exercise relative to rest                                                                                     |
| `slope`    | The slope of the peak exercise ST segment (Values: 1: upsloping, 2: flat, 3: downsloping)                                                       |
| `ca`       | Number of major vessels (0-3) colored by fluoroscopy                                                                                   |
| `thal`     | Thallium stress test result (Values: 3 = normal; 6 = fixed defect; 7 = reversible defect)                                                       |
| `diagnosis`| **Target Variable**: Diagnosis of heart disease.  A value of 0 indicates no presence, while values 1-4 indicate the presence of heart disease  |

---

## 💻 Technologies Used
- **Python 3.x**
- **Jupyter Notebook**
- **Libraries:**
    - `Pandas` (for data manipulation)
    - `NumPy` (for numerical operations)
    - `Scikit-learn` (for machine learning models and metrics)
    - `Matplotlib` & `Seaborn` (for data visualization)

---

## 📂 Project Structure

# heart-disease-prediction
[├── HeartDisease.ipynb]  # Main Jupyter Notebook with all analysis  
[├── heart-disease-cleveland.csv]  # The dataset for the project  
[└── README.md]  # This documentation file  

## 📂 Setup  
use any virtual environments by installing jupyter extension in vs code  
## 📒 Notebook Walkthrough — Cell by Cell Explanation

This section explains each notebook cell in plain English (no code).  
Use this to follow the logic of the project step by step.

---
### I am leaving the cell of importing modules if u can understand a ML model you can understand this lol

### **Cell 1 — Load the dataset**

Reads the CSV file `heart-disease-cleveland.csv` into a DataFrame and shows the first rows.

**What it does / why it matters**
- Confirms the file is present and loads correctly.
- Shows column names and sample rows to verify dataset format.
- The columns are: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, diagnosis`.
- Seeing the raw rows early helps spot formatting issues (extra spaces in headers, unexpected characters, missing-value markers like `?`).

**Notes to check**
- In this CSV the final column header includes a leading space (`' diagnosis'`). That’s harmless but important: the column name must be used exactly.

---

### **Cell 2 — Data cleaning / missing-value handling / type conversion**

Replaces non-standard missing-value marks, converts columns to numeric types, and fills missing values with the column median. Then prints dataset info.

**What it does / why it matters**
- Replaces `"?"` with a proper missing-value marker.
- Converts all columns to numeric types so models can work with them.
- Fills missing values with the **median** (robust to outliers).
- Keeps dataset size intact instead of dropping rows.
- Reports the final data types and non-null counts so you know every column is numeric and has no missing values.

**Key observations**
- After cleaning there are 303 rows and 14 columns.
- Most columns are integers; a few (`oldpeak`, `ca`, `thal`) are floats — expected because of fractional or imputed values.

---

### **Cell 3 — Define features (X) and target (y)**

Separates the dataset into inputs (X) and target labels (y).

**What it does / why it matters**
- `X` becomes all patient features (age, sex, cp, …, thal).
- `y` is the `diagnosis` column — the value we want the model to predict.
- This separation is essential so the model learns patterns without “cheating” by seeing the label as input.

**Important detail**
- Because the header has a leading space, the target column is named `' diagnosis'`. Be careful to use the exact name.

**Note about the target values**
- In Cleveland data, `diagnosis` often contains `0` (no disease) and `1–4` (different disease levels).
- workflows convert this to binary (0 vs >0).  
- If not, the models will treat it as a multiclass target.

---

### **Cell 4 — Train / test split**

Splits X and y into training and testing sets (`test_size=0.2`, `random_state=100`).

**What it does / why it matters**
- Splits into ~80% training and ~20% testing (≈ 242 train rows, 61 test rows).
- Training set → used to fit model parameters.
- Test set → held back and used only once to evaluate model performance on unseen data.
- `random_state=100` ensures the same split each time (reproducibility).

**Why we keep a test set**
- Prevents over-optimistic performance.  
- If you test on the same data you trained on, accuracy is meaningless.

---

### **Cell 5 (expected) — Train Logistic Regression (fit on training data)**

**What it does / why it matters**
- Learns weights for each feature to output a probability of disease.
- Logistic Regression is a simple, interpretable baseline classifier.
- After training, predicts labels for the test set and computes accuracy.

**What to look for in outputs**
- Training completes without errors.
- Test accuracy (correct predictions / total).  
  This gives a baseline to compare with other models.

---

### **Cell 6 (expected) — Evaluate Logistic Regression with additional metrics**

**What it does / why it matters**
- Reports test metrics beyond accuracy:
  - **Precision** — of predicted positives, how many were real.
  
**Why this matters**
- Accuracy alone can mislead if the classes are imbalanced.

---

### **Cell 7 (expected) — Train Random Forest (fit on training data)**

**What it does / why it matters**
- Builds many decision trees and aggregates their votes.
- Captures complex, non-linear patterns and often outperforms Logistic Regression.
- After training, predicts on the test set and reports accuracy.

**Extra useful output**
- Feature importance ranking — shows which patient features (age, thalach, cp, etc.) most influence predictions.

**Why check feature importance**
- In medicine, this gives interpretability: which health measures are most predictive.

---

### **Cell 8 (expected) — Evaluate Random Forest & compare with Logistic Regression**

**What it does / why it matters**
- Prints accuracy and other metrics for Random Forest.
- Compares model performance:
  - Random Forest often has higher accuracy/recall.
  - Logistic Regression is easier to interpret.

**Decision guidance**
- Prefer Random Forest if you want higher performance.  
- Prefer Logistic Regression if interpretability is crucial.

---


## 📈 Model Performance

This section summarizes the performance of the trained models on the test data. The Logistic Regression model performed better in terms of overall accuracy.

| Model                   | Accuracy | 
| :---------------------- | :------: | 
| **Logistic Regression** |  78%     |
| **Random Forest** |  69%     |


**Conclusion:** Based on the evaluation metrics, the **Logistic Regression** model and its random forest classifier was found to be effective.

