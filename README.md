# Diabetes Predictionn using Logistic Regression

A machine learning project that predicts whether a patient is diabetic based on medical parameters.
This project includes data cleaning, EDA, preprocessing, model building, and evaluation.

## 📌 Overview

The aim of this project is to build a robust and interpretable model that can assist in early detection of diabetes. The dataset includes diagnostic measurements such as glucose level, BMI, insulin, age, etc.
This project showcases a complete ML workflow - from raw data to a trained prediction model.

## 📂 Project Structure

```
DIABETES-PREDICTION/
│
├── data/
│   ├── diabetes.csv
│   ├── refined_data.csv
│
├── notebooks/
│   ├── EDA and preprocessing.ipynb
│   ├── prediction.ipynb
│
├── images/
│   ├── (plots & visualizations)
│
├── models/
│   ├── classification_model.pkl
│
└── README.md
```

## 🎯 Objectives

- Perform exploratory data analysis
- Clean and preprocess data
- Handle zero/missing values
- Detect and treat outliers
- Standardize numerical features
- Fix class imbalance using SMOTE
- Build ML models for prediction
- Compare model performance
- Save the best model

## 🧠 Machine Learning Workflow

### 1️⃣ Data Loading

Load the original dataset (diabetes.csv) and create a cleaned version (refined_data.csv).


### 2️⃣ Exploratory Data Analysis

- Descriptive statistics
- Distribution of features
- Correlations
- Outlier analysis 
- Target variable balance

  
### 3️⃣ Data Preprocessing

- Replace zero values 
- Impute missing values
- Remove or cap outliers
- Standardize features using StandardScaler
- Extract target column

  
### 4️⃣ Train–Test Split

test_size = 0.2, random_state = 42


### 5️⃣ Handling Class Imbalance

Applied SMOTE:

Oversamples the minority class to improve model performance.


### 6️⃣ Model Training

Models evaluated: Logistic Regression


### 7️⃣ Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall
- F1-score 


8️⃣ Saving the Best Model

pickle.dump(model, open('classification_model.pkl', 'wb'))

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (SMOTE)
- Pickle
- Jupyter Notebook


## 📈 Future Improvements

- Deploy using Streamlit or Flask
- Hyperparameter tuning
- Add cross-validation
- Use advanced ML models
- Build an interactive dashboard


## 👩‍💻 Author

**[Srishti Sahu](https://github.com/srishtisahu03)**

**[This takes you to my LinkedIn!](www.linkedin.com/in/srishti-sahu-cl0316)**
