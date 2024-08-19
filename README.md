# Customer Churn Analysis Using Machine Learning

## Overview

This project is focused on analyzing customer churn data using various machine learning techniques to predict whether a customer will churn, join, or stay with the company. The analysis involves several steps, including data preprocessing, feature engineering, anomaly detection, and model training using the XGBoost algorithm with hyperparameter tuning.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Installation and Setup](#installation-and-setup)
4. [Data Preprocessing](#data-preprocessing)
    - [Handling Missing Values](#handling-missing-values)
    - [Feature Engineering](#feature-engineering)
    - [Anomaly Detection and Handling](#anomaly-detection-and-handling)
5. [Model Training](#model-training)
    - [Handling Imbalanced Data](#handling-imbalanced-data)
    - [XGBoost Model](#xgboost-model)
6. [Model Evaluation](#model-evaluation)
7. [Skills Demonstrated](#skills-demonstrated)
8. [Conclusion](#conclusion)
9. [License](#license)

## Introduction

The aim of this project is to perform a comprehensive analysis of customer churn data to build a predictive model. This model will help in identifying the customers who are likely to churn, stay, or join, enabling better decision-making and customer retention strategies.

## Data Description

The dataset used in this project consists of customer data including demographic details, account information, and service usage. It includes features such as age, tenure, internet service type, total charges, and customer status.

## Installation and Setup

To run this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/customer-churn-analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Launch the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## Data Preprocessing

### Handling Missing Values

The dataset contains several missing values, particularly in features related to internet services. These missing values were handled by filling in with appropriate placeholder values like 'no_internet_service' and 'no_offer' for nominal features.

### Feature Engineering

Feature engineering steps include encoding categorical variables, performing skewness transformations using logarithmic and square root transformations, and removing redundant columns. 

### Anomaly Detection and Handling

Anomalies in the data were detected using the Interquartile Range (IQR) method. Outliers were identified and handled by capping them at the lower and upper bounds derived from the training data.

## Model Training

### Handling Imbalanced Data

The dataset was highly imbalanced, with significantly more instances of certain classes. To address this, the SMOTETomek technique was applied to balance the class distribution in the training data.

### XGBoost Model

The XGBoost algorithm was chosen for its performance and efficiency in handling classification tasks. Hyperparameter tuning was performed using GridSearchCV to find the best model parameters. The model was trained using the resampled training data and evaluated on the test set.

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Grid Search
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='f1_macro'
)

grid_search_xgb.fit(X_train_resampled, y_train_resampled)

## Model Evaluation

The best model was evaluated on the test set using classification metrics like precision, recall, and F1-score. A confusion matrix was also generated to visualize the model’s performance.

## Skills Demonstrated

Python, Data Preprocessing, Feature Engineering, Anomaly Detection, Imbalanced Data Handling, SMOTETomek, XGBoost, GridSearchCV, Classification Metrics, Data Visualization

## Conclusion

The project successfully implemented a predictive model for customer churn using advanced data preprocessing techniques and machine learning. The XGBoost model, optimized through hyperparameter tuning, provided a robust solution for handling imbalanced data and achieving a high F1-score.

