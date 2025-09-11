# USED-CAR-PREDICTION-MODEL
The primary objective of this project is to gain valuable insights into the dataset through EDA and build a robust machine learning model to predict the price of used cars.
## Problem Statement
The dataset from DSN, which includes both training and testing data, contains a wide range of features such as car make, model, year of manufacture, mileage, fuel type, and several other factors that influence the pricing of used cars.
## Features in the Dataset
- `id`: Just an identifier for each car listing.
- `brand`: The manufacturer (e.g., Toyota, BMW, Honda)..
- `model`: The specific car model under the brand (e.g., Corolla, Civic, X5)..
- `year`: The year the car was manufactured..
- `model_age`: Derived feature: current year minus manufacture year..
- `mileage`: Total distance the car has been driven (in km or miles)..
- `horsepower (hp)`: Engine power rating..
- `hp_per_liter`: Engine efficiency measure (horsepower per liter of engine capacity)..
- `fuel_type`: Type of fuel (petrol, diesel, electric, hybrid)..
- `transmission`: Gear system: automatic, manual, CVT..
- `exterior_color`: Car’s paint color.
- `interior_color`: Color of seats/dashboard..
## Models and Techniques
This project evaluates multiple regression models using pipelines:
- **Linear Regression**
- **Lasso Regression**
- **Gradient Boosting**
- **Random Forest**
- **AdaBoost**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**
Each model's performance is measured using the mean squared error (MSE) on the test dataset. The best-performing model is selected for deployment.
## Implementation Steps
## 1. Data Collection  
Collecting a diverse dataset is a foundational step where the focus is on gathering comprehensive information relevant to the problem at hand. This process involves identifying and obtaining data sources that provide the necessary context for training and testing machine learning models.  

---

## 2. Exploratory Data Analysis (EDA)  
Exploring and analyzing the dataset is a critical phase that involves uncovering insights, understanding data distributions, and identifying potential patterns. Through visualization and statistical techniques, EDA aims to reveal the structure and characteristics of the data, aiding in feature selection and model development.  

---

## 3. Data Preprocessing  
Data preprocessing focuses on preparing the dataset for model training by addressing issues such as missing values, outliers, and inconsistencies. Tasks include cleaning the data, handling null values, scaling features, and encoding categorical variables to ensure a consistent and reliable input for machine learning algorithms.  

---

## 4. Feature Engineering  
Feature engineering involves creating new features or modifying existing ones to enhance the model's ability to capture meaningful patterns. This step requires domain knowledge and creativity, as engineers aim to extract relevant information and improve the model's predictive performance by providing it with more informative features.  

---

## 5. Model Selection  
Model selection is the process of choosing an appropriate machine learning algorithm based on the characteristics of the data and the nature of the problem. Different algorithms have strengths and weaknesses, and the selection process involves identifying the most suitable model architecture for achieving the desired outcomes.  

---

## 6. Model Training  
Model training involves feeding the chosen algorithm with the prepared dataset to enable it to learn patterns and relationships. During this phase, the model adjusts its internal parameters iteratively to minimize the difference between its predictions and the actual outcomes in the training data.  

---

## 7. Model Evaluation  
Model evaluation is the critical assessment of the trained model's performance using validation data. Metrics such as accuracy, precision, recall, F1 score, or area under the ROC curve are employed to gauge the model's effectiveness and generalization capabilities to new, unseen data.  

---

## 8. Hyperparameter Tuning  
Hyperparameter tuning involves optimizing the model's hyperparameters to achieve better performance. Techniques such as grid search or random search are employed to find the best combination of hyperparameter values, fine-tuning the model for optimal results.  

# Results
The best model was **Lasso**, achieving an RMSE of **38001.684877432745**.
## Prerequisites
- Python 3.x
- Libraries: pandas, NumPy, scikit-learn, XGBoost

To install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```
## Project Structure
```
.
├── rental_info.csv       # Dataset
├── notebook.ipynb      # Data preprocessing, model training, model evaluation
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies
```
## Acknowledgements
- The train and test used car prediction dataset provided by the DSN
- Libraries and frameworks: scikit-learn, XGBoost, pandas, and NumPy.

