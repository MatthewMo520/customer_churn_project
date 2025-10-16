# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave (churn) using classification algorithms.

## Project Overview

This project analyzes customer data from a telecommunications company to predict churn. The model helps identify at-risk customers so the company can take proactive retention measures.

- **Dataset**: Telco Customer Churn from Kaggle
- **Total Customers**: 7,032 (after cleaning)
- **Features**: 30 features including contract type, monthly charges, tenure, services used, etc.
- **Target Variable**: Churn (Yes/No)

## Objective

Predict whether a customer will churn based on their account information, services, and usage patterns.

## 🛠️ Technologies Used

- **Python 3.13.5**
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization

## 📁 Project Structure
```
customer_churn_project/
├── churn_analysis.py          
├── customer-churn.csv  # Dataset
├── feature_importance.png  
├── README.md    
└── requirements.txt                  
```

## 🔍 Process

### 1. Data Cleaning
- Converted `TotalCharges` from object to numeric type
- Handled 11 missing values by removing incomplete records
- Removed `customerID` column (not predictive)

### 2. Data Preprocessing
- Converted target variable `Churn` to binary (Yes=1, No=0)
- Applied one-hot encoding to categorical variables
- Split data into training (80%) and testing (20%) sets

### 3. Model Training
Trained and compared two classification models:

- **Logistic Regression**: 78.75% accuracy
- **Random Forest**: 78.54% accuracy

### 4. Model Evaluation
Used confusion matrix and classification report to analyze performance:
```
Confusion Matrix (Random Forest):
- Correctly predicted NO churn: 927 (90% of non-churners)
- Correctly predicted YES churn: 178 (48% of churners)
- False alarms: 106
- Missed churners: 196
```

## 📈 Key Findings

### Top 3 Churn Predictors:
1. **Total Charges** (19.3% importance)
2. **Monthly Charges** (17.0% importance)
3. **Tenure** (16.8% importance)

### Business Insights:
- New customers (low tenure) are at higher risk of churning
- Customers with high monthly charges are more likely to leave
- Fiber optic internet customers churn more than DSL customers
- Electronic check payment method correlates with higher churn

## 💡 Recommendations

Based on the analysis, the company should:
1. Focus retention efforts on new customers (first 6-12 months)
2. Offer competitive pricing or discounts for high monthly charge customers
3. Improve fiber optic service quality or provide better support
4. Encourage automatic payment methods over electronic checks

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/MatthewMo520/customer_churn_project
cd customer_churn_project
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project folder

4. Run the analysis
```bash
python churn_analysis.py
```

## 📊 Results

- **Final Model**: Random Forest Classifier
- **Accuracy**: 78.54%
- **Key Metric**: Successfully identified 90% of loyal customers
- **Area for Improvement**: Only caught 48% of actual churners (class imbalance issue)

## 🔮 Future Improvements

- Address class imbalance using SMOTE or weighted classes
- Try additional algorithms (XGBoost, Neural Networks)
- Perform hyperparameter tuning with GridSearchCV
- Add more feature engineering
- Create an interactive dashboard for predictions

## 👤 Author

Matthew Mo