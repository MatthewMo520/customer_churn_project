import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('customer_churn.csv')

# Cleaning data 

# removing any numerical data that cannot be converted to numbers
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors='coerce')
data = data.dropna()

# converting the data we trying to predict into numerical values
data["Churn"] = data["Churn"].map({"Yes" : 1, "No" : 0})

# dropping the column because it is not useful
data = data.drop(["customerID"], axis=1)

# one-hot encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# splitting into x and y (y is what we predict)
X = data.drop("Churn", axis=1)
y = data["Churn"]

# splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#----LOGISTIC REGRESSION MODEL----#
# training the model/ creating it
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# making predictions and evaluating the model
y_pred = model.predict(X_test)

# calculating accuracy
accuracy = accuracy_score(y_test, y_pred)


#----RANDOM FOREST MODEL----#
# training the model/ creating it
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#making predictions and evaluating the model
rf_y_pred = rf_model.predict(X_test)

# calculating accuracy
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")

# comparing the two models
print("Model Comparison:")
print(f"Logistic Regression Accuracy: {accuracy * 100:.4f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.4f}%")

# detailed classification report for Random Forest
print("\nDETAILED EVALUATION")
print("Random Forest Confusion Matrix:")
cm = confusion_matrix(y_test, rf_y_pred)
print(cm)
print("\nExplanation:")
print(f"Correctly predicted NO churn: {cm[0][0]}")
print(f"Incorrectly predicted YES(false alarm): {cm[0][1]}")
print(f"Incorrectly predicted NO(missed churn): {cm[1][0]}")
print(f"Correctly predicted YES churn: {cm[1][1]}")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred, target_names=["No Churn", "Churn"]))

# feature importance visualization for Random Forest
print("Feature Importance from Random Forest Model:")

# Get feature importances
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# show top 10 important features
print("Top 10 Important Features:")
print(feature_importance_df.head(10))

# visualization
plt.figure(figsize=(10, 6))
top_10 = feature_importance_df.head(10)
plt.barh(top_10['Feature'], top_10['Importance'])
plt.xlabel('Importance Score')
plt.title('Top 10 Feature That Influence Customer Churn')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# saving the chart
plt.savefig('feature_importance.png', dpi =  300, bbox_inches='tight')
print("Feature importance chart saved as 'feature_importance.png'")

print("\n" + "-"*50)
print("PROJECT SUMMARY")
print("-"*50)
print(f"""
Dataset: Telcon Customer Churn (Kaggle link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Total Customers: {len(data)}
Features Used: {X.shape[1]}
Target: Predict Customer Churn (Yes/No)

Models Trained:
    - Logistic Regression: {accuracy * 100:.4f}% accuracy
    - Random Forest Classifier: {rf_accuracy * 100:.4f}% accuracy

Random Forest Stats:
    - Accuracy: {rf_accuracy * 100:.4f}%
    - Correct Identified {cm[0,0]} loyal customers (No Churn)
    - Caught {cm[1,1]} out of {cm[1,0] + cm[1,1]} churners

Top 3 Important Features Influencing Churn:
    1. {feature_importance_df.iloc[0]['Feature']} (Importance: {feature_importance_df.iloc[0]['Importance']:.4f})
    2. {feature_importance_df.iloc[1]['Feature']} (Importance: {feature_importance_df.iloc[1]['Importance']:.4f})
    3. {feature_importance_df.iloc[2]['Feature']} (Importance: {feature_importance_df.iloc[2]['Importance']:.4f})

Insights:
    - Customers with higher monthly charges are more likely to churn.
      """)

print("-"*50)

