import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# training the model/ creating it
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# making predictions and evaluating the model
y_pred = model.predict(X_test)

# calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(f"That's {accuracy * 100:.2f}% accuracy!")