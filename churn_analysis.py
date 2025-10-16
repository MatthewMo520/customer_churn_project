import pandas as pd
from sklearn.model_selection import train_test_split

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

X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("training set:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("test set:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)