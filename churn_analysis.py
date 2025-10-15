import pandas as pd

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


print(data["Churn"].value_counts())
print(data.head())
