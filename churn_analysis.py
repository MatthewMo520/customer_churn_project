import pandas as pd

data = pd.read_csv('customer_churn.csv')

print(data.head())
print(data.info())