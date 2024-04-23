# Step 1: import libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures

#Functions 
def IQR_outlier_detection(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Step 2: Load the data
car_data = pd.read_csv(r'MLR\car details v4.csv') # or use 'MLR\\car details v4.csv' instead of the raw string

# Step 3: Data Overview
print(car_data.head(5)) # Display the first 5 rows of the data

car_data.info()

print(car_data.describe(include='number'))

print(car_data.describe(include='object'))

#Step 4: Data Preparation

#Subset Selection

# Remove "Max Power" and "Max Torque" columns
car_data.drop(["Max Power", "Max Torque"], axis=1, inplace=True)

car_data.dropna(inplace=True)  # Drop rows with null values
#print(car_data.isnull().sum()) # Confirming that all null values are dropped

car_data['Engine'] = car_data['Engine'].str[:4].astype(np.int64) # Extracting only the first 4 characters of the "Engine" column and converting it to an int64

# List of categorical columns
categorical_cols = car_data.select_dtypes(include=['object']).columns
#print("Categorical columns: ", categorical_cols)

# List of numerical columns
numerical_cols = car_data.select_dtypes(include=['int64', 'float64']).columns
#print("Numerical columns: ", numerical_cols)

drop_cols = []
for col in categorical_cols:
    unique_entries = car_data[col].nunique()
    if unique_entries > 50:
        drop_cols.append(col)
    #print(f"Number of unique entries in {col}: {unique_entries}")
print(drop_cols)

# Drop columns with more than 50 unique entries
categorical_cols = [col for col in categorical_cols if col not in drop_cols]
car_data.drop(drop_cols, axis=1, inplace=True)

#print(car_data[categorical_cols].head())
#print(car_data[numerical_cols].head())

#Outlier Detections
