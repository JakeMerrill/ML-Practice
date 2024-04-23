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

# Step 2: Load the data
car_data = pd.read_csv(r'MLR\car details v4.csv') # or use 'MLR\\car details v4.csv' instead of the raw string

# Step 3: Data Overview/General Picture
print(car_data.shape)
