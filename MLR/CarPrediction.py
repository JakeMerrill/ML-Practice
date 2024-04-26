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
    return data[(data < lower_bound) | (data > upper_bound)]

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

outliers = []

for col in numerical_cols:
    outliers.append(IQR_outlier_detection(car_data[col]))

# Display the outliers
outliers_df = pd.DataFrame(outliers).T
#print(outliers_df.head())
outliers_df.info()

# Cleaning the data by removing the outliers
for col in numerical_cols:
    car_data = car_data[~car_data[col].isin(outliers_df[col])]

# Display the cleaned data
#print(car_data.head())
car_data['Price'] = car_data['Price']/100 # Convert the pricefrom INR to USD
print(car_data['Price'].head())
car_data.info()

# EDA

# Categorical Variable Univariate Analysis (Countplot)

# Calculate the number of rows and columns for subplots
num_cols = len(categorical_cols)
num_rows = (num_cols + 2) // 4  # Adjust the number of rows based on the number of columns and desired number of columns per row

# Create subplots with adjusted spacing
fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows), dpi=90)

# Manually adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=1.3)

# Iterate over each categorical column and plot
for i, col in enumerate(categorical_cols):
    ax = sns.countplot(y=car_data[col], ax=axes[i // 4, i % 4])  # Adjust indexing for row and column
    ax.set_title(f"Countplot of {col}")
    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
    ax.tick_params(axis='y', which='major', labelsize=12)  # Increase y-axis label size

    # Rotate y-axis labels and reduce label size if there are too many labels
    if len(car_data[col].unique()) > 10:
        ax.tick_params(axis='y', rotation=20, labelsize=8)
        ax.get_figure().subplots_adjust(top=0.9)  # Adjust top parameter as needed

    # Add counts to the end of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_width()}',
                    (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points')

# Hide empty subplots if necessary
for j in range(num_cols, num_rows * 4):
    fig.delaxes(axes[j // 4, j % 4])

plt.show()

# Numeric Variable Univariate Analysis (Histograms)


fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15,10), dpi=100)
c = '#0055ff'

for i in range(len(numerical_cols)):
    row = i//3
    col = i%3
    values, bin_edges = np.histogram(car_data[numerical_cols[i]], 
                                     range=(np.floor(car_data[numerical_cols[i]].min()), np.ceil(car_data[numerical_cols[i]].max())))                
    graph = sns.histplot(data=car_data, x=numerical_cols[i], bins=bin_edges, kde=True, ax=ax[row,col],
                         edgecolor='none', color=c, alpha=0.4, line_kws={'lw': 2.5})
    ax[row,col].set_xlabel(numerical_cols[i], fontsize=15)
    ax[row,col].set_ylabel('Count', fontsize=12)
    ax[row,col].set_xticks(np.round(bin_edges,1))
    ax[row,col].set_xticklabels(ax[row,col].get_xticks(), rotation = 45)
    ax[row,col].grid(color='lightgrey')
    for j,p in enumerate(graph.patches):
        ax[row,col].annotate('{}'.format(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()+1),
                             ha='center', fontsize=10 ,fontweight="bold")
    
    textstr = '\n'.join((
    r'$\mu=%.2f$' %car_data[numerical_cols[i]].mean(),
    r'$\sigma=%.2f$' %car_data[numerical_cols[i]].std(),
    r'$\mathrm{median}=%.2f$' %np.median(car_data[numerical_cols[i]]),
    r'$\mathrm{min}=%.2f$' %car_data[numerical_cols[i]].min(),
    r'$\mathrm{max}=%.2f$' %car_data[numerical_cols[i]].max()
    ))
    ax[row,col].text(0.6, 0.9, textstr, transform=ax[row,col].transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round',facecolor='#509aff', edgecolor='black', pad=0.5))

ax[1, 2].axis('off')
plt.suptitle('Distribution of Numerical Variables', fontsize=20) 
plt.tight_layout()   
plt.show()

# Bivariate Analysis (Scatter Plots)

fig, ax = plt.subplots(nrows=4 ,ncols=2, figsize=(10,10), dpi=90)
c = '#0055ff'

num_features = ['Year', 'Kilometer', 'Engine', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']
target = 'Price'

for i in range(len(num_features)):
    row = i//2
    col = i%2
    ax[row,col].scatter(car_data[num_features[i]], car_data[target], color=c, edgecolors='w', linewidths=0.25)
    ax[row,col].set_title('{} vs. {}'.format(target, num_features[i]), size = 12)
    ax[row,col].set_xlabel(num_features[i], size = 12)
    ax[row,col].set_ylabel(target, size = 12)
    ax[row,col].grid()

plt.suptitle('Selling Price vs. Numerical Features', size = 20)
plt.tight_layout()
plt.show()

# Bi-variate Analysis (Strip Plots)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), dpi=100)
cat_features = ['Fuel Type', 'Transmission', 'Color', 'Owner', 'Seller Type', 'Drivetrain']
target = 'Price'
c = '#0055ff'

# Iterate over each subplot position
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        if i * len(ax_row) + j >= len(cat_features):
            # If we run out of categories, break the loop
            break
        sns.stripplot(ax=ax, x=cat_features[i * len(ax_row) + j], y=target, data=car_data, size=6, color=c)
        ax.set_title('{} vs. {}'.format(target, cat_features[i * len(ax_row) + j]), size=13)
        ax.set_xlabel(cat_features[i * len(ax_row) + j], size=12)
        ax.set_ylabel(target, size=12)
        ax.grid()

plt.suptitle('Selling Price vs. Categorical Features', size=20)
plt.tight_layout()
plt.show()

# Categorical Encoding (one-hot encoding)

car_data.drop(['Make','Color'], axis=1, inplace=True)

CatCols = ['Fuel Type', 'Transmission', 'Owner', 'Seller Type', 'Drivetrain']

car_data = pd.get_dummies(car_data, columns=CatCols, drop_first=True, dtype=int)
print(car_data.head(25))


# Correlation Matrix

corr = car_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()