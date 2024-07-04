# CipherByte...
You import the necessary libraries for data manipulation and visualization (pandas, numpy, matplotlib.pyplot, and seaborn).

You set the visualization style using seaborn.

You load two Excel files into pandas DataFrames (df1 and df2).

You print the first few rows of each DataFrame to inspect the data.

You check for missing values in both DataFrames and then drop the rows with missing values using dropna().

You reset the index of the DataFrames after dropping the missing values.

You merge the two DataFrames on the 'Date' column, adding suffixes to differentiate between columns from the two datasets.

You print the merged dataset and display its summary statistics.

You create a line plot to visualize the unemployment rates over time from both datasets.

You create a line plot to visualize the average unemployment rates by region from the first dataset.

You create a heatmap to visualize the unemployment rates by region and date from the first dataset.

Here are a few things to consider and potential issues to check:

Make sure that the 'Date' column is in a datetime format for proper handling and plotting. If it's not, you can convert it using pd.to_datetime().

When dropping missing values, ensure that it doesn't significantly reduce your dataset size, which could impact your analysis.

The plt.plot() function expects the x-axis to be sorted. If your 'Date' column is not sorted, you may need to sort it before plotting.

In the heatmap section, ensure that the 'Region_1' and 'Date' columns are suitable for creating a pivot table. The regions should be categorical, and the dates should be in a format that allows for proper indexing.

The heatmap may become cluttered if there are too many dates. You might want to aggregate the data by month or year before creating the pivot table for the heatmap.

UNEMPLOYEE DATA CODE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set(style='whitegrid')
# Load the data
file_path1 = 'C:\\Users\\admin\\Desktop\\Unemployment in India.xlsx'

file_path2 = 'C:\\Users\\admin\\Desktop\\Unemployment_Rate_upto_11_2020.xlsx'  # Replace with your file path
df1 = pd.read_excel(file_path1, engine='openpyxl')
df2 = pd.read_excel(file_path2, engine='openpyxl')
#df = pd.read_csv("C:\\Users\\admin\\Desktop\\Unemployment in India.xlsx")

# Display the first few rows of the dataframe
print(df1.head())
print(df2.head())
print(df1.isnull().sum())
print(df2.isnull().sum())
df1.dropna(inplace=True)
df2.dropna(inplace=True)
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)
print(df1.head())
print(df2.head())

# Merge datasets on the 'Date' column
df_merged = pd.merge(df1, df2, on='Date', suffixes=('_1', '_2'))

# Display the merged dataset
print("Merged dataset:")
print(df_merged.head())
# Summary statistics of the merged dataset
print(df_merged.describe())

# Plotting unemployment rates over time from both datasets
plt.figure(figsize=(12, 6))
plt.plot(df_merged['Date'], df_merged['Estimated Unemployment Rate (%)_1'], label='Dataset 1')
plt.plot(df_merged['Date'], df_merged['Estimated Unemployment Rate (%)_2'], label='Dataset 2')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Plotting average unemployment rates by region
regions = df_merged['Region_1'].unique()

plt.figure(figsize=(12, 8))
for region in regions:
    region_data = df_merged[df_merged['Region_1'] == region]
    plt.plot(region_data['Date'], region_data['Estimated Unemployment Rate (%)_1'], label=region)

plt.title('Unemployment Rate by Region (Dataset 1)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
# Create a heatmap to visualize unemployment rates by region and date
heatmap_data = df_merged.pivot_table(index='Region_1', columns='Date', values='Estimated Unemployment Rate (%)_1')

plt.figure(figsize=(14, 10))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".1f")
plt.title('Unemployment Rate Heatmap by Region and Date (Dataset 1)')
plt.xlabel('Date')
plt.ylabel('Region')
plt.show()

IRIS.PY CODE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('C:\\Users\\admin\\Downloads\\Iris Flower - Google Sheets.csv', header=0)

# Extract features and labels
feature_names = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X = df[feature_names]
y = df["Species"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.2f}")

# Define new flower measurements for prediction
multiple_flowers = [
    [5.8, 2.7, 5.1, 1.9],  # Flower 1
    [6.4, 3.2, 5.5, 2.1],  # Flower 2
    [5.1, 2.5, 4.0, 1.3]   # Flower 3
]

# Convert list to DataFrame with the same feature names
multiple_flowers_df = pd.DataFrame(multiple_flowers, columns=feature_names)

# Predict species for new samples
predicted_species = knn.predict(multiple_flowers_df)
print(predicted_species)  # Output predicted species for each flower
