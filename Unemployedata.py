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
