import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Visualize the data using a pairplot
sns.pairplot(df, hue="Species", markers=["o", "s", "D"])
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# Visualize feature correlation using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()
