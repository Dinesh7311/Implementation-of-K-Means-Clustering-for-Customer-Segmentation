# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas and matplotlib.pyplot
2.Read the dataset and transform it
3.Import KMeans and fit the data in the model
4.Plot the Cluster graph 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Dinesh s
RegisterNumber:24900464
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv("Mall_Customers.csv")

# Check the first few rows and basic info
print(data.head())
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Prepare for the Elbow Method
wcss = []  # Within-Cluster Sum of Square (WCSS)

# Elbow Method to find the optimal number of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])  # Assuming the relevant data starts at column index 3
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Apply KMeans with the optimal number of clusters (5 clusters in this case)
km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])  # Fit the KMeans model to the relevant columns
y_pred = km.predict(data.iloc[:, 3:])

# Assign the cluster labels to the data
data["cluster"] = y_pred

# Separate the data by clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Plot the clusters
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 1")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="black", label="Cluster 2")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="blue", label="Cluster 3")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="green", label="Cluster 4")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 5")

# Add labels and legend
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.show()
...

## Output:
![ml 10-1 - Copy](https://github.com/user-attachments/assets/c4589cfe-1a98-4be8-a250-e22dd6c3e0a4)
![ml10-2](https://github.com/user-attachments/assets/52d67587-d17e-456a-837a-ea28c5ef24b2)
![ml 10-3](https://github.com/user-attachments/assets/84fec363-51fb-4af9-8cdd-6cd0d1c16f1d)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
