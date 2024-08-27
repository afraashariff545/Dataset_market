import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
url = "C:/Users/545af/Downloads/inter_prod/Mall_Customers.csv"
df = pd.read_csv(url)

# Display the first few rows
df.head()

# Check for missing values
print(df.info())

# Summary statistics
print(df.describe())

# Drop 'CustomerID' column since it's not useful for clustering
df = df.drop(columns=['CustomerID'])

# Handle categorical data (e.g., 'Gender' column)
df = pd.get_dummies(df, drop_first=True)

# Feature Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Elbow method to find the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

# Silhouette analysis
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    score = silhouette_score(df_scaled, kmeans.labels_)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Assume optimal clusters found is 4
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(df_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Group data by clusters and calculate the mean
cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# Visualize the clusters
sns.pairplot(df, hue='Cluster', palette='tab10')
plt.show()


