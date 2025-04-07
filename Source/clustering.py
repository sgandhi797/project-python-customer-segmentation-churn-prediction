"""
Customer Clustering Script
- Standardizes RFM features
- Applies KMeans clustering
"""

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

def perform_clustering(rfm, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans

def plot_clusters(rfm, filename="visuals/clusters_plot.png"):
    sns.pairplot(rfm.reset_index(), hue='Cluster', palette='Set1')
    plt.savefig(filename)