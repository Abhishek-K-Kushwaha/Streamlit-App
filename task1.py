import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to identify cluster for a given data point
def identify_cluster(data_point, model, scaler):
    data_point_scaled = scaler.transform([data_point])
    cluster = model.predict(data_point_scaled)
    return cluster[0]

# Streamlit app
st.title("KMeans Clustering Visualization")

# File uploader for train and test data
train_file = st.file_uploader("Upload train data (Excel)", type=["xlsx"])
test_file = st.file_uploader("Upload test data (Excel)", type=["xlsx"])

if train_file is not None and test_file is not None:
    # Load the data
    train_data = pd.read_excel(train_file)
    test_data = pd.read_excel(test_file)
    train_data = train_data.iloc[:, :-1]  # Remove the target column
    
    # Data Preprocessing
    features = train_data.columns.tolist()
    
    # Standardize the data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # PCA for 2D visualization
    pca_2d = PCA(n_components=2)
    train_reduced_2d = pca_2d.fit_transform(train_scaled)
    
    # 2D Visualization
    st.subheader("Visualization of Train Data in 2D")
    fig, ax = plt.subplots()
    ax.scatter(train_reduced_2d[:, 0], train_reduced_2d[:, 1], s=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('2D PCA Visualization')
    st.pyplot(fig)
    
    # PCA for 3D visualization
    pca_3d = PCA(n_components=3)
    train_reduced_3d = pca_3d.fit_transform(train_scaled)
    
    # 3D Visualization
    st.subheader("Visualization of Train Data in 3D")
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_reduced_3d[:, 0], train_reduced_3d[:, 1], train_reduced_3d[:, 2], s=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Visualization')
    st.pyplot(fig)
    
    # Elbow method to determine the optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=2)
        kmeans.fit(train_scaled)
        wcss.append(kmeans.inertia_)
    
    st.subheader("Elbow Method for Optimal Clusters")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    ax.grid('both')
    st.pyplot(fig)
    
    # Applying KMeans clustering
    n_clusters = st.slider("Select number of clusters", 1, 10, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_scaled)
    
    # Predict clusters for training and test data
    train_clusters = kmeans.predict(train_scaled)
    test_clusters = kmeans.predict(test_scaled)
    
    # 2D Cluster Visualization
    st.subheader("Cluster Visualization in 2D")
    fig, ax = plt.subplots()
    ax.scatter(train_reduced_2d[:, 0], train_reduced_2d[:, 1], c=train_clusters, s=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('2D Cluster Visualization')
    st.pyplot(fig)
    
    # 3D Cluster Visualization
    st.subheader("Cluster Visualization in 3D")
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_reduced_3d[:, 0], train_reduced_3d[:, 1], train_reduced_3d[:, 2], c=train_clusters, s=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D Cluster Visualization')
    st.pyplot(fig)
    
    # Input for identifying cluster of a new data point
    st.subheader("Identify Cluster for a New Data Point")
    data_point_input = [st.number_input(f"Feature {feature}", value=0.0) for feature in features]
    if st.button("Identify Cluster"):
        cluster = identify_cluster(data_point_input, kmeans, scaler)
        st.write(f"The data point belongs to cluster: {cluster}")
