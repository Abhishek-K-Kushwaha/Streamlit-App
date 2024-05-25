import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the data
st.title('Task:2 Machine Learning - Classification')
st.write("Loading data...")

train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

st.write("Data loaded successfully!")
st.write("First few rows of training data:")
st.write(train_data.head())

# Separate features and target
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# Encode the target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest classifier
st.write("Training Random Forest classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the validation set
y_val_pred = clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
st.write(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Predict on the test data
st.write("Predicting on test data...")
test_pred = clf.predict(test_data.values)
test_pred_labels = le.inverse_transform(test_pred)
test_results = pd.DataFrame(test_pred_labels, columns=['Predicted'])
st.write("""###Predictions for test data:""")
st.write(test_results)

# Calculate accuracy on the training data
y_train_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
st.write(f'Training Accuracy: {train_accuracy * 100:.2f}%')

# Calculate training and validation loss over iterations for RandomForest
train_losses = []
val_losses = []

for n_estimators in range(10, 101, 10):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    
    train_loss = 1 - clf.score(X_train, y_train)
    val_loss = 1 - clf.score(X_val, y_val)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plot the training and validation loss
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(10, 101, 10), train_losses, label='Training Loss')
ax.plot(range(10, 101, 10), val_losses, label='Validation Loss')
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss vs. Number of Estimators')
ax.legend()

st.pyplot(fig)

# Explanation for choosing Random Forest Classifier
st.write("""
### Why Random Forest Classifier?
Random Forest Classifier is chosen for the following reasons:
- **High Accuracy**: Random Forests generally provide high accuracy compared to other algorithms.
- **Robustness**: It is robust to overfitting and handles well in high-dimensional spaces.
- **Implicit Feature Selection**: It automatically selects important features, reducing the need for feature engineering.
- **Efficiency**: It is computationally efficient and can handle large datasets with high dimensionality.
- **Versatility**: It can be used for both classification and regression tasks.
""")
