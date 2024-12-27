import streamlit as st
import pandas as pd
import numpy as np
from sklearn  import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that both filtered features and target variable are available
if "filtered_df" in st.session_state and "target" in st.session_state:
    st.write("### Ranked Data Preview:")
    st.write(st.session_state["filtered_df"].head())

    # Load features and target variable from session state
    X = st.session_state["filtered_df"]
    y = st.session_state["target"]  # Load the target column from session state

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save processed data back to session state
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    # Show the number of samples in train and test sets
    st.write(f"Number of training samples: {X_train.shape[0]}")
    st.write(f"Number of testing samples: {X_test.shape[0]}")

    # Algorithm selection by the user
    algorithm = st.selectbox("Select Algorithm", ["Support Vector Machine (SVM)", "Decision Tree", "K-Nearest Neighbors (KNN)", "Random Forest"])

    # Train and evaluate model based on selection
    if algorithm == "Support Vector Machine (SVM)":
        model = SVC()
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algorithm == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

 # Custom confusion matrix visualization
st.write("### Customized Confusion Matrix with Outside Coloring:")
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

# Set custom background colors
fig.patch.set_facecolor('#f0f0f5')  # Light grey background for the figure
ax.set_facecolor('#e6e6ff')         # Light blue background for the axes area outside the matrix

# Normalize the confusion matrix if required (optional)
normalized_cm = cm / cm.sum(axis=1)[:, np.newaxis]  # Row-wise normalization for percentages

# Plot heatmap
sns.heatmap(
    cm, 
    annot=True,
    fmt='d',               # Raw values format
    cmap="coolwarm",        # Colormap for the matrix
    cbar=True,
    square=True,
    linewidths=0.5,         # Borders between the squares
    linecolor='black',      # Border color for squares
    xticklabels=np.unique(y),
    yticklabels=np.unique(y),
    ax=ax                   # Use the customized axes
)

# Styling for title, labels, and colorbar
ax.set_xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="black")
ax.set_ylabel("True Labels", fontsize=12, fontweight="bold", color="black")
ax.set_title(f"Customized Confusion Matrix", fontsize=14, fontweight="bold", color="black")

# Add additional padding outside the axes if needed
fig.tight_layout(pad=3)

# Display the plot
st.pyplot(fig)
