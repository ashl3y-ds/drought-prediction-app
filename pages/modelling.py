import streamlit as st
import pandas as pd
import numpy as np
from sklearn import tree
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


    bg_color = "#f0f2f6"
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

st.write("### Customized Confusion Matrix without Heatmap:")
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed

# Set custom background colors for the figure and axes
fig.patch.set_facecolor("#363837")  # Background color for the figure
ax.set_facecolor("#363837")         # Background color for the axes

# Draw grid manually
num_classes = len(np.unique(y))  # Number of unique classes in the target
for i in range(num_classes):
    for j in range(num_classes):
        # Draw each grid square
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="white", edgecolor="black"))

        # Place confusion matrix value in the middle of the cell
        ax.text(
            j + 0.5,
            i + 0.5,
            f"{cm[i, j]}",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            weight="bold"
        )

# Set tick labels
ax.set_xticks(np.arange(num_classes) + 0.5)
ax.set_yticks(np.arange(num_classes) + 0.5)
ax.set_xticklabels(np.unique(y), fontsize=10, fontweight="bold", color="white")
ax.set_yticklabels(np.unique(y), fontsize=10, fontweight="bold", color="white")

# Add labels and title
ax.set_xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="white")
ax.set_ylabel("True Labels", fontsize=12, fontweight="bold", color="white")
ax.set_title(f"Customized Confusion Matrix", fontsize=14, fontweight="bold", color="white")

# Adjust axes limits and remove spines for a clean look
ax.set_xlim(0, num_classes)
ax.set_ylim(num_classes, 0)
ax.spines[:].set_visible(False)

# Remove any extra margin
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Display the plot
st.pyplot(fig)
