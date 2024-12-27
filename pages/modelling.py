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

st.write("### Customized Confusion Matrix (Dynamic Color Based on Values):")
fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size as needed

# Set custom background colors for the figure and axes
fig.patch.set_facecolor("#808080")  # Background color for the figure
ax.set_facecolor("#808080")         # Background color for the axes

# Number of classes
classes = np.unique(y)
n_classes = len(classes)

# Normalizing the values to create dynamic coloring (scaling between 0 and 1)
normalized_cm = cm / cm.max()

# Draw grid for the matrix
ax.set_xticks(np.arange(n_classes) + 0.5, minor=False)
ax.set_yticks(np.arange(n_classes) + 0.5, minor=False)
ax.set_xticks(np.arange(n_classes + 1), minor=True)
ax.set_yticks(np.arange(n_classes + 1), minor=True)

# Configure grid and labels
ax.set_xticklabels(classes, fontsize=12, fontweight="bold", color="black")
ax.set_yticklabels(classes, fontsize=12, fontweight="bold", color="black")
ax.xaxis.tick_top()  # Move the x-axis labels to the top
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', which='both', length=0)  # Hide tick marks

# Customize appearance
ax.grid(False)  # Turn off the default grid
ax.grid(True, which="minor", color="black", linewidth=1)  # Add gridlines around cells
ax.set_xlim(0, n_classes)
ax.set_ylim(n_classes, 0)

# Add dynamic background colors and text
for i in range(n_classes):
    for j in range(n_classes):
        # Compute a color intensity based on the value
        intensity = normalized_cm[i, j]
        color = plt.cm.Blues(intensity)  # Use Blues colormap to assign colors

        # Draw the cell rectangle with dynamic color
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black")
        ax.add_patch(rect)

        # Add the value as text (always in black)
        ax.text(j + 0.5, i + 0.5, str(cm[i, j]), 
                ha="center", va="center", fontsize=12, color="black")

# Label titles
ax.set_xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="black", labelpad=20)
ax.set_ylabel("True Labels", fontsize=12, fontweight="bold", color="black", labelpad=20)
ax.set_title("Customized Confusion Matrix", fontsize=14, fontweight="bold", color="black", pad=20)

# Tight layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Display the plot
st.pyplot(fig)
