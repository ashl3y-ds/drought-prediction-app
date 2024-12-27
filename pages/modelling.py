import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

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

    # Initialize an empty list to store classification reports in session state
    if "model_reports" not in st.session_state:
        st.session_state["model_reports"] = []

    # Algorithm selection by the user
    algorithm = st.selectbox("Select Algorithm", ["Support Vector Machine (SVM)", "Decision Tree", "K-Nearest Neighbors (KNN)", "Random Forest"])

    # Train and evaluate model based on selection
    if algorithm == "Support Vector Machine (SVM)":
        model = SVC(kernel='linear', probability=True)
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

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    # Store the classification report for this model
    classes = np.unique(y)
    report_dict = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Add model report to the list of reports in session state
    st.session_state["model_reports"].append({
        "model_name": algorithm,
        "accuracy": accuracy * 100,
        "classification_report": report_df
    })

    # Display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 3))  # Adjust figure size as needed
    fig.patch.set_alpha(0) 
    # Normalizing the values to create dynamic coloring (scaling between 0 and 1)
    normalized_cm = cm / cm.max()
    n_classes = len(classes)

    ax.set_xticks(np.arange(n_classes) + 0.5, minor=False)
    ax.set_yticks(np.arange(n_classes) + 0.5, minor=False)
    ax.set_xticks(np.arange(n_classes + 1), minor=True)
    ax.set_yticks(np.arange(n_classes + 1), minor=True)
    ax.set_xticklabels(classes, fontsize=12, fontweight="bold", color="yellow")
    ax.set_yticklabels(classes, fontsize=12, fontweight="bold", color="yellow")
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="both", which="both", length=0)  # Hide tick marks

    # Customize gridlines for cells
    ax.grid(False)
    ax.grid(True, which="minor", color="gray", linewidth=1)
    ax.set_xlim(0, n_classes)
    ax.set_ylim(n_classes, 0)

    # Add dynamic cell colors and values
    for i in range(n_classes):
        for j in range(n_classes):
            intensity = normalized_cm[i, j]  # Compute color intensity
            red_intensity = 1.0  # Red channel fixed
            green_intensity = 1.0 - 0.5 * intensity  # Vary green from 1 (yellow) to 0.5 (orange)
            blue_intensity = 0.0  # Blue channel fixed at 0 for yellow/orange tones
            cell_color = (red_intensity, green_intensity, blue_intensity, 1.0)  # RGBA format
            rect = plt.Rectangle((j, i), 1, 1, facecolor=cell_color, edgecolor="gray")
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]), ha="center", va="center", fontsize=12, color="black")

    # Labeling the axes
    ax.set_xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="red", labelpad=20)
    ax.set_ylabel("True Labels", fontsize=12, fontweight="bold", color="red", labelpad=20)
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)

    # Display the plot
    st.pyplot(fig)

    # Display the classification report for the selected model
    st.write(f"### {algorithm} Classification Report:")
    st.dataframe(report_df.style.format(precision=2))

    # Button to show comparison of model accuracies
    if st.button("Compare Model Accuracies"):
        # Only proceed if we have more than one model
        if len(st.session_state["model_reports"]) > 1:
            # Extract model names and accuracy scores
            model_names = [report["model_name"] for report in st.session_state["model_reports"]]
            accuracy_scores = [report["accuracy"] for report in st.session_state["model_reports"]]

            # Create a bar plot for model comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(model_names, accuracy_scores, color='skyblue')
            ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight="bold")
            ax.set_ylabel('Models', fontsize=14, fontweight="bold")
            ax.set_title('Comparison of Model Accuracies', fontsize=16, fontweight="bold")

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("Please train more than one model to see the comparison.")
