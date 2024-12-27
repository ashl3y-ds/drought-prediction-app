import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Ensure required data is in session state
if "filtered_df" in st.session_state and "target" in st.session_state:
    st.write("### Ranked Data Preview:")
    st.write(st.session_state["filtered_df"].head())

    # Load features and target variable
    X = st.session_state["filtered_df"]
    y = st.session_state["target"]

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

    st.write(f"Number of training samples: {X_train.shape[0]}")
    st.write(f"Number of testing samples: {X_test.shape[0]}")

    # Initialize session state for model reports
    if "model_reports" not in st.session_state:
        st.session_state["model_reports"] = []

    # Algorithm selection
    algorithm = st.selectbox("Select Algorithm", ["Support Vector Machine (SVM)", "Decision Tree", "K-Nearest Neighbors (KNN)", "Random Forest"])

    # Train model based on selected algorithm
    if algorithm == "Support Vector Machine (SVM)":
        model = SVC(kernel='linear', probability=True)
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    elif algorithm == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier()
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")

    # Classification report
    classes = np.unique(y)
    report_dict = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Check if the model has already been added
    existing_model_names = [report["model_name"] for report in st.session_state["model_reports"]]
    if algorithm not in existing_model_names:
        # Add the model report
        st.session_state["model_reports"].append({
            "model_name": algorithm,
            "accuracy": accuracy * 100,
            "classification_report": report_df
        })
    else:
        st.warning(f"Model '{algorithm}' is already trained and saved. Train a new model or update existing models.")

    # Select metric for comparison
    metric = st.selectbox("Select Metric to Compare", ["Precision", "Recall", "F1-Score", "Accuracy"])

    # Compare model metrics
    if st.button("Compare Model Metrics"):
        if len(st.session_state["model_reports"]) > 1:
            # Get unique models and metrics
            unique_model_reports = {
                report["model_name"]: report
                for report in st.session_state["model_reports"]
            }

            model_names = list(unique_model_reports.keys())
            metric_scores = []

            # Extract metric scores for unique models
            for report in unique_model_reports.values():
                if metric == "Precision":
                    metric_scores.append(report["classification_report"].loc["weighted avg", "precision"] * 100)
                elif metric == "Recall":
                    metric_scores.append(report["classification_report"].loc["weighted avg", "recall"] * 100)
                elif metric == "F1-Score":
                    metric_scores.append(report["classification_report"].loc["weighted avg", "f1-score"] * 100)
                elif metric == "Accuracy":
                    metric_scores.append(report["accuracy"])

            # Plot the bar graph
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(model_names, metric_scores, color='skyblue', edgecolor="black")

            # Customize the bar graph
            ax.set_xlabel("Model Names", fontsize=14, fontweight="bold")
            ax.set_ylabel(f"{metric} (%)", fontsize=14, fontweight="bold")
            ax.set_title(f"Comparison of {metric} Across Models", fontsize=16, fontweight="bold")
            ax.set_ylim(0, 100)  # y-axis range: 0 to 100
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, fontsize=12, rotation=45, ha="right")

            # Display the graph
            st.pyplot(fig)
        else:
            st.write("Please train more than one model to see the comparison.")
