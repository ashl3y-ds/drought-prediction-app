import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
    cleaned_df = pd.read_csv("data/independent_variables_dataset.csv")  # Replace with your file path
    return cleaned_df

# Load data
cleaned_df = load_data()

# Ensure required data is in session state
if "score" in cleaned_df.columns:

    # Load features and target variable
    X = cleaned_df.drop(columns=["score"])  # Use parentheses and specify 'columns' argument
    y = cleaned_df["score"] 

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save processed data back to session state
    if "X_train" not in st.session_state:
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

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Check if the model has already been added
    existing_model_names = [report["model_name"] for report in st.session_state["model_reports"]]
    if algorithm not in existing_model_names:
        # Add the model report
        st.session_state["model_reports"].append({
            "model_name": algorithm,
            "accuracy": accuracy * 100,
            "classification_report": report_df,
            "confusion_matrix": cm
        })
    else:
        st.warning(f"Model '{algorithm}' is already trained and saved. Train a new model or update existing models.")

    # Display confusion matrix
st.write("### Confusion Matrix")
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_facecolor((0, 0, 0, 0))
fig.patch.set_alpha(0.0)

# Plot the confusion matrix
cax = ax.matshow(cm, cmap='coolwarm', alpha=0.7)

# Display the matrix values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], ha="center", va="center", color="black", fontsize=10)

# Add gridlines by drawing rectangles around each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        rect = plt.Rectangle([j-0.5, i-0.5], 1, 1, fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)

# Add labels and title
plt.xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="red")
plt.ylabel("True Labels", fontsize=12, fontweight="bold", color="red")
plt.title("Confusion Matrix", fontsize=14, fontweight="bold", color="red")

# Customize tick labels
ax.tick_params(axis="x", colors="red", labelsize=10)
ax.tick_params(axis="y", colors="red", labelsize=10)

st.pyplot(fig)

    # Display classification report for the selected model
st.write(f"### {algorithm} Classification Report:")
st.dataframe(report_df.style.format(precision=2))

metric = st.selectbox("Select Metric to Compare", ["Precision", "Recall", "F1-Score", "Accuracy"])

# Compare model metrics
if st.button("Compare Model Metrics"):
    if len(st.session_state["model_reports"]) > 1:
        # Extract unique models and metrics
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
