import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")  # Replace with your file path
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")  # Replace with your file path
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load data
combined_df = load_data()

# Convert 'date' column to datetime format (ensure the 'date' column exists in the dataset)
if 'date' in combined_df.columns:
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['month'] = combined_df['date'].dt.month
else:
    st.error("The dataset must include a 'date' column for time-based visualizations.")

# Check if "score" column exists
if "score" not in combined_df.columns:
    st.error("The dataset does not contain a 'score' column. Please ensure the data includes this column.")
else:
    # Scatter Plot Generator
    st.title("Scatter Plot to Display the Relationship Between Two Variables")
    st.write("Choose the features to plot. The points will be colored based on the 'score' column.")

    # Dropdowns for user to select X-axis and Y-axis features
    x_feature = st.selectbox("Select X-axis Feature", options=combined_df.columns)
    y_feature = st.selectbox("Select Y-axis Feature", options=combined_df.columns)

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scores = combined_df['score']
    scatter = ax.scatter(
        combined_df[x_feature],
        combined_df[y_feature],
        c=scores,
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=100
    )
    plt.colorbar(scatter, ax=ax).set_label("Score")
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature} (Colored by Score)")
    st.pyplot(fig)

    # Line Graph: Feature Trends Over Time
    st.title("Line Graph for Feature Trends Over Time")

    # Select a feature for line graph
    feature = st.selectbox("Select a Feature for Trend Visualization", options=combined_df.columns)

    # Group data by month and calculate average for the selected feature
    if feature in combined_df.columns and 'month' in combined_df.columns:
        trend = combined_df.groupby('month')[feature].mean()

        # Plot line graph
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trend.index, trend.values, marker='o', color='blue', linestyle='-', linewidth=2)
        ax.set_title(f"Average {feature} by Month", fontsize=14)
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel(f"{feature} (Average)", fontsize=12)
        ax.set_xticks(range(1, 13))  # Show all 12 months on the x-axis
        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray')
        st.pyplot(fig)
    else:
        st.error("Unable to generate trend visualization. Ensure the dataset includes valid time-related data.")

