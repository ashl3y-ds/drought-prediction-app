import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")  # Replace with your file path
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")  # Replace with your file path
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Function to generate scatter plot
def generate_scatter_plot(data):
    st.title("Scatter Plot to Display the Relationship Between Two Variables")
    st.write("Choose the features to plot. The points will be colored based on the 'score' column.")
    x_feature = st.selectbox("Select X-axis Feature", options=data.columns, key="scatter_x")
    y_feature = st.selectbox("Select Y-axis Feature", options=data.columns, key="scatter_y")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0.0)
    scores = data['score']
    scatter = ax.scatter(
        data[x_feature],
        data[y_feature],
        c=scores,
        cmap='viridis',
        alpha=0.7,
        edgecolors='w',
        s=100
    )
    plt.colorbar(scatter, ax=ax).set_label("Score", fontsize=12, fontweight='bold', color='red')
    ax.set_xlabel(x_feature, fontsize=12, fontweight='bold', color='red')
    ax.set_ylabel(y_feature, fontsize=12, fontweight='bold', color='red')
    ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature} (Colored by Score)", fontsize=14, fontweight='bold', color='red')

    for label in ax.get_xticklabels():
        label.set_color('red')
    for label in ax.get_yticklabels():
        label.set_color('red')

    st.pyplot(fig)

# Function to generate line graph
def generate_line_graph(data):
    st.title("Line Graph for Feature Trends Over Time")
    feature = st.selectbox("Select a Feature for Trend Visualization", options=data.columns, key="line_feature")
    
    if feature in data.columns and 'month' in data.columns:
        trend = data.groupby('month')[feature].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor((0, 0, 0, 0))
        fig.patch.set_alpha(0.0)
        ax.plot(trend.index, trend.values, marker='o', color='blue', linestyle='-', linewidth=2)
        ax.set_title(f"Average {feature} by Month", fontsize=14, fontweight='bold', color='red')
        ax.set_xlabel("Month", fontsize=12, fontweight='bold', color='red')
        ax.set_ylabel(f"{feature} (Average)", fontsize=12, fontweight='bold', color='red')
        ax.set_xticks(range(1, 13))
        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray')

        for label in ax.get_xticklabels():
            label.set_color('red')
        for label in ax.get_yticklabels():
            label.set_color('red')

        st.pyplot(fig)
    else:
        st.error("Unable to generate trend visualization. Ensure the dataset includes valid time-related data.")


def generate_heatmap(data):
    st.title("Heatmap of Correlation Analysis")
    st.write("This heatmap displays the correlation between all numerical features in the dataset.")
    
    # List of features to exclude from correlation analysis
    exclude_features = ['date', 'fips', 'day', 'month', 'year']
    
    # Remove excluded features
    numerical_data = data.drop(columns=[col for col in exclude_features if col in data.columns])
    numerical_data = numerical_data.select_dtypes(include=["float64", "int64"])

    if not numerical_data.empty:
        corr_matrix = numerical_data.corr()

        # Create a custom blue-to-green colormap with high intensity
        blue_green_cmap = LinearSegmentedColormap.from_list(
            'blue_to_green', ['#0000FF', '#00FF00']  # Transition from blue to green
        )

        # Set up a much larger figure for an enhanced size (20x larger than default)
        fig, ax = plt.subplots(figsize=(80, 72))  # Increased figure size (20x larger than default)
        ax.set_facecolor('black')  # Set background color to black
        fig.patch.set_alpha(0.0)

        # Plot heatmap with blue-to-green colormap and bigger cell size
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=blue_green_cmap,  # Using blue-to-green colormap
            cbar=True,
            square=True,
            linewidths=2,  # Thicker line separating cells for more prominence
            ax=ax,
            annot_kws={"size": 60, "weight": "bold", "color": "#FF0000"},  # Red text for annotation
            cbar_kws={"label": "Correlation Coefficient", 'shrink': 0.75},  # Shrink the color bar slightly
            xticklabels=corr_matrix.columns,  # Show feature names on both axes
            yticklabels=corr_matrix.columns,  # Show feature names on both axes
        )

        # Make cells bigger and adjust tick positions
        ax.set_xticks(np.arange(len(corr_matrix.columns)) + 0.5)  # Moves ticks closer to center of cells
        ax.set_yticks(np.arange(len(corr_matrix.columns)) + 0.5)  # Moves ticks closer to center of cells
        ax.set_xticklabels(corr_matrix.columns, fontsize=56, fontweight='bold', color='red')  # Red font for labels
        ax.set_yticklabels(corr_matrix.columns, fontsize=56, fontweight='bold', color='red')  # Red font for labels

        # Title with a larger and bold font size
        ax.set_title("Feature Correlation Heatmap", fontsize=72, fontweight='bold', color='red')

        # Display the plot
        st.pyplot(fig)
    else:
        st.error("No numerical data available to compute correlations.")
# Main app logic
combined_df = load_data()
if 'date' in combined_df.columns:
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df['month'] = combined_df['date'].dt.month
else:
    st.error("The dataset must include a 'date' column for time-based visualizations.")

if "score" not in combined_df.columns:
    st.error("The dataset does not contain a 'score' column. Please ensure the data includes this column.")
else:
    visualization_choice = st.radio("Select Visualization Type", ["Scatter Plot", "Line Graph", "Heatmap"])

    if visualization_choice == "Scatter Plot":
        generate_scatter_plot(combined_df)
    elif visualization_choice == "Line Graph":
        generate_line_graph(combined_df)
    elif visualization_choice == "Heatmap":
        generate_heatmap(combined_df)
