import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load and combine the datasets (adjust paths as needed)
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")  # Replace with your file path
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")  # Replace with your file path
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# Load data
combined_df = load_data()

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

    # Get the background color from Streamlit theme (default is light gray)
    bg_color = "#f0f2f6"  # Default light gray for the app background

    # Create a figure with a larger size
    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size for better visualization

    # Ensure the figure background is transparent and matches Streamlit's background
    fig.patch.set_facecolor(bg_color)  # Set figure background to match app's background color
    fig.patch.set_alpha(0.0)  # Make the figure background transparent

    # Use the 'score' column for coloring
    scores = combined_df['score']
    scatter = ax.scatter(
        combined_df[x_feature], 
        combined_df[y_feature], 
        c=scores, 
        cmap='viridis',  # Color scheme
        alpha=0.7, 
        edgecolors='w',  # White border around points
        s=100,  # Size of the points
        marker='o'  # Circle marker
    )

    # Add a color bar for the score values
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Score", fontsize=12, fontweight='bold')

    # Apply custom background color and style for the plot area
    ax.set_facecolor(bg_color)  # Plot background color
    ax.set_xlabel(x_feature, fontsize=12, fontweight='bold', color='black')  # X-axis label
    ax.set_ylabel(y_feature, fontsize=12, fontweight='bold', color='black')  # Y-axis label
    ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature} (Colored by Score)", fontsize=14, fontweight='bold', color='black')  # Title styling

    # Customize grid style (light dashed lines for better contrast)
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Subtle gridlines

    # Customize tick labels (optional)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(10)
        tick.set_color('darkgray')

    for tick in ax.get_yticklabels():
        tick.set_fontsize(10)
        tick.set_color('darkgray')

    # Remove the top and right spines (optional, cleaner plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to ensure everything fits without a white border
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

