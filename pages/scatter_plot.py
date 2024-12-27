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

# Scatter Plot Generator
st.title("Interactive Scatter Plot with Color Coding")

st.write("Choose the features to plot and optionally color code scatter points.")

# Dropdowns for user to select X-axis and Y-axis features
x_feature = st.selectbox("Select X-axis Feature", options=combined_df.columns)
y_feature = st.selectbox("Select Y-axis Feature", options=combined_df.columns)

# Dropdown for selecting color grouping
color_by = st.selectbox(
    "Optional: Select a Feature to Color Points", 
    options=["None"] + list(combined_df.columns)
)

# Custom Style (manual customization)
# Customize the plot colors and background
fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size for better visualization

if color_by != "None":
    unique_values = combined_df[color_by].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_values)).colors  # Color scheme 'viridis'
    colormap = ListedColormap(colors)
    scatter = ax.scatter(
        combined_df[x_feature], 
        combined_df[y_feature], 
        c=combined_df[color_by].astype('category').cat.codes,  # Assigning category codes to each color
        cmap=colormap, 
        alpha=0.7, 
        edgecolors='w',  # White border around scatter points
        s=100,  # Size of points
        marker='o'  # Circle marker
    )
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_values)))
    cbar.ax.set_yticklabels(unique_values)
    cbar.set_label(color_by)
else:
    ax.scatter(
        combined_df[x_feature], 
        combined_df[y_feature], 
        alpha=0.7, 
        color='blue', 
        edgecolors='w',  # White border
        s=100,  # Set point size
        marker='o'  # Circle marker
    )

# Apply custom background and style to grid, labels, title, and axes
ax.set_facecolor('#f2f2f2')  # Light grey background
ax.set_xlabel(x_feature, fontsize=12, fontweight='bold', color='black')  # X-axis label styling
ax.set_ylabel(y_feature, fontsize=12, fontweight='bold', color='black')  # Y-axis label styling
ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature}", fontsize=14, fontweight='bold', color='black')  # Title styling

# Customize the grid style (light dashed lines for better contrast)
ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)  # Gridline styling

# Customize tick labels (optional)
for tick in ax.get_xticklabels():
    tick.set_fontsize(10)
    tick.set_color('darkgray')
    
for tick in ax.get_yticklabels():
    tick.set_fontsize(10)
    tick.set_color('darkgray')

# Customize the legend (color mapping) and plot readability
ax.legend(loc='upper left', fontsize=10)  # Position the legend at the top-left

# Show plot
st.pyplot(fig)

