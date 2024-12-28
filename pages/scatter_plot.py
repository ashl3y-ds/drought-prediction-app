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
st.title("Scatter Plot to  to display the relationship between two variables")

st.write("Choose the features to plot and optionally color code scatter points.")

# Dropdowns for user to select X-axis and Y-axis features
x_feature = st.selectbox("Select X-axis Feature", options=combined_df.columns)
y_feature = st.selectbox("Select Y-axis Feature", options=combined_df.columns)

# Dropdown for selecting color grouping
color_by = st.selectbox(
    "Optional: Select a Feature to Color Points", 
    options=["None"] + list(combined_df.columns)
)

# Get the background color from Streamlit theme (default is light gray)
bg_color = "#f0f2f6"  # Default light gray for the app background

# Create a figure with a larger size
fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size for better visualization

# Ensure the figure background is transparent and matches Streamlit's background
fig.patch.set_facecolor(bg_color)  # Set figure background to match app's background color
fig.patch.set_alpha(0.0)  # Make the figure background transparent

if color_by != "None":
    unique_values = combined_df[color_by].unique()
    colors = plt.cm.get_cmap('viridis', len(unique_values)).colors  # Color scheme 'viridis'
    colormap = ListedColormap(colors)
    scatter = ax.scatter(
        combined_df[x_feature], 
        combined_df[y_feature], 
        c=combined_df[color_by].astype('category').cat.codes,  # Assign category codes to colors
        cmap=colormap, 
        alpha=0.7, 
        edgecolors='w',  # White border around points
        s=100,  # Size of the points
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
        edgecolors='w',  # White border around points
        s=100,  # Set point size
        marker='o'  # Circle marker
    )

# Apply custom background color and style for the plot area
ax.set_facecolor(bg_color)  # Plot background color
ax.set_xlabel(x_feature, fontsize=12, fontweight='bold', color='black')  # X-axis label
ax.set_ylabel(y_feature, fontsize=12, fontweight='bold', color='black')  # Y-axis label
ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature}", fontsize=14, fontweight='bold', color='black')  # Title styling

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
