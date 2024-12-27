import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load and combine the datasets (adjust paths as needed)
@st.cache_data
def load_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")  # Replace with your file
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")  # Replace with your file
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

# Generate Scatter Plot
if x_feature and y_feature:
    st.write(f"### Scatter Plot: {x_feature} vs. {y_feature}")
    fig, ax = plt.subplots()

    # Scatter plot with color logic
    if color_by != "None":
        unique_values = combined_df[color_by].unique()
        colors = plt.cm.get_cmap('viridis', len(unique_values)).colors
        colormap = ListedColormap(colors)
        scatter = ax.scatter(
            combined_df[x_feature], 
            combined_df[y_feature], 
            c=combined_df[color_by].astype('category').cat.codes,
            cmap=colormap, alpha=0.7
        )
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_values)))
        cbar.ax.set_yticklabels(unique_values)
        cbar.set_label(color_by)
    else:
        ax.scatter(combined_df[x_feature], combined_df[y_feature], alpha=0.7, color='blue')

    # Labeling
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature}")

    # Display in Streamlit
    st.pyplot(fig)
