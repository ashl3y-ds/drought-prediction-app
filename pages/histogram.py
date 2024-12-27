import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
st.title("Interactive Scatter Plot Generator")

st.write("Choose the features to plot a scatter plot based on the combined dataset.")

# Dropdowns for user to select X-axis and Y-axis features
x_feature = st.selectbox("Select X-axis Feature", options=combined_df.columns)
y_feature = st.selectbox("Select Y-axis Feature", options=combined_df.columns)

# Optional color grouping
color_by = st.selectbox(
    "Optional: Select a Feature to Color By (Leave Empty for No Coloring)", 
    options=["None"] + list(combined_df.columns)
)

# Generate Scatter Plot
if x_feature and y_feature:
    st.write(f"### Scatter Plot: {x_feature} vs. {y_feature}")
    fig, ax = plt.subplots()
    
    # Scatter plot logic
    if color_by != "None":
        scatter = ax.scatter(
            combined_df[x_feature], 
            combined_df[y_feature], 
            c=combined_df[color_by], 
            cmap="viridis", alpha=0.7
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(combined_df[x_feature], combined_df[y_feature], alpha=0.7)

    # Labeling
    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(f"Scatter Plot: {x_feature} vs. {y_feature}")

    # Display in Streamlit
    st.pyplot(fig)
