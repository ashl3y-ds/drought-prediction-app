import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

        # Set up a larger figure with a black background and purple-pink colormap
        fig, ax = plt.subplots(figsize=(20, 18))  # Increase figure size for larger matrix
        ax.set_facecolor('black')  # Set background color to black
        fig.patch.set_alpha(0.0)

        # Using a purple-pink gradient color palette
        purple_pink_cmap = plt.cm.Spectral  # A good purple-pink transition colormap

        # Plot heatmap with the customized color scheme
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap=purple_pink_cmap,  # Purple-pink colormap
            cbar=True,
            square=True,
            linewidths=1.2,  # Thicker line separating cells
            ax=ax,
            annot_kws={"size": 20, "weight": "bold", "color": "#F5F5F5"},  # Larger annotation text
            cbar_kws={"label": "Correlation Coefficient", 'shrink': 0.8},  # Color bar label and shrink size
            xticklabels=corr_matrix.columns,  # Show feature names on both axes
            yticklabels=corr_matrix.columns  # Show feature names on both axes
        )

        # Increase the size of the cells further by adjusting tick spacing
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))

        # Adjust the font size of the axis labels
        ax.set_xticklabels(corr_matrix.columns, fontsize=16, fontweight='bold', color='white')
        ax.set_yticklabels(corr_matrix.columns, fontsize=16, fontweight='bold', color='white')

        # Title with a larger font size
        ax.set_title("Feature Correlation Heatmap", fontsize=22, fontweight='bold', color='white')

        # Display the plot
        st.pyplot(fig)
    else:
        st.error("No numerical data available to compute correlations.")
