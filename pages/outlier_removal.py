import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Check if the combined data is available
if "combined_df" not in st.session_state:
    st.error("No combined data found! Please go back to the upload page to load the data.")
else:
    # Use the combined data from session state
    combined_df = st.session_state["combined_df"]

    # Exclude specific columns
    columns_to_exclude = ["date", "fips", "score", "day", "month", "year"]
    numeric_features = [
        col for col in combined_df.columns 
        if col not in columns_to_exclude and pd.api.types.is_numeric_dtype(combined_df[col])
    ]

    # Function to remove outliers
    def remove_outliers(df, features):
        for feature in features:
            mean = df[feature].mean()
            std = df[feature].std()
            df = df[(df[feature] <= mean + 3 * std) & (df[feature] >= mean - 3 * std)]
        return df

    # Streamlit UI for outlier removal
    st.title("Outlier Removal for Dataset Features")
    st.write("This page allows you to remove outliers using the 3-sigma rule for selected features.")

    # Dropdown for selecting features to clean
    selected_features = st.multiselect(
        "Select Features for Outlier Removal (Default: All Eligible Features):", 
        options=numeric_features, 
        default=numeric_features
    )

    # Remove outliers on button click
    if st.button("Remove Outliers"):
        cleaned_df = remove_outliers(combined_df.copy(), selected_features)
        st.success(f"Outliers removed! Total rows after cleaning: {len(cleaned_df)}")
        
        # Save cleaned data in session state for other app sections
        st.session_state["cleaned_df"] = cleaned_df
        st.write(cleaned_df)  # Display cleaned data

    else:
        st.write("No outliers removed yet.")

# Check if cleaned data is available in session state
if 'cleaned_df' not in st.session_state:
    st.error("Please load the data and remove outliers first.")
else:
    # Use cleaned_df for correlation analysis
    cleaned_df = st.session_state["cleaned_df"]

    # List of features to calculate correlation
    measures_column_list = ['PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN', 'T2M_RANGE',
                            'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE', 'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE']

    # Calculate correlation matrix on the cleaned data
    correlation_matrix = cleaned_df[measures_column_list].corr()

    # Streamlit title
    st.title("Feature Correlation Analysis on Cleaned Data")

    # Display correlation matrix as a table
    st.write("### Correlation Matrix (Cleaned Data)")
    st.write(correlation_matrix)

    # Create a heatmap plot of the correlation matrix
    st.write("### Correlation Heatmap")

    # Set up matplotlib figure and plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})

    # Set title and labels
    plt.title("Correlation Heatmap of Selected Features")

    # Display the plot in Streamlit
    st.pyplot(plt)
