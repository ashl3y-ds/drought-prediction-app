import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

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

# Check if the cleaned data is available in session state
if 'cleaned_df' not in st.session_state:
    st.error("Please load the data and remove outliers first.")
else:
    # Use cleaned_df for further analysis
    cleaned_df = st.session_state["cleaned_df"]

    # Define independent variables (exclude 'score', 'fips', 'date') and target variable ('score')
    independent_variables = cleaned_df.drop(['score', 'fips', 'date'], axis=1)
    target = cleaned_df['score']
    
    # Save them to session state
    st.session_state["independent_variables"] = independent_variables
    st.session_state["target"] = target

    # Display the independent variables
    st.write("### Independent Variables (First 5 rows):")
    st.write(independent_variables.head())

    # Display the target variable
    st.write("### Target Variable (Score):")
    st.write(target.head())

    # Confirm that the variables are saved
    st.success("Independent Variables and Target Variable saved to session state.")

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

    plt.figure(figsize=(12, 8))

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8})

    # Set title and labels
    plt.title("Correlation Heatmap of Selected Features")

    # Display the plot in Streamlit
    st.pyplot(plt)


# Check if the independent variables and target are saved in session state
if 'independent_variables' not in st.session_state or 'target' not in st.session_state:
    st.error("Please load and preprocess the data first on the previous pages.")
else:
    # Load the saved independent variables and target variable
    independent_variables = st.session_state["independent_variables"]
    target = st.session_state["target"]

    # Streamlit title
    st.title("Feature Selection with RFE (Recursive Feature Elimination)")

    # Perform train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(independent_variables, target, test_size=0.2, random_state=0)
    
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

        # Save the split and scaled data
    with open('split_scaled_data.pkl', 'wb') as f:
        pickle.dump({
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler
        }, f)

    # Create and fit the RandomForest model
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    rfe = RFE(model, n_features_to_select=16)  # Trial and error for n_features_to_select
    
    # Fit RFE on the training data
    fit = rfe.fit(X_train, y_train)

    # Display the results
    st.write(f"### Selected Features:")
    selected_features = independent_variables.columns[fit.support_].to_list()
    st.write(f"Selected Features: {selected_features}")
    

    st.write(f"### Number of Features Selected: {fit.n_features_}")
    
    # Optionally, show the rank of all features to show how well each was ranked for selection
    feature_rank_table = pd.DataFrame({
        'Feature': independent_variables.columns,
        'Rank': fit.ranking_
    })
    feature_rank_table = feature_rank_table.sort_values('Rank')
    st.write("### Features Sorted by Rank:")
    st.write(feature_rank_table)

    # Button to remove non-ranked 1 features and save the filtered dataset
    if st.button("Remove Non-Rank 1 Features"):
        # Remove the non-rank 1 features from the dataset
        filtered_df = independent_variables[selected_features]
        
        # Display the filtered dataframe
        st.write("### Filtered Dataframe with Only Rank 1 Features")
        st.write(filtered_df)
        
        # Save the filtered dataframe in session state for later use
        st.session_state["filtered_df"] = filtered_df

        st.success("Non-rank 1 features removed successfully!")

    else:
        st.write("Click the button to remove features with rank higher than 1.")
