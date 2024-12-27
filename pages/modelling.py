import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For re-use of ranked data in other parts of the app
if "filtered_df" in st.session_state:
    st.write("### Ranked Data Preview:")
    st.write(st.session_state["ranked_data"].head())

      # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # Scale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X_train)
        x_test = scaler.transform(X_test)
