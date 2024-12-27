# For re-use of ranked data in other parts of the app
if "filtered_df" in st.session_state:
    st.write("### Ranked Data Preview:")
    st.write(st.session_state["ranked_data"].head())

      # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
