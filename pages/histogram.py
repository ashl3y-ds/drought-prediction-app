import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Title
st.title("Advanced Histogram Analysis")

@st.cache_data
def load_combined_data():
    df1 = pd.read_csv("data/cleaned_drought_data_part1.csv")
    df2 = pd.read_csv("data/cleaned_drought_data_part2.csv")
    return pd.concat([df1, df2], ignore_index=True)

df = load_combined_data()

# Feature selection
feature = st.selectbox("Select a feature for analysis:", ['T2M', 'PRECTOT', 'score'])

# Add threshold filtering
threshold = st.slider(f"Filter {feature} (choose a threshold):", 
                      min_value=float(df[feature].min()), 
                      max_value=float(df[feature].max()),
                      value=float(df[feature].mean()))

# Filter dataset
filtered_df = df[df[feature] >= threshold]

# Plot histogram
fig, ax = plt.subplots()
ax.hist(filtered_df[feature], bins=20, density=True)
ax.set_title(f"{feature} Distribution (Values ≥ {threshold})")
ax.set_xlabel(feature)
ax.set_ylabel('Density')
st.pyplot(fig)

