import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Load dataset with caching to improve performance
@st.cache_data
def load_data():
    """Load the California Housing Dataset from a CSV file."""
    df = pd.read_csv('./data/housing.csv')
    return df

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Load the pre-trained XGBoost model from a pickle file."""
    return joblib.load('xgb_housing_model.pkl')

# Data Exploration
def explore_data(df):
    """Perform exploratory data analysis and visualize key statistics.
    Displays dataset shape, summary statistics, missing values, feature distributions,
    correlation matrix (for numeric columns only), and relationships between median_income
    and median_house_value."""
    st.subheader("Data Exploration")
    st.write("Dataset Shape:", df.shape)
    st.write("First 5 Rows:")
    st.dataframe(df.head())
    
    # Summary statistics
    st.write("Summary Statistics:")
    st.dataframe(df.describe())
    
    # Missing values
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum())
    
    # Feature distributions
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        st.pyplot(plt)
    
    # Correlation matrix (only for numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix (Numeric Features Only)')
    st.pyplot(plt)
    
    # Scatter plot: median_income vs median_house_value
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['median_income'], y=df['median_house_value'])
    plt.title('Median Income vs Median House Value')
    st.pyplot(plt)

# Prediction Function
def make_prediction(model, input_data):
    """Make a prediction using the pre-trained model and input data."""
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit UI
def main():
    """Main function to run the Streamlit application with navigation between pages."""
    st.title("California Housing Price Prediction")
    
    # Load data and model
    df = load_data()
    model = load_model()
    
    # Sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Select Page", ["Data Exploration", "Prediction"])
    
    if page == "Data Exploration":
        explore_data(df)
    
    elif page == "Prediction":
        st.subheader("Make a Prediction")
        
        # User input for prediction
        longitude = st.number_input("Longitude", value=-122.23)
        latitude = st.number_input("Latitude", value=37.88)
        housing_median_age = st.number_input("Housing Median Age", value=41.0)
        total_rooms = st.number_input("Total Rooms", value=880.0)
        total_bedrooms = st.number_input("Total Bedrooms", value=129.0)
        population = st.number_input("Population", value=322.0)
        households = st.number_input("Households", value=126.0)
        median_income = st.number_input("Median Income", value=8.3252)
        ocean_proximity = st.selectbox("Ocean Proximity", ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'])
        
        # Create input DataFrame for prediction
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity],
            'rooms_per_household': [total_rooms / households],
            'bedrooms_per_room': [total_bedrooms / total_rooms],
            'population_per_household': [population / households],
            'distance_to_coast': [np.sqrt((longitude - (-122))**2 + (latitude - 37)**2)],
            'income_per_person': [median_income / (population / households)]
        })
        
        # Make prediction
        if st.button("Predict"):
            prediction = make_prediction(model, input_data)
            st.write(f"Predicted Median House Value: ${prediction:,.2f}")

if __name__ == "__main__":
    main()