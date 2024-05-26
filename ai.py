import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras

# Load the advertising dataset
@st.cache_data
def load_data():
    data = pd.read_csv("advertising.csv")
    return data

# Model training and evaluation
def sales_prediction(data):
    st.subheader("Annual Sales Prediction")
    st.write("Select variables to predict sales:")
    features = st.multiselect("Select features", ["TV", "Radio", "Newspaper"])

    if len(features) > 0:
        X = data[features]
        y = data["Sales"]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build the neural network model
        model = keras.Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=(len(features),)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")

        # Input form for predicting sales
        st.subheader("Predict Sales")
        st.write("Enter values for the selected features:")
        input_features = {}
        for feature in features:
            input_features[feature] = st.number_input(feature, value=0.0)

        if st.button("Predict"):
            input_data = np.array([[input_features[feature] for feature in features]])
            input_data_scaled = scaler.transform(input_data)
            predicted_sales = model.predict(input_data_scaled)
            st.write(f"Predicted Sales: {predicted_sales[0][0]}")

def main():
    st.title("Sales Prediction with Neural Networks")
    data = load_data()

    # Sidebar
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select option", ["Sales Prediction"])

    if option == "Sales Prediction":
        sales_prediction(data)

if __name__ == "__main__":
    main()
