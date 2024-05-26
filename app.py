import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
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

# Data profiling
def data_profiling(data):
    st.subheader("Data Profiling")
    st.write("Dataset Information:")
    st.write("Dataset from 2018-2023")
    st.write("Null Values:")
    st.write(data.isnull().sum())
    st.write("Correlation Matrix:")
    st.write(data.corr())
    st.write("Descriptive Statistics:")
    st.write(data.describe())
    
    # Distribution plots for each feature
    st.write("Distribution Plots:")
    for column in data.columns:
        if column != 'Sales':
            st.write(f"**{column}**")
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

    st.write("Correlation Heatmap:")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax)
    st.pyplot(fig)

# Outlier detection
def outlier_detection(data):
    st.subheader("Outlier Detection")
    for column in data.columns:
        st.write(column)
        fig, ax = plt.subplots()
        sns.boxplot(data[column], ax=ax)
        st.pyplot(fig)

# Model training and evaluation
def model_training_evaluation(data):
    st.subheader("Model Training and Evaluation")
    # Split data
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=60)
    
    # Model training
    LE = LinearRegression()
    LE.fit(xtrain, ytrain)
    
    # Model evaluation
    score = LE.score(xtest, ytest)
    st.write("Model Score:", score)

# Standardization and retraining
def standardization_retraining(data):
    st.subheader("Standardization and Retraining")
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=60)
    
    # Standardization
    sc = StandardScaler()
    xtrain_sc = sc.fit_transform(xtrain)
    xtest_sc = sc.transform(xtest)
    
    # Retraining
    LE = LinearRegression()
    LE.fit(xtrain_sc, ytrain)
    
    # Model evaluation
    score = LE.score(xtest_sc, ytest)
    st.write("Model Score after Standardization:", score)

# Admin login
def admin_login():
    st.subheader("Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":
            st.success("Login successful!")
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")

# Money division and vendor booking
def money_division_and_booking():
    st.subheader("Money Division and Vendor Booking")
    streams = st.multiselect("Select streams of revenue", ["Advertising", "Sponsorship", "Merchandising", "Events"])
    total_budget = st.number_input("Enter total budget")
    if len(streams) > 0:
        per_stream_budget = total_budget / len(streams)
        st.write(f"Allocate {per_stream_budget} to each stream")
        for stream in streams:
            st.write(f"Booking vendors for {stream}")
            # Here you can add code for booking vendors based on the selected stream
            # For simplicity, let's just print the message
            st.write(f"Vendors booked for {stream}")

        # Suggestions based on selected streams
        st.subheader("Suggestions")
        if "Advertising" in streams:
            st.write("Consider investing in online ads, social media campaigns, and influencer marketing.")
        if "Sponsorship" in streams:
            st.write("Explore sponsorship opportunities for events, sports teams, or community programs.")
        if "Merchandising" in streams:
            st.write("Create branded merchandise such as clothing, accessories, or collectibles.")
        if "Events" in streams:
            st.write("Host or sponsor events related to your industry or target audience.")
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

        # Calculate accuracy in terms of percentage
        accuracy_percentage = 100-mse-r2

        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        st.write(f"Accuracy (Percentage): {accuracy_percentage:.2f}%")

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
    st.title("Advertising Revenue Management")
    data = load_data()

    # Check if logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Sidebar
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Select option", ["Data Profiling", "Outlier Detection", "Model Training and Evaluation",
                                                     "Standardization and Retraining", "Admin","Sales Prediction"])

    if option == "Data Profiling":
        data_profiling(data)

    elif option == "Outlier Detection":
        outlier_detection(data)

    elif option == "Model Training and Evaluation":
        model_training_evaluation(data)

    elif option == "Standardization and Retraining":
        standardization_retraining(data)

    elif option == "Admin":
        if not st.session_state.logged_in:
            admin_login()
        else:
            money_division_and_booking()
    elif option == "Sales Prediction":
        sales_prediction(data)

if __name__ == "__main__":
    main()
