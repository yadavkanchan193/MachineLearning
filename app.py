import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load dataset
df = pd.read_csv('house_price_prediction.csv')

# Preprocessing
# For simplicity, assume all features are numeric and there are no missing values
X = df.drop('price', axis=1)
y = df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Create Streamlit interface
st.title('Housing Price Prediction')

st.header('Enter House Details')

# User input for house features
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
balconies = st.number_input('Number of Balconies', min_value=1, max_value=10, value=2)
house_age = st.number_input('House Age(Year)', min_value=1, max_value=10, value=2)
sqft_living = st.number_input('SquareFeet Area', min_value=1000, value=1500,max_value=3000,)

# Make prediction
input_data = [[house_age, balconies, bedrooms,  sqft_living]]
predicted_price = model.predict(input_data)[0]

st.header('Predicted Price')
st.write(f'The predicted price for the house is {predicted_price:,.2f}')

# Display model evaluation
st.header('Model Evaluation')
st.write(f'Mean Squared Error: {mse:.2f}')

# Plotting
st.header('Data Visualization')

sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');

# Scatter plot of square footage vs. price
fig, ax = plt.subplots()
ax.scatter(df['area'], df['price'], alpha=0.5)
ax.set_xlabel('Square Footage of Living Area')
ax.set_ylabel('Price')
ax.set_title('Square Footage vs. Price')
st.pyplot(fig)



# Scatter plot of number of bedrooms vs. price
fig, ax = plt.subplots()
ax.scatter(df['bedrooms'], df['price'], alpha=0.5)
ax.set_xlabel('Number of Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Number of Bedrooms vs. Price')
st.pyplot(fig)

# Scatter plot of number of bathrooms vs. price
fig, ax = plt.subplots()
ax.scatter(df['balconies'], df['price'], alpha=0.5)
ax.set_xlabel('Number of Balconies')
ax.set_ylabel('Price')
ax.set_title('Number of Balconies vs. Price')
st.pyplot(fig)
