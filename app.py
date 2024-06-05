import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = {
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]
}
df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['Experience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Streamlit app
st.title('Simple Linear Regression with Streamlit')

st.write("""
## Explore the relationship between Experience and Salary
""")

# Display the data
st.write("### Data", df)

# Plotting the data
fig, ax = plt.subplots()
ax.scatter(df['Experience'], df['Salary'], color='blue', label='Actual Data')
ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
ax.legend()
st.pyplot(fig)

# Display the coefficients
st.write("### Model Coefficients")
st.write(f"Intercept: {model.intercept_}")
st.write(f"Coefficient: {model.coef_[0]}")

# User input for prediction
st.write("### Predict Salary Based on Experience")
experience = st.number_input('Years of Experience', min_value=0, max_value=50, value=5)
predicted_salary = model.predict(np.array([[experience]]))[0]
st.write(f"Predicted Salary: {predicted_salary:.2f}")
