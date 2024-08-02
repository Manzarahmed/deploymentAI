import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

st.title('Machine Learning Application')

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + np.random.randn(100) * 2.5
data = pd.DataFrame({'X': X.squeeze(), 'y': y})

# Display the synthetic data
st.write('### Synthetic Data')
st.write(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
score = model.score(X_test, y_test)
st.write('### Analysis Results')
st.write(f'Model Score: {score}')

# Plotting
fig, ax = plt.subplots()
ax.scatter(X, y, label='Data Points')
ax.plot(X, model.predict(X), color='red', label='Fitted Line')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.legend()

st.pyplot(fig)
