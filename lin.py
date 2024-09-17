import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import cm

# Set up streamlit app
st.title("Linear Regression Visualizer")

# Step 1: Upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Step 2: Load the dataset
    data = pd.read_csv(uploaded_file)

    # Step 3: Allow the user to select x and y axes from the dataset
    columns = data.columns.tolist()
    x_axis = st.selectbox('Select the X-axis (Independent Variable):', columns)
    y_axis = st.selectbox('Select the Y-axis (Dependent/Target Variable):', columns)

    # Prepare the data for Linear Regression
    X = data[[x_axis]].values
    y = data[y_axis].values

    # Step 4: Train a Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    m_best = model.coef_[0]
    c_best = model.intercept_

    # Step 5: Create a meshgrid for the parameter space (m and c)
    m_range = np.linspace(m_best - 5, m_best + 5, 100)
    c_range = np.linspace(c_best - 5, c_best + 5, 100)
    M, C = np.meshgrid(m_range, c_range)

    # Step 6: Compute the sum of squared errors (SSE) for each combination of m and c
    def compute_sse(m, c):
        y_pred = m * X + c
        sse = mean_squared_error(y, y_pred) * len(y)
        return sse

    SSE = np.array([[compute_sse(m, c) for m in m_range] for c in c_range])

    # Step 7: Plot the 3D heatmap of the parameter space (m, c, SSE)
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.scatter(X, y, label='Data Points')
    ax1.plot(X, model.predict(X), color='red', label=f'Best Fit Line: y={m_best:.2f}x + {c_best:.2f}')
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    ax1.legend()
    ax1.title.set_text('Linear Regression')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(M, C, SSE, cmap=cm.coolwarm, edgecolor='none')
    ax2.set_xlabel('m (slope)')
    ax2.set_ylabel('c (intercept)')
    ax2.set_zlabel('SSE')
    ax2.title.set_text('Parameter Space')

    # Step 8: Interactive slider to adjust m and c and see its effect
    m_slider = st.slider('Adjust slope (m)', min_value=float(m_range.min()), max_value=float(m_range.max()), value=float(m_best), step=0.01)
    c_slider = st.slider('Adjust intercept (c)', min_value=float(c_range.min()), max_value=float(c_range.max()), value=float(c_best), step=0.01)

    # Step 9: Plot the adjusted regression line
    y_pred_adjusted = m_slider * X + c_slider
    ax1.plot(X, y_pred_adjusted, color='green', linestyle='--', label=f'Adjusted Line: y={m_slider:.2f}x + {c_slider:.2f}')
    ax1.legend()

    # Display the updated plot
    st.pyplot(fig)

    # Step 10: Residuals Plot
    residuals = y - model.predict(X)
    fig_residuals = plt.figure(figsize=(8, 4))
    plt.scatter(X, residuals, color='blue', label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel(x_axis)
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.legend()
    st.pyplot(fig_residuals)

    # Step 11: Contour Plot of SSE
    fig_contour = plt.figure(figsize=(8, 5))
    plt.contourf(M, C, SSE, levels=50, cmap='coolwarm')
    plt.colorbar(label='SSE')
    plt.xlabel('m (slope)')
    plt.ylabel('c (intercept)')
    plt.title('Contour Plot of SSE')
    st.pyplot(fig_contour)

    # Step 12: 3D Visualization of Data and Fit
    fig_3d = plt.figure(figsize=(10, 7))
    ax3 = fig_3d.add_subplot(111, projection='3d')
    ax3.scatter(X, y, zs=0, zdir='y', label='Data Points', alpha=0.5)
    ax3.plot_surface(X, model.predict(X), zs=0, zdir='y', color='red', alpha=0.5, label='Best Fit Plane')
    ax3.set_xlabel(x_axis)
    ax3.set_ylabel(y_axis)
    ax3.set_zlabel('Predicted Values')
    ax3.title.set_text('3D Linear Regression Fit')
    st.pyplot(fig_3d)

    # Additional Physics Explanation
    st.markdown("""
    ### Physics Explanation
    - The slope (m) can be thought of as the angle of a ramp, affecting how steeply the dependent variable changes with respect to the independent variable.
    - The intercept (c) represents the starting height of the ramp. Adjusting these parameters allows us to see how different angles and heights affect the data points.
    - The residuals plot shows how well our model fits the data; ideally, residuals should be randomly distributed around zero.
    - The contour plot illustrates the SSE landscape, where lower regions indicate better fitting parameters, akin to a ball rolling to the lowest point in a gravitational field.
    """)
