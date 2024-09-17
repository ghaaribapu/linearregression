import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set up streamlit app
st.title("Interactive Linear Regression Visualizer")

# Step 1: Upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Step 2: Load the dataset
    data = pd.read_csv(uploaded_file)

    # Step 3: Allow the user to select x and y axes from the dataset
    columns = data.columns.tolist()
    x_axis = st.selectbox('Select the X-axis (Independent Variable):', columns)
    y_axis = st.selectbox('Select the Y-axis (Dependent Variable):', columns)

    # Prepare the data for Linear Regression
    X = data[[x_axis]].values
    y = data[y_axis].values

    # Step 4: Train a Linear Regression model
    model = LinearRegression()
    model.fit(X, y)
    m_best = model.coef_[0]
    c_best = model.intercept_

    # Create the best fit line
    y_fit = model.predict(X)

    # Plot using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Data Points', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X.flatten(), y=y_fit, mode='lines', name='Best Fit Line', line=dict(color='red')))
    
    fig.update_layout(title='Linear Regression',
                      xaxis_title=x_axis,
                      yaxis_title=y_axis)

    st.plotly_chart(fig)

    # Interactive physics analogy: ball on a curve
    st.markdown("### Physics Analogy: The Ball in a Canyon")
    st.image("ball.png", caption="A ball in a curved canyon")

    st.markdown("""
    Imagine a canyon shaped like a bowl. The lowest point of this bowl represents the equilibrium position, where potential energy is at its minimum. If you place a ball anywhere on the curve of the canyon, it will roll down to this lowest point due to gravity.

    In linear regression, the goal is to find the best fit line that minimizes the distance between the actual data points and the line itself. This distance is akin to how far the ball is from the lowest point in the canyon. 

    - **Slope (m):** The slope of the line determines how steep the ramp into the canyon is. A steeper slope indicates a quicker change in the y values for small changes in x.

    - **Intercept (c):** The intercept represents the height at which the ramp starts. Changing the intercept moves the entire line up or down without affecting its angle.

    The process of adjusting the slope and intercept to minimize error can be likened to tilting the ramp or adjusting its starting height to help the ball roll to the bottom more easily. 

    When the line of best fit is correctly positioned, it minimizes the errors—much like how the ball finds its natural resting place at the bottom of the canyon.

    The line of best fit has the equation: **y = {m_best:.2f}x + {c_best:.2f}**, representing the relationship between the variables in your dataset.

    ### Deeper Understanding of Linear Regression
    Linear regression is a statistical method that models the relationship between a dependent variable (y) and one or more independent variables (x). The primary goal is to find a linear equation that predicts the value of y based on x.

    - **Why Linear Regression?** Linear regression is used because it is simple and effective for many applications. It assumes that the relationship between the variables is linear, which means that as x changes, y changes in a predictable manner.

    - **Error Minimization:** In our analogy, the "error" is the vertical distance between the data points and the regression line. The goal is to adjust the slope and intercept to minimize these errors, ensuring the line is as close to all the points as possible. The line of best fit essentially represents the average trend of the data.

    - **Application:** This concept is widely used in various fields such as economics, biology, engineering, and social sciences, where predicting outcomes based on input variables is essential.
    """)

    # 3D visualization of the parameter space
    m_range = np.linspace(-5, 5, 100)
    c_range = np.linspace(-5, 5, 100)
    M, C = np.meshgrid(m_range, c_range)

    # Calculate SSE for the surface
    SSE = np.array([[mean_squared_error(y, m * X + c) * len(y) for m in m_range] for c in c_range])

    # 3D Surface Plot with Ball Position
    fig_3d = go.Figure(data=[go.Surface(z=SSE, x=M, y=C, colorscale='Viridis')])

    # Initial ball position based on best fit
    ball_position_x = m_best
    ball_position_y = c_best
    ball_position_z = mean_squared_error(y, model.predict(X)) * len(y)  # SSE at the best fit

    # Add the ball's position as a point
    fig_3d.add_trace(go.Scatter3d(
        x=[ball_position_x],
        y=[ball_position_y],
        z=[ball_position_z],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Ball Position (Best Fit)'
    ))

    fig_3d.update_layout(title='Surface Plot of SSE with Ball Position',
                         scene=dict(
                             xaxis_title='Slope (m)',
                             yaxis_title='Intercept (c)',
                             zaxis_title='SSE'),
                         autosize=True)
    
    st.plotly_chart(fig_3d)

    # Sliders for controlling ball position on all axes
    st.markdown("### Move the Ball to Adjust the Fit")
    ball_position_x = st.slider('Adjust Slope (m)', min_value=-5.0, max_value=5.0, value=float(m_best), step=0.01)
    ball_position_y = st.slider('Adjust Intercept (c)', min_value=-5.0, max_value=5.0, value=float(c_best), step=0.01)
    ball_position_z = st.slider('Adjust SSE', min_value=0.0, max_value=float(SSE.max()), value=float(ball_position_z), step=0.1)

    # Calculate new line based on ball position
    new_m = ball_position_x
    new_c = ball_position_y
    new_y_fit = new_m * X + new_c

    fig_adjusted = go.Figure()
    fig_adjusted.add_trace(go.Scatter(x=X.flatten(), y=y, mode='markers', name='Data Points', marker=dict(color='blue')))
    fig_adjusted.add_trace(go.Scatter(x=X.flatten(), y=new_y_fit, mode='lines', name='Adjusted Fit', line=dict(color='green')))
    
    fig_adjusted.update_layout(title='Adjusted Fit Based on Ball Position',
                                xaxis_title=x_axis,
                                yaxis_title=y_axis)

    st.plotly_chart(fig_adjusted)

    # Display the line of best fit values
    st.markdown(f"### Line of Best Fit: y = {m_best:.2f}x + {c_best:.2f}")
