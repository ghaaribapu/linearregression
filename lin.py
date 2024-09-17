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
    Imagine a canyon shaped like a bowl. The lowest point of this bowl represents the equilibrium position, where the potential energy is at its minimum. If you place a ball anywhere on the curve of the canyon, it will roll down to this lowest point due to gravity. 

    In linear regression, the goal is to find the best fit line that minimizes the distance between the actual data points and the line itself. This distance is similar to how far the ball is from the lowest point in the canyon. 

    The slope (m) of the line determines how steep the ramp into the canyon is, while the intercept (c) represents the height at which the ramp starts. By adjusting the slope and intercept, we can move the line to better fit the data, just like how you might tilt the ramp to help the ball reach the bottom more easily. 

    When the line of best fit is correctly positioned, it minimizes the errors—much like how the ball finds its natural resting place at the bottom of the canyon.
    """)

    # 3D visualization of the parameter space
    m_range = np.linspace(m_best - 5, m_best + 5, 100)
    c_range = np.linspace(c_best - 5, c_best + 5, 100)
    M, C = np.meshgrid(m_range, c_range)

    # Calculate SSE for the surface
    SSE = np.array([[mean_squared_error(y, m * X + c) * len(y) for m in m_range] for c in c_range])

    # 3D Surface Plot with Ball Position
    fig_3d = go.Figure(data=[go.Surface(z=SSE, x=M, y=C, colorscale='Viridis')])
    
    # Position of the ball on the curve
    ball_position = m_best, c_best
    ball_sse = mean_squared_error(y, model.predict(X)) * len(y)  # SSE at the best fit
    
    # Add the ball's position as a point
    fig_3d.add_trace(go.Scatter3d(
        x=[ball_position[0]],
        y=[ball_position[1]],
        z=[ball_sse],
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

    # Additional interactive feature to move the ball
    st.markdown("### Move the Ball to Adjust the Fit")
    ball_position = st.slider('Adjust Ball Position (on the X-axis)', min_value=float(X.min()), max_value=float(X.max()), value=float(X.mean()), step=0.1)
    
    # Calculate new line based on ball position
    new_m = (ball_position - c_best) / (X.max() - X.min())
    new_c = c_best
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
