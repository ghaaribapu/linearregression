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
    x_axis = st.selectbox('Select the X-axis:', columns)
    y_axis = st.selectbox('Select the Y-axis:', columns)

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
    st.image("https://example.com/path_to_ball_in_canyon_image.jpg", caption="A ball in a curved canyon")

    st.markdown("""
    Imagine a canyon shaped like a bowl. The bottom of the bowl represents the equilibrium position, where potential energy is at its lowest. If you place a ball anywhere on the curve of the canyon, it will roll down to the bottom due to gravity, finding the lowest point.

    In linear regression, the line of best fit represents this lowest point of error. The slope (m) and intercept (c) determine the angle and position of this line. The distance the ball rolls to reach the bottom represents the errors in prediction. 

    By adjusting the slope and intercept, we can change the line of best fit to minimize these errors, just like how a ball will naturally find the lowest point in a canyon.
    """)

    # 3D visualization of the parameter space
    m_range = np.linspace(m_best - 5, m_best + 5, 100)
    c_range = np.linspace(c_best - 5, c_best + 5, 100)
    M, C = np.meshgrid(m_range, c_range)

    # Calculate SSE for the surface
    SSE = np.array([[mean_squared_error(y, m * X + c) * len(y) for m in m_range] for c in c_range])

    fig_3d = go.Figure(data=[go.Surface(z=SSE, x=M, y=C, colorscale='Viridis')])
    fig_3d.update_layout(title='Surface Plot of SSE',
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
