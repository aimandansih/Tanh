
import streamlit as st
import numpy as np
import pandas as pd

# Streamlit page configuration
st.set_page_config(
    page_title="Tanh Activation Function",
    layout="wide"
)

st.title("Tanh Activation Function")
st.write("Visualisation of Hyperbolic Tangent (Tanh).")

# Generate input values
x = np.linspace(-6, 6, 400)

# Tanh activation function
def tanh(x):
    return np.tanh(x)

y = tanh(x)

# Display formula
st.subheader("Mathematical Formula")
st.latex(r"f(x) = \tanh(x)")

# Create DataFrame
df = pd.DataFrame({
    "Input (x)": x,
    "Tanh Output": y
})

# Plot graph
st.subheader("Tanh Activation Function Graph")

st.line_chart(
    df.set_index("Input (x)")
)

# Explanation
st.subheader("Explanation")

st.write("""
Tanh converts input values into an output range
between -1 and 1.

It produces an S-shaped curve and is zero-centred,
making it useful in neural network hidden layers.
""")
