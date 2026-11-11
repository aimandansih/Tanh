
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

# Plot graph
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, y, label="Tanh", linewidth=2)
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)

ax.set_title("Tanh Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-1.2, 1.2)
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.subheader("Explanation")
st.write("""
Tanh converts input values into an output range
between -1 and 1.

It produces an S-shaped curve and is zero-centred,
making it useful in neural network hidden layers.
""")
