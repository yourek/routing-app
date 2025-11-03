import streamlit as st

slider_key = "my_slider"

# The slider reflects the current session state
slider_value = st.slider(
    "Time in Store",
    min_value=0,
    max_value=100,
    value=50,
    key=slider_key
)

# Initialize default
if slider_key not in st.session_state:
    st.session_state[slider_key] = 50

def estimate_and_update():
    new_val = 75  # your calculation
    st.session_state[slider_key] = new_val
    st.rerun()

# Button to perform calculation and update
st.button("Estimate Value", on_click=estimate_and_update)

st.write("Slider is at:", slider_value)