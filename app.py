import streamlit as st
import pandas as pd

# Sample dictionaries (replace these with your actual analysis results)
product_top_vars = {
    "Product A": ["ISE_CO"],
    "Product B": ["TUAV"],
    "Product C": ["TUAV"],
    "Product D": ["VTOTAL_19", "ISE_CO"]
}

grouped_products = {
    "ISE_CO": ["Product A", "Product D"],
    "TUAV": ["Product B", "Product C"],
    "VTOTAL_19": ["Product D"]
}

st.title("Forecasting with Explainable AI")
st.markdown(
    """
    This application shows, on a monthly basis, which external variable(s) are most relevant 
    for forecasting the **Customer Order Quantity** for each product. 
    Products are grouped based on their dominant external variable.
    """
)

st.sidebar.header("Visualization Options")
selected_feature = st.sidebar.selectbox("Select Dominant External Variable", list(grouped_products.keys()))
st.sidebar.write("Products with dominant variable **" + selected_feature + "**:")
st.sidebar.write(grouped_products[selected_feature])
selected_product = st.sidebar.selectbox("Select a product", list(product_top_vars.keys()))
st.header(f"Details for {selected_product}")
st.write("Dominant External Variable(s):", product_top_vars[selected_product])
st.markdown(
    """
    **Note:** The dominant external variable for each product was determined by training an XGBoost 
    forecasting model on a monthly basis and analyzing the SHAP values for each external variable.
    """
)
