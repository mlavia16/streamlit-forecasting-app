import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

st.title("Forecasting with Explainable AI – Global Model Group Analysis")
st.markdown("""
This app loads precomputed merged data, trains a **single global model** for forecasting, and then:
- Determines each product’s dominant external variable (from the global model’s SHAP values).
- Groups products by that dominant feature.
- Allows you to filter by group (dominant external variable) or by a specific product.
- Displays a SHAP summary plot for the selected group (using the global model's SHAP values).
""")

@st.cache_data
def load_merged_data():
    # Attempt to load merged data (precomputed during a heavy precomputation step)
    try:
        merged_df = pd.read_csv("merged_data.csv", parse_dates=["Date"])
    except Exception as e:
        st.error("Error loading merged_data.csv: " + str(e))
        merged_df = pd.DataFrame()
    return merged_df

@st.cache_data(show_spinner=True)
def compute_global_model_and_groupings(merged_df):
    # Define the external features to use in the model
    ext_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    
    # Prepare the data for training the global model
    X = merged_df[ext_features]
    y = merged_df["Customer Order Quantity"]
    
    # Train one global model (using fewer rounds for speed)
    global_model = xgb.XGBRegressor(n_estimators=10, random_state=0)
    global_model.fit(X, y)
    
    # Compute global SHAP values for the entire dataset
    explainer = shap.TreeExplainer(global_model)
    global_shap_values = explainer.shap_values(X)
    
    # Compute each product’s dominant external variable using the global model’s SHAP values.
    products = merged_df['Product Name'].unique()
    product_top_vars = {}
    for prod in products:
        # Select rows corresponding to the product.
        prod_mask = merged_df['Product Name'] == prod
        # If there are no rows, skip (should not happen)
        if prod_mask.sum() == 0:
            continue
        # Compute mean absolute SHAP values for each feature for this product.
        mean_abs_shap = np.abs(global_shap_values[prod_mask]).mean(axis=0)
        # Determine the dominant feature (using global ordering)
        dominant_feature = ext_features[np.argmax(mean_abs_shap)]
        product_top_vars[prod] = dominant_feature
    
    # Group products by their dominant external variable.
    grouped_products = {}
    for prod, feat in product_top_vars.items():
        if feat not in grouped_products:
            grouped_products[feat] = []
        grouped_products[feat].append(prod)
    
    return global_model, global_shap_values, product_top_vars, grouped_products

# Load the merged data
merged_df = load_merged_data()

if merged_df.empty:
    st.error("Merged data is empty. Please check that merged_data.csv is correctly loaded.")
else:
    # Compute the global model and grouping info
    global_model, global_shap_values, product_top_vars, grouped_products = compute_global_model_and_groupings(merged_df)

    # Sidebar filtering options
    st.sidebar.header("Filter Options")
    filter_option = st.sidebar.radio("Filter by:", ("Group", "Product"))
    
    if filter_option == "Group":
        selected_group = st.sidebar.selectbox("Select Dominant External Variable Group", list(grouped_products.keys()))
        st.sidebar.markdown(f"**Products in group ({selected_group}):**")
        st.sidebar.write(grouped_products[selected_group])
        filtered_products = grouped_products[selected_group]
    else:
        selected_product = st.sidebar.selectbox("Select a Product", list(product_top_vars.keys()))
        primary_group = product_top_vars[selected_product]
        st.sidebar.markdown(f"**Products in the group for {selected_product} (Group: {primary_group}):**")
        filtered_products = grouped_products.get(primary_group, [selected_product])
        st.sidebar.write(filtered_products)
    
    # Filter merged data based on selected products
    filtered_df = merged_df[merged_df["Product Name"].isin(filtered_products)]
    st.markdown("### Data Overview for Selected Group")
    st.write(filtered_df.head())
    
    # For visualization, extract data corresponding to the filtered group
    ext_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    X_group = filtered_df[ext_features]
    
    # For the SHAP plot, re-use the global model's SHAP values for the filtered rows.
    # Here, we find the indices of the filtered rows, then extract those rows from global_shap_values.
    filtered_indices = filtered_df.index
    # Because global_shap_values is computed for all rows in merged_df,
    # we need to extract those corresponding to the filtered subset.
    # For simplicity, we re-compute the filtered SHAP values directly:
    # (This is fast because the global model is already trained.)
    explainer = shap.TreeExplainer(global_model)
    shap_values_group = explainer.shap_values(X_group)
    
    st.markdown("### SHAP Summary Plot for Selected Group")
    if len(filtered_df) < 10:
        st.write("Not enough data in this group to compute a reliable SHAP summary plot.")
    else:
        plt.figure()
        shap.summary_plot(shap_values_group, X_group, feature_names=X_group.columns,
                            plot_type='bar', show=False)
        st.pyplot(plt.gcf())
        plt.clf()
