import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.title("Forecasting with Explainable AI – Group Analysis")
st.markdown("""
This app computes groupings of products based on their dominant external variable (determined via SHAP analysis) and then allows you to:
- **Filter by Group:** Choose a dominant external variable group and see which products belong to it.
- **Filter by Product:** Select a specific product and display all products in its group.
Finally, the app trains a model on the filtered group data and displays the SHAP summary plot.
""")

@st.cache_data
def load_merged_data():
    # Load Sales Data from Dropbox
    sales_url = ("https://www.dropbox.com/scl/fi/qs4xxntaz77hxoqqv12ol/"
                 "Customer-Order-Quantity_Dispatched-Quantity.xlsx?rlkey=w13c0i2hl2h1c85wfugvy7hz2&dl=1")
    corona_sales = pd.read_excel(sales_url, engine="openpyxl")
    corona_sales['Date'] = pd.to_datetime(corona_sales['Date'], format="%d.%m.%Y")
    corona_sales = corona_sales.set_index('Date')
    
    # Load External Variables Data from Dropbox (sheet 'Variables')
    ext_url = ("https://www.dropbox.com/scl/fi/f3pfypvlegrmv3d15bpw0/"
               "External_Variables.xlsx?rlkey=51dv464wk0wy703535m27vutu&st=bn3i3su0&dl=1")
    ext_var = pd.read_excel(ext_url, engine="openpyxl", sheet_name='Variables')
    ext_var['DATE'] = pd.to_datetime(ext_var['DATE'], format="%b-%y")
    ext_var = ext_var.set_index('DATE')
    
    # Group monthly sales per product using Month End ('ME')
    corona_monthly_sales_material = corona_sales.groupby(
        [pd.Grouper(level='Date', freq='ME'), 'Product Name']
    )['Customer Order Quantity'].sum().reset_index()
    
    # Adjust to monthly period timestamps
    corona_monthly_sales_material.set_index('Date', inplace=True)
    corona_monthly_sales_material.index = corona_monthly_sales_material.index.to_period('M').to_timestamp()
    
    # Filter ext_var to match the date range of sales
    ext_var.index = pd.to_datetime(ext_var.index)
    min_date = corona_monthly_sales_material.index.min()
    max_date = corona_monthly_sales_material.index.max()
    ext_var = ext_var[(ext_var.index >= min_date) & (ext_var.index <= max_date)]
    
    corona_monthly_sales_material_reset = corona_monthly_sales_material.reset_index()
    corona_monthly_sales_material_reset.set_index('Date', inplace=True)
    corona_monthly_sales_material_reset.index = corona_monthly_sales_material_reset.index.to_period('M').to_timestamp()
    
    # Merge both DataFrames on the date index (inner join)
    merged_df = pd.merge(corona_monthly_sales_material_reset, ext_var,
                         left_index=True, right_index=True, how='inner')
    return merged_df

@st.cache_data
def compute_groupings(merged_df):
    ext_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    products = merged_df['Product Name'].unique()
    product_top_vars = {}
    threshold_min_records = 10

    st.text("Performing product analysis... (this may take a few moments)")
    for prod in products:
        prod_df = merged_df[merged_df['Product Name'] == prod]
        if len(prod_df) < threshold_min_records:
            continue
        X_prod = prod_df[ext_features]
        y_prod = prod_df['Customer Order Quantity']
        
        model = xgb.XGBRegressor(n_estimators=100, random_state=0)
        model.fit(X_prod, y_prod)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_prod)
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        
        max_importance = np.max(shap_abs_mean)
        threshold_val = 0.9 * max_importance
        top_features = [feature for feature, imp in zip(ext_features, shap_abs_mean) if imp >= threshold_val]
        product_top_vars[prod] = top_features

    grouped_products = {}
    for prod, features in product_top_vars.items():
        for feat in features:
            if feat not in grouped_products:
                grouped_products[feat] = []
            grouped_products[feat].append(prod)
    
    return product_top_vars, grouped_products

merged_df = load_merged_data()
product_top_vars, grouped_products = compute_groupings(merged_df)

st.sidebar.header("Filter Options")
filter_option = st.sidebar.radio("Filter by:", ("Group", "Product"))

if filter_option == "Group":
    selected_group = st.sidebar.selectbox("Select Dominant External Variable Group", list(grouped_products.keys()))
    st.sidebar.markdown(f"**Products in group ({selected_group}):**")
    st.sidebar.write(grouped_products[selected_group])
    filtered_products = grouped_products[selected_group]
else:
    selected_product = st.sidebar.selectbox("Select a Product", list(product_top_vars.keys()))
    primary_group = product_top_vars[selected_product][0] if product_top_vars.get(selected_product) else "N/A"
    st.sidebar.markdown(f"**Products in the group for {selected_product} (Group: {primary_group}):**")
    filtered_products = grouped_products.get(primary_group, [selected_product])
    st.sidebar.write(filtered_products)

filtered_df = merged_df[merged_df["Product Name"].isin(filtered_products)]
st.markdown("### Data Overview for Selected Group")
st.write(filtered_df.head())

st.markdown("### SHAP Summary Plot for Selected Group")
if len(filtered_df) < 10:
    st.write("Not enough data in this group to train a reliable model and compute SHAP values.")
else:
    group_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    X_group = filtered_df[group_features]
    y_group = filtered_df["Customer Order Quantity"]
    
    model_group = xgb.XGBRegressor(n_estimators=100, random_state=0)
    model_group.fit(X_group, y_group)
    
    explainer_group = shap.TreeExplainer(model_group)
    shap_values_group = explainer_group.shap_values(X_group)
    
    plt.figure()
    shap.summary_plot(shap_values_group, X_group, feature_names=X_group.columns,
                        plot_type='bar', show=False)
    st.pyplot(plt.gcf())
    plt.clf()
