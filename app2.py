import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

st.title("Forecasting with Explainable AI – Group Analysis (Precomputed)")
st.markdown("""
This app loads precomputed groupings and merged data, allowing you to:
- Filter by group (i.e. dominant external variable) or by a specific product.
- View the list of products in the selected group.
- See a SHAP summary plot for the selected group’s forecasting model.
""")

@st.cache_data
def load_precomputed_data():
    # Load precomputed dictionaries
    with open("product_top_vars.pkl", "rb") as f:
        product_top_vars = pickle.load(f)
    with open("grouped_products.pkl", "rb") as f:
        grouped_products = pickle.load(f)
    # Try loading the merged data with the Date column parsed
    try:
        merged_df = pd.read_csv("merged_data.csv", parse_dates=["Date"])
    except ValueError:
        merged_df = pd.read_csv("merged_data.csv")
        st.warning("The 'Date' column was not found or parsed. Please check the CSV format.")
    return product_top_vars, grouped_products, merged_df

# (Optional) This function recomputes groupings from merged_df.
@st.cache_data
def compute_groupings(merged_df):
    ext_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    products = merged_df['Product Name'].unique()
    product_top_vars = {}
    threshold_min_records = 10  # Only analyze products with at least 10 months of data

    st.text("Performing product analysis... (this may take a few moments)")
    for prod in products:
        prod_df = merged_df[merged_df['Product Name'] == prod]
        if len(prod_df) < threshold_min_records:
            continue
        X_prod = prod_df[ext_features]
        y_prod = prod_df['Customer Order Quantity']
        model = xgb.XGBRegressor(n_estimators=10, random_state=0)
        model.fit(X_prod, y_prod)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_prod)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        dominant_feature_index = np.argmax(mean_abs_shap)
        dominant_feature = ext_features[dominant_feature_index]
        product_top_vars[prod] = dominant_feature
    grouped_products = {}
    for prod, feat in product_top_vars.items():
        if feat not in grouped_products:
            grouped_products[feat] = []
        grouped_products[feat].append(prod)
    return product_top_vars, grouped_products

# Load precomputed data from files
product_top_vars, grouped_products, merged_df = load_precomputed_data()

# You may alternatively recompute groupings by uncommenting the next line:
# product_top_vars, grouped_products = compute_groupings(merged_df)

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

# Filter merged data for the selected products
filtered_df = merged_df[merged_df["Product Name"].isin(filtered_products)]
st.markdown("### Data Overview for Selected Group")
st.write(filtered_df.head())

# Train a model on group data and display the SHAP summary plot
st.markdown("### SHAP Summary Plot for Selected Group")
if len(filtered_df) < 10:
    st.write("Not enough data in this group to train a reliable model and compute SHAP values.")
else:
    group_features = ['ECG_DESP', 'TUAV', 'PIB_CO', 'ISE_CO', 'VTOTAL_19', 'OTOTAL_19', 'ICI']
    X_group = filtered_df[group_features]
    y_group = filtered_df["Customer Order Quantity"]
    
    model_group = xgb.XGBRegressor(n_estimators=10, random_state=0)
    model_group.fit(X_group, y_group)
    
    explainer_group = shap.TreeExplainer(model_group)
    shap_values_group = explainer_group.shap_values(X_group)
    
    plt.figure()
    shap.summary_plot(shap_values_group, X_group, feature_names=X_group.columns,
                        plot_type='bar', show=False)
    st.pyplot(plt.gcf())
    plt.clf()

