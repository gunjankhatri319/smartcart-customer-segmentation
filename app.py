import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

# Responsive Glassmorphism Styling compatible with Dark & Light modes
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .kpi-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .kpi-title {
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.75;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .persona-box {
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.04);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Cluster personas and strategies
CLUSTER_META = {
    0: {
        "name": "Budget Browsers",
        "color": "#f87171",
        "desc": "Frequent site visitors with conservative spending and low average basket size.",
        "action": "Flash promotions, bundle volume discounts, and low-threshold free-shipping vouchers.",
    },
    1: {
        "name": "Affluent High Spenders",
        "color": "#34d399",
        "desc": "Top-tier household income and maximum spending across premium wines & meats.",
        "action": "Invite to VIP Tier, early access releases, exclusive sommelier/concierge collections.",
    },
    2: {
        "name": "Deal-Seeking Families",
        "color": "#c084fc",
        "desc": "Larger family units with high discount purchase frequency and strong store/catalog use.",
        "action": "Family-pack bundles, seasonal holiday offers, and targeted discount campaigns.",
    },
    3: {
        "name": "Conservative Shoppers",
        "color": "#fbbf24",
        "desc": "Moderate income, low recent engagement, balanced across standard essentials.",
        "action": "Re-engagement email check-ins, targeted product discovery, and loyalty point multipliers.",
    },
}

# Artifact loader
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    df_clustered = pd.read_csv("models/clustered_data.csv")
    return scaler, pca, kmeans, df_clustered

try:
    scaler, pca, kmeans, df = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'python src/train.py' was run.")
    st.stop()

# Title Header
st.title("🛒 SmartCart Customer Segmentation System")
st.caption("AI-driven behavioral clustering & persona-based retention platform")

st.markdown("---")

# KPI Summary Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Total Customers</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Identified Segments</div>
            <div class="kpi-value">4 Cohorts</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Avg Annual Spend</div>
            <div class="kpi-value">${df['Total_Spent'].mean():.0f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Avg Monthly Web Visits</div>
            <div class="kpi-value">{df['NumWebVisitsMonth'].mean():.1f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# Dashboard Navigation Tabs
tab1, tab2, tab3 = st.tabs(["📊 Segment Analytics", "🎯 Customer Classifier", "📋 Persona Profiles"])

df["Segment"] = df["Cluster"].map(lambda x: CLUSTER_META[x]["name"])
color_map = {v["name"]: v["color"] for v in CLUSTER_META.values()}

with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("PCA 2D Cluster Space")
        fig_pca = px.scatter(
            df,
            x="PCA1",
            y="PCA2",
            color="Segment",
            hover_data=["Income", "Total_Spent", "Total_Purchases"],
            color_discrete_map=color_map,
        )
        fig_pca.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_right:
        st.subheader("Income vs. Total Spending")
        fig_scatter = px.scatter(
            df,
            x="Income",
            y="Total_Spent",
            color="Segment",
            size="NumDealsPurchases",
            hover_data=["Recency", "Total_Children"],
            color_discrete_map=color_map,
        )
        fig_scatter.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Predict Customer Segment")
    st.write("Input profile data to immediately identify the user's cluster and recommended engagement strategy.")

    p1, p2, p3 = st.columns(3)
    with p1:
        in_income = st.number_input("Household Income ($)", min_value=5000, max_value=200000, value=52000, step=1000)
        in_recency = st.slider("Recency (Days since last purchase)", 0, 100, 30)
        in_spent = st.number_input("Total Amount Spent ($)", min_value=0, max_value=4000, value=650, step=50)
    with p2:
        in_purchases = st.slider("Total Purchases (Web + Catalog + Store)", 1, 40, 12)
        in_deals = st.slider("Purchases Made With Deals", 0, 15, 2)
        in_visits = st.slider("Monthly Website Visits", 0, 20, 5)
    with p3:
        in_children = st.slider("Total Children & Teens", 0, 4, 1)
        in_days = st.number_input("Customer Tenure (Days)", min_value=100, max_value=5000, value=1500, step=50)

    if st.button("Classify Customer", type="primary", use_container_width=True):
        features_vec = np.array([[in_income, in_recency, in_spent, in_purchases, in_deals, in_visits, in_children, in_days]])
        scaled_vec = scaler.transform(features_vec)
        pred_cluster = kmeans.predict(scaled_vec)[0]
        meta = CLUSTER_META[pred_cluster]

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f"""<div class="persona-box" style="border-left: 5px solid {meta['color']};">
                    <div style="font-size: 0.85rem; opacity: 0.8;">PREDICTED COHORT</div>
                    <h2 style="color: {meta['color']}; margin: 5px 0;">{meta['name']}</h2>
                    <p style="opacity: 0.9; font-size: 0.95rem;">{meta['desc']}</p>
                </div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class="persona-box">
                    <div style="font-size: 0.85rem; opacity: 0.8;">RECOMMENDED RETENTION STRATEGY</div>
                    <h3 style="margin: 5px 0 10px 0;">Next Marketing Action</h3>
                    <p style="font-size: 1.05rem; line-height: 1.6;">{meta['action']}</p>
                </div>""",
                unsafe_allow_html=True,
            )

with tab3:
    st.subheader("Cohort Attribute Aggregation")
    stats_df = (
        df.groupby("Segment")[["Income", "Total_Spent", "Total_Purchases", "NumDealsPurchases", "NumWebVisitsMonth"]]
        .mean()
        .round(1)
        .reset_index()
    )
    st.dataframe(stats_df, use_container_width=True)