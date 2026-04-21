import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="E-Commerce Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CSS KUSTOM
# ============================================
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        # Gunakan sep=',' karena dataset Olist standar menggunakan koma
        # Gunakan path relatif yang aman
        orders_df = pd.read_csv('dataset/orders.csv', sep=',')
        order_reviews_df = pd.read_csv('dataset/order_reviews.csv', sep=',')
        order_items_df = pd.read_csv('dataset/order_items.csv', sep=',')
        sellers_df = pd.read_csv('dataset/sellers.csv', sep=',')
        
        # Konversi datetime untuk orders
        datetime_columns_orders = [
            "order_purchase_timestamp", "order_approved_at", 
            "order_delivered_carrier_date", "order_delivered_customer_date", 
            "order_estimated_delivery_date"
        ]
        for col in datetime_columns_orders:
            if col in orders_df.columns:
                orders_df[col] = pd.to_datetime(orders_df[col], errors='coerce')
        
        # Konversi review_score ke numerik
        if 'review_score' in order_reviews_df.columns:
            order_reviews_df['review_score'] = pd.to_numeric(order_reviews_df['review_score'], errors='coerce')
        
        return orders_df, order_reviews_df, order_items_df, sellers_df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# ============================================
# PROSES DATA
# ============================================
with st.spinner("Memproses data..."):
    orders_df, order_reviews_df, order_items_df, sellers_df = load_data()
    
    # 1. Analisis Pengiriman & Review
    delivery_review_df = pd.merge(
        orders_df[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
        order_reviews_df[['order_id', 'review_score']],
        on='order_id',
        how='inner'
    ).dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])

    # Hitung selisih hari (delivery vs estimated)
    delivery_review_df['delivery_days'] = (
        delivery_review_df['order_delivered_customer_date'].dt.date - 
        delivery_review_df['order_estimated_delivery_date'].dt.date
    ).apply(lambda x: x.days)

    delivery_review_df['delivery_status'] = delivery_review_df['delivery_days'].apply(
        lambda x: '🚀 Lebih Cepat' if x < 0 else ('⚠️ Terlambat' if x > 0 else '✅ Tepat Waktu')
    )

    # 2. Analisis Seller
    seller_performance_df = pd.merge(
        order_items_df[['seller_id', 'order_id']],
        sellers_df[['seller_id', 'seller_city', 'seller_state']],
        on='seller_id',
        how='inner'
    )
    
    city_performance = seller_performance_df.groupby(['seller_city', 'seller_state']).agg(
        total_sellers=('seller_id', 'nunique'),
        total_products_sold=('order_id', 'count')
    ).reset_index()
    
    city_performance['avg_products_per_seller'] = city_performance['total_products_sold'] / city_performance['total_sellers']
    city_performance = city_performance.sort_values('total_products_sold', ascending=False)

# ============================================
# SIDEBAR & FILTER
# ============================================
st.sidebar.title("📊 E-Commerce Dashboard")
page = st.sidebar.radio("📌 Navigasi", ["🏠 Overview", "📦 Analisis Pengiriman", "📍 Analisis Seller", "📈 Kesimpulan"])

st.sidebar.markdown("---")
min_date = orders_df['order_purchase_timestamp'].min().date()
max_date = orders_df['order_purchase_timestamp'].max().date()

start_date = st.sidebar.date_input("📅 Tanggal Mulai", min_date)
end_date = st.sidebar.date_input("📅 Tanggal Akhir", max_date)

# Filter Data Berdasarkan Tanggal
filtered_delivery = delivery_review_df[
    (delivery_review_df['order_purchase_timestamp'].dt.date >= start_date) &
    (delivery_review_df['order_purchase_timestamp'].dt.date <= end_date)
]

# ============================================
# PAGE 1: OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("📊 E-Commerce Overview")
    st.caption(f"Periode: {start_date} hingga {end_date}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Orders", f"{len(filtered_delivery):,}")
    col2.metric("⭐ Avg Rating", f"{filtered_delivery['review_score'].mean():.2f}")
    col3.metric("🏪 Total Sellers", f"{sellers_df['seller_id'].nunique():,}")
    col4.metric("🏙️ Kota Aktif", f"{len(city_performance):,}")
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📦 Status Pengiriman")
        status_counts = filtered_delivery['delivery_status'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#2ecc71', '#f39c12', '#e74c3c'], startangle=90)
        st.pyplot(fig)

    with c2:
        st.subheader("⭐ Distribusi Rating")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_delivery, x='review_score', palette='viridis', ax=ax)
        st.pyplot(fig)

# ============================================
# PAGE 2: ANALISIS PENGIRIMAN
# ============================================
elif page == "📦 Analisis Pengiriman":
    st.title("📦 Pengiriman vs Kepuasan")
    
    avg_rating = filtered_delivery.groupby('delivery_status')['review_score'].mean().sort_values(ascending=False)
    
    col1, col2, col3 = st.columns(3)
    for i, (status, score) in enumerate(avg_rating.items()):
        cols = [col1, col2, col3]
        cols[i].metric(status, f"{score:.2f}")

    st.markdown("---")
    st.subheader("📊 Uji Signifikansi (T-Test)")
    
    faster = filtered_delivery[filtered_delivery['delivery_status'] == '🚀 Lebih Cepat']['review_score']
    late = filtered_delivery[filtered_delivery['delivery_status'] == '⚠️ Terlambat']['review_score']
    
    if not faster.empty and not late.empty:
        t_stat, p_val = stats.ttest_ind(faster, late)
        st.write(f"**P-Value:** `{p_val:.4f}`")
        if p_val < 0.05:
            st.success("✅ **Signifikan**: Keterlambatan berpengaruh nyata terhadap rendahnya rating.")
        else:
            st.warning("⚠️ **Tidak Signifikan**: Perbedaan rating mungkin karena faktor lain.")

# ============================================
# PAGE 3: ANALISIS SELLER
# ============================================
elif page == "📍 Analisis Seller":
    st.title("📍 Lokasi Seller & Penjualan")
    
    st.subheader("🏙️ Top 10 Kota Penjualan Tertinggi")
    top_10 = city_performance.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top_10, x='total_products_sold', y='seller_city', palette='magma', ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Korelasi Jumlah Seller vs Penjualan")
    fig, ax = plt.subplots()
    sns.regplot(data=city_performance, x='total_sellers', y='total_products_sold', ax=ax, scatter_kws={'alpha':0.5})
    st.pyplot(fig)
    
    corr = city_performance['total_sellers'].corr(city_performance['total_products_sold'])
    st.metric("📊 Korelasi Pearson", f"{corr:.3f}")

# ============================================
# PAGE 4: KESIMPULAN
# ============================================
else:
    st.title("📈 Kesimpulan Akhir")
    st.info("""
    1. **Pengiriman**: Ada korelasi kuat antara kecepatan pengiriman dan kepuasan. Pelanggan jauh lebih toleran jika barang sampai sebelum estimasi.
    2. **Geografis**: Penjualan terpusat di kota-kota besar (seperti Sao Paulo). Menambah jumlah seller di kota potensial secara langsung meningkatkan volume transaksi (Korelasi Positif).
    """)
    
    st.download_button(
        "📥 Download Data Terfilter",
        data=filtered_delivery.to_csv(index=False),
        file_name="filtered_data.csv",
        mime="text/csv"
    )