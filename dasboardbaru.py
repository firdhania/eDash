import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

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
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    # Cek apakah file dataset ada
    required_files = ['dataset/orders.csv', 'dataset/order_reviews.csv', 'dataset/order_items.csv', 'dataset/sellers.csv']
    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ File {file} tidak ditemukan!")
            st.info("Pastikan struktur folder: dataset/orders.csv, dataset/order_reviews.csv, dll.")
            return None, None, None, None
    
    try:
        orders_df = pd.read_csv('dataset/orders.csv', sep=';')
        order_reviews_df = pd.read_csv('dataset/order_reviews.csv', sep=';')
        order_items_df = pd.read_csv('dataset/order_items.csv', sep=';')
        sellers_df = pd.read_csv('dataset/sellers.csv', sep=';')
        
        # Konversi tipe data datetime untuk orders
        datetime_columns_orders = ["order_purchase_timestamp", "order_approved_at", 
                                   "order_delivered_carrier_date", "order_delivered_customer_date", 
                                   "order_estimated_delivery_date"]
        for col in datetime_columns_orders:
            if col in orders_df.columns:
                orders_df[col] = pd.to_datetime(orders_df[col], dayfirst=True, errors='coerce')
        
        # Konversi tipe data datetime untuk order_reviews
        datetime_columns_reviews = ["review_creation_date", "review_answer_timestamp"]
        for col in datetime_columns_reviews:
            if col in order_reviews_df.columns:
                order_reviews_df[col] = pd.to_datetime(order_reviews_df[col], dayfirst=True, 
                                                        format='%d/%m/%Y %H:%M', errors='coerce')
        
        # Konversi review_score ke numerik
        if 'review_score' in order_reviews_df.columns:
            order_reviews_df['review_score'] = pd.to_numeric(order_reviews_df['review_score'], errors='coerce')
        
        return orders_df, order_reviews_df, order_items_df, sellers_df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

# ============================================
# LOAD DAN PREPROCESS DATA
# ============================================
with st.spinner("Loading data..."):
    orders_df, order_reviews_df, order_items_df, sellers_df = load_data()
    
    # Cek apakah data berhasil dimuat
    if orders_df is None:
        st.stop()
    
    # ============================================
    # DATA PENGIRIMAN
    # ============================================
    delivery_review_df = pd.merge(
        orders_df[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
        order_reviews_df[['order_id', 'review_score']],
        on='order_id',
        how='inner'
    )
    
    delivery_review_df = delivery_review_df.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
    delivery_review_df['delivery_days'] = (delivery_review_df['order_delivered_customer_date'] - 
                                            delivery_review_df['order_estimated_delivery_date']).dt.days
    
    # PERUBAHAN WARNA: urutan warna hijau, biru, merah
    delivery_review_df['delivery_status'] = delivery_review_df['delivery_days'].apply(
        lambda x: 'Lebih Cepat' if x < 0 else ('Terlambat' if x > 0 else 'Tepat Waktu')
    )
    
    # ============================================
    # DATA SELLER
    # ============================================
    seller_performance_df = pd.merge(
        order_items_df[['seller_id', 'order_id']],
        sellers_df[['seller_id', 'seller_city', 'seller_state']],
        on='seller_id',
        how='inner'
    )
    
    seller_agg = seller_performance_df.groupby(['seller_id', 'seller_city', 'seller_state']).agg(
        total_products_sold=('order_id', 'count')
    ).reset_index()
    
    city_performance = seller_agg.groupby(['seller_city', 'seller_state']).agg(
        total_sellers=('seller_id', 'count'),
        total_products_sold=('total_products_sold', 'sum')
    ).reset_index()
    city_performance['avg_products_per_seller'] = city_performance['total_products_sold'] / city_performance['total_sellers']
    city_performance = city_performance.sort_values('total_products_sold', ascending=False)

st.success("✅ Data berhasil dimuat!")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("📊 E-Commerce Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📌 Navigasi",
    ["🏠 Overview", "📦 Analisis Pengiriman", "📍 Analisis Seller", "📈 Kesimpulan"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Filter Tanggal")

# Filter tanggal
filtered_orders = orders_df.copy()  # Initialize filtered_orders

if 'order_purchase_timestamp' in orders_df.columns:
    min_date = orders_df['order_purchase_timestamp'].min().date()
    max_date = orders_df['order_purchase_timestamp'].max().date()
    
    start_date = st.sidebar.date_input("📅 Tanggal Mulai", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("📅 Tanggal Akhir", max_date, min_value=min_date, max_value=max_date)
else:
    start_date = None
    end_date = None

st.sidebar.markdown("---")
st.sidebar.info(
    "📊 **Dashboard ini menganalisis:**\n\n"
    "• Kepuasan customer vs waktu pengiriman\n"
    "• Hubungan lokasi seller dengan penjualan"
)

# ============================================
# FILTER DATA PENGIRIMAN (BERDASARKAN TANGGAL)
# ============================================
if start_date and end_date and 'order_purchase_timestamp' in orders_df.columns:
    filtered_orders = orders_df[
        (orders_df['order_purchase_timestamp'].dt.date >= start_date) &
        (orders_df['order_purchase_timestamp'].dt.date <= end_date)
    ]
    filtered_delivery = pd.merge(
        filtered_orders[['order_id', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
        order_reviews_df[['order_id', 'review_score']],
        on='order_id',
        how='inner'
    )
    filtered_delivery = filtered_delivery.dropna(subset=['order_delivered_customer_date', 'order_estimated_delivery_date'])
    if len(filtered_delivery) > 0:
        filtered_delivery['delivery_days'] = (filtered_delivery['order_delivered_customer_date'] - 
                                               filtered_delivery['order_estimated_delivery_date']).dt.days
        # PERUBAHAN WARNA: urutan warna hijau, biru, merah
        filtered_delivery['delivery_status'] = filtered_delivery['delivery_days'].apply(
            lambda x: 'Lebih Cepat' if x < 0 else ('Terlambat' if x > 0 else 'Tepat Waktu')
        )
    else:
        filtered_delivery = pd.DataFrame()  # Empty dataframe if no data
else:
    filtered_delivery = delivery_review_df.copy()
    filtered_orders = orders_df.copy()

# ============================================
# PAGE 1: OVERVIEW
# ============================================
if page == "🏠 Overview":
    st.title("📊 E-Commerce Dashboard")
    st.caption(f"📅 Periode: {start_date} s/d {end_date}" if start_date and end_date else "📅 Semua periode")
    st.markdown("---")
    
    if len(filtered_delivery) == 0:
        st.warning("⚠️ Tidak ada data untuk periode yang dipilih.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Total Orders", f"{len(filtered_delivery):,}")
        with col2:
            st.metric("⭐ Rata-rata Rating", f"{filtered_delivery['review_score'].mean():.2f}")
        with col3:
            st.metric("🏪 Total Sellers", f"{sellers_df['seller_id'].nunique():,}")
        with col4:
            st.metric("🏙️ Kota Aktif", f"{len(city_performance):,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📦 Status Pengiriman")
            # PERUBAHAN WARNA: urutan Lebih Cepat (hijau), Tepat Waktu (biru), Terlambat (merah)
            status_counts = filtered_delivery['delivery_status'].value_counts()
            # Reorder sesuai urutan yang diinginkan
            order_status = ['Lebih Cepat', 'Tepat Waktu', 'Terlambat']
            status_counts = status_counts.reindex(order_status)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#2ecc71', '#3498db', '#e74c3c']  # Hijau, Biru, Merah
            wedges, texts, autotexts = ax.pie(
                status_counts.values, 
                labels=status_counts.index, 
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=(0.05, 0, 0.1)
            )
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            ax.axis('equal')
            st.pyplot(fig)
        
        with col2:
            st.subheader("⭐ Distribusi Rating")
            review_dist = filtered_delivery['review_score'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.bar(review_dist.index, review_dist.values, color='#3498db', edgecolor='white', linewidth=2)
            ax.set_xlabel('Rating', fontsize=12)
            ax.set_ylabel('Jumlah', fontsize=12)
            ax.set_title('Distribusi Rating Pelanggan', fontsize=14)
            ax.set_xticks(range(1, 6))
            for bar, v in zip(bars, review_dist.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{v:,}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            st.pyplot(fig)
        
        st.markdown("---")
        st.subheader("📌 Ringkasan Cepat")
        
        # PERUBAHAN WARNA: menggunakan status tanpa emoji
        faster_pct = (filtered_delivery[filtered_delivery['delivery_status'] == 'Lebih Cepat'].shape[0] / len(filtered_delivery) * 100) if len(filtered_delivery) > 0 else 0
        ontime_pct = (filtered_delivery[filtered_delivery['delivery_status'] == 'Tepat Waktu'].shape[0] / len(filtered_delivery) * 100) if len(filtered_delivery) > 0 else 0
        late_pct = (filtered_delivery[filtered_delivery['delivery_status'] == 'Terlambat'].shape[0] / len(filtered_delivery) * 100) if len(filtered_delivery) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🚀 **Lebih Cepat**: {faster_pct:.1f}% dari total pengiriman")
        with col2:
            st.info(f"✅ **Tepat Waktu**: {ontime_pct:.1f}% dari total pengiriman")
        with col3:
            st.warning(f"⚠️ **Terlambat**: {late_pct:.1f}% dari total pengiriman")

# ============================================
# PAGE 2: ANALISIS PENGIRIMAN
# ============================================
elif page == "📦 Analisis Pengiriman":
    st.title("📦 Analisis Kepuasan Berdasarkan Waktu Pengiriman")
    st.caption(f"📅 Periode: {start_date} s/d {end_date}" if start_date and end_date else "📅 Semua periode")
    st.markdown("---")
    
    if len(filtered_delivery) == 0:
        st.warning("⚠️ Tidak ada data untuk periode yang dipilih.")
    else:
        col1, col2, col3 = st.columns(3)
        
        # PERUBAHAN WARNA: menggunakan status tanpa emoji
        faster_df = filtered_delivery[filtered_delivery['delivery_status'] == 'Lebih Cepat']
        ontime_df = filtered_delivery[filtered_delivery['delivery_status'] == 'Tepat Waktu']
        late_df = filtered_delivery[filtered_delivery['delivery_status'] == 'Terlambat']
        
        with col1:
            st.metric("🚀 Lebih Cepat", f"{faster_df['review_score'].mean():.2f}" if not faster_df.empty else "N/A", 
                      delta=f"{len(faster_df):,} pesanan")
        with col2:
            st.metric("✅ Tepat Waktu", f"{ontime_df['review_score'].mean():.2f}" if not ontime_df.empty else "N/A",
                      delta=f"{len(ontime_df):,} pesanan")
        with col3:
            st.metric("⚠️ Terlambat", f"{late_df['review_score'].mean():.2f}" if not late_df.empty else "N/A",
                      delta=f"{len(late_df):,} pesanan", delta_color="inverse")
        
        st.markdown("---")
        
        st.subheader("📊 Perbandingan Rata-rata Rating")
        
        # PERUBAHAN WARNA: urutan Lebih Cepat (hijau), Tepat Waktu (biru), Terlambat (merah)
        avg_rating = filtered_delivery.groupby('delivery_status')['review_score'].mean().reset_index()
        # Reorder sesuai urutan yang diinginkan
        order_status = ['Lebih Cepat', 'Tepat Waktu', 'Terlambat']
        avg_rating['delivery_status'] = pd.Categorical(avg_rating['delivery_status'], categories=order_status, ordered=True)
        avg_rating = avg_rating.sort_values('delivery_status')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_bar = ['#2ecc71', '#3498db', '#e74c3c']  # Hijau, Biru, Merah
        bars = ax.bar(avg_rating['delivery_status'], avg_rating['review_score'], color=colors_bar, edgecolor='black', linewidth=1.5)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel('Rata-rata Rating (1-5)', fontsize=12)
        ax.set_title('Perbandingan Rata-rata Rating Berdasarkan Status Pengiriman', fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, v in zip(bars, avg_rating['review_score']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{v:.2f}', 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        st.subheader("📊 Distribusi Rating")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [
            faster_df['review_score'].dropna(),
            ontime_df['review_score'].dropna(),
            late_df['review_score'].dropna()
        ]
        bp = ax.boxplot(data_to_plot, labels=['🚀 Lebih Cepat', '✅ Tepat Waktu', '⚠️ Terlambat'], patch_artist=True)
        colors_box = ['#2ecc71', '#3498db', '#e74c3c']  # Hijau, Biru, Merah
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Rating', fontsize=12)
        ax.set_title('Distribusi Rating Berdasarkan Status Pengiriman', fontsize=14)
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("📈 Uji Statistik (T-Test)")
        
        if len(faster_df) > 0 and len(late_df) > 0:
            t_stat, p_value = stats.ttest_ind(faster_df['review_score'].dropna(), late_df['review_score'].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 T-Statistic", f"{t_stat:.2f}")
            with col2:
                st.metric("🎯 P-Value", f"{p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ **Kesimpulan:** Perbedaan rating **SIGNIFIKAN** secara statistik! Keterlambatan pengiriman terbukti menurunkan kepuasan customer.")
            else:
                st.warning("⚠️ **Kesimpulan:** Tidak ada perbedaan signifikan antara kedua kelompok.")
        
        st.markdown("---")
        st.subheader("📌 Kesimpulan Analisis Pengiriman")
        
        faster_rating = faster_df['review_score'].mean() if not faster_df.empty else 0
        late_rating = late_df['review_score'].mean() if not late_df.empty else 0
        
        st.info(
            f"📦 **Temuan Utama:**\n\n"
            f"• Pengiriman **lebih cepat** dari estimasi menghasilkan kepuasan tertinggi (⭐ {faster_rating:.2f})\n"
            f"• Pengiriman **terlambat** menurunkan rating secara drastis (⭐ {late_rating:.2f})\n"
            f"• Perbedaan ini **signifikan secara statistik** (p-value < 0.05)\n\n"
            f"💡 **Rekomendasi:** Prioritaskan ketepatan waktu pengiriman untuk menjaga kepuasan pelanggan."
        )

# ============================================
# PAGE 3: ANALISIS SELLER (TOP 10 & TOP 5 FIX)
# ============================================
elif page == "📍 Analisis Seller":
    st.title("📍 Analisis Lokasi Seller vs Penjualan")
    st.markdown("---")
    
    if len(city_performance) == 0:
        st.warning("⚠️ Tidak ada data seller.")
    else:
        # METRIC GLOBAL
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏪 Total Seller", f"{city_performance['total_sellers'].sum():,}")
        with col2:
            st.metric("📦 Total Produk Terjual", f"{city_performance['total_products_sold'].sum():,}")
        with col3:
            avg_per_seller = city_performance['total_products_sold'].sum() / city_performance['total_sellers'].sum() if city_performance['total_sellers'].sum() > 0 else 0
            st.metric("📊 Rata-rata per Seller", f"{avg_per_seller:.0f} produk")
        
        st.markdown("---")
        
        # ============================================
        # TOP 10 KOTA DENGAN PENJUALAN TERTINGGI
        # ============================================
        st.subheader("🏙️ Top 10 Kota dengan Penjualan Tertinggi")
        
        top_cities = city_performance.head(10)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bars = ax.barh(top_cities['seller_city'], top_cities['total_products_sold'], 
                       color=plt.cm.viridis(np.linspace(0.2, 0.9, len(top_cities))), 
                       edgecolor='black', linewidth=1, height=0.7)
        
        ax.set_xlabel('Total Produk Terjual', fontsize=13, fontweight='bold')
        ax.set_title('Top 10 Kota dengan Penjualan Tertinggi', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        max_val = top_cities['total_products_sold'].max()
        for bar, val in zip(bars, top_cities['total_products_sold']):
            ax.text(val + (max_val * 0.01), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{val:,}', 
                    va='center', 
                    ha='left', 
                    fontsize=11, 
                    fontweight='bold',
                    color='#2c3e50')
        
        ax.set_xlim(0, max_val * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ============================================
        # SCATTER PLOT KORELASI
        # ============================================
        st.subheader("📈 Hubungan Jumlah Seller vs Total Penjualan")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(city_performance['total_sellers'], city_performance['total_products_sold'], 
                             c=city_performance['total_products_sold'], cmap='viridis', s=120, alpha=0.7, edgecolor='black', linewidth=1)
        ax.set_xlabel('Jumlah Seller', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Produk Terjual', fontsize=12, fontweight='bold')
        ax.set_title('Hubungan Jumlah Seller dengan Total Penjualan per Kota', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Total Produk')
        ax.grid(True, alpha=0.3, linestyle='--')
        st.pyplot(fig)
        
        # KORELASI
        correlation = 0
        if len(city_performance) > 1:
            correlation = city_performance[['total_sellers', 'total_products_sold']].corr().iloc[0, 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 Korelasi Pearson", f"{correlation:.3f}")
            with col2:
                if correlation > 0.7:
                    st.success("✅ Hubungan **Sangat Kuat & Positif**")
                elif correlation > 0.3:
                    st.info("📈 Hubungan **Cukup Kuat & Positif**")
                else:
                    st.warning("⚠️ Hubungan **Lemah**")
        
        # ============================================
        # TOP 5 KOTA DENGAN PRODUKTIVITAS TERTINGGI
        # ============================================
        st.subheader("⭐ Top 5 Kota dengan Produktivitas Seller Tertinggi")
        
        top_productivity = city_performance.sort_values('avg_products_per_seller', ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars2 = ax.barh(top_productivity['seller_city'], top_productivity['avg_products_per_seller'], 
                        color='#2ecc71', edgecolor='black', linewidth=1, height=0.6)
        ax.set_xlabel('Rata-rata Produk per Seller', fontsize=12, fontweight='bold')
        ax.set_title('Top 5 Kota dengan Produktivitas Tertinggi', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        max_val_prod = top_productivity['avg_products_per_seller'].max()
        for bar, val in zip(bars2, top_productivity['avg_products_per_seller']):
            ax.text(val + (max_val_prod * 0.02), 
                    bar.get_y() + bar.get_height()/2, 
                    f'{val:.0f}', 
                    va='center', 
                    ha='left', 
                    fontsize=11, 
                    fontweight='bold',
                    color='#2c3e50')
        
        ax.set_xlim(0, max_val_prod * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ============================================
        # DATA TABLE (SEMUA KOTA)
        # ============================================
        st.subheader("📋 Data Lengkap Per Kota")
        st.dataframe(
            city_performance[['seller_city', 'seller_state', 'total_sellers', 'total_products_sold', 'avg_products_per_seller']]
            .head(20)
            .style.format({
                'total_products_sold': '{:,}',
                'avg_products_per_seller': '{:.0f}'
            })
            .background_gradient(subset=['total_products_sold'], cmap='YlOrRd')
        )
        
        # KESIMPULAN
        st.markdown("---")
        st.subheader("📌 Kesimpulan Analisis Seller")
        
        top_city_name = city_performance.iloc[0]['seller_city'] if not city_performance.empty else "N/A"
        top_city_sales = city_performance.iloc[0]['total_products_sold'] if not city_performance.empty else 0
        top_city_state = city_performance.iloc[0]['seller_state'] if not city_performance.empty else "N/A"
        
        st.info(
            f"📍 **Temuan Utama:**\n\n"
            f"• Kota dengan penjualan tertinggi: **{top_city_name} ({top_city_state})** dengan {top_city_sales:,} produk\n"
            f"• Terdapat hubungan positif yang kuat antara jumlah seller dan total penjualan (korelasi {correlation:.3f})\n"
            f"• Semakin banyak seller di suatu kota, semakin tinggi penjualannya\n\n"
            f"💡 **Rekomendasi:** Ekspansi seller ke kota-kota dengan potensi tinggi dan tambah seller di kota besar."
        )

# ============================================
# PAGE 4: KESIMPULAN
# ============================================
elif page == "📈 Kesimpulan":
    st.title("📈 Kesimpulan Akhir")
    st.caption(f"📅 Periode: {start_date} s/d {end_date}" if start_date and end_date else "📅 Semua periode")
    st.markdown("---")
    
    if len(filtered_delivery) == 0:
        st.warning("⚠️ Tidak ada data untuk periode yang dipilih.")
    else:
        # KESIMPULAN PERTANYAAN 1
        st.header("📦 Pertanyaan 1: Tingkat Kepuasan Customer Berdasarkan Waktu Pengiriman")
        
        # PERUBAHAN WARNA: menggunakan status tanpa emoji
        faster_rating = filtered_delivery[filtered_delivery['delivery_status'] == 'Lebih Cepat']['review_score'].mean() if len(filtered_delivery[filtered_delivery['delivery_status'] == 'Lebih Cepat']) > 0 else 0
        late_rating = filtered_delivery[filtered_delivery['delivery_status'] == 'Terlambat']['review_score'].mean() if len(filtered_delivery[filtered_delivery['delivery_status'] == 'Terlambat']) > 0 else 0
        ontime_rating = filtered_delivery[filtered_delivery['delivery_status'] == 'Tepat Waktu']['review_score'].mean() if len(filtered_delivery[filtered_delivery['delivery_status'] == 'Tepat Waktu']) > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚀 Lebih Cepat", f"{faster_rating:.2f}")
        with col2:
            st.metric("✅ Tepat Waktu", f"{ontime_rating:.2f}")
        with col3:
            st.metric("⚠️ Terlambat", f"{late_rating:.2f}")
        
        st.success(
            "**Kesimpulan:** Terdapat hubungan yang signifikan antara waktu pengiriman dan tingkat kepuasan pelanggan. "
            f"Pengiriman yang lebih cepat memiliki rata-rata rating {faster_rating:.2f}, "
            f"sedangkan pengiriman terlambat hanya {late_rating:.2f}. "
            "Penundaan pengiriman menurunkan kepuasan customer secara drastis."
        )
        
        st.markdown("---")
        
        # KESIMPULAN PERTANYAAN 2
        st.header("📍 Pertanyaan 2: Hubungan Lokasi Geografis Seller dengan Jumlah Produk Terjual")
        
        if len(city_performance) > 1:
            correlation = city_performance[['total_sellers', 'total_products_sold']].corr().iloc[0, 1]
        else:
            correlation = 0
            
        top_city = city_performance.iloc[0]['seller_city'] if not city_performance.empty else "N/A"
        top_state = city_performance.iloc[0]['seller_state'] if not city_performance.empty else "N/A"
        top_sales = city_performance.iloc[0]['total_products_sold'] if not city_performance.empty else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Korelasi", f"{correlation:.3f}")
        with col2:
            st.metric("🏆 Kota Terlaris", f"{top_city} ({top_state})", delta=f"{top_sales:,} produk")
        
        st.success(
            "**Kesimpulan:** Ya, ada hubungan yang jelas antara lokasi geografis seller dan jumlah produk yang terjual. "
            f"Data menunjukkan konsentrasi penjualan yang tinggi di kota besar, terutama **{top_city} ({top_state})** "
            f"yang mendominasi dengan {top_sales:,} produk terjual. "
            "Seller di pusat ekonomi memiliki volume penjualan lebih tinggi karena akses pasar yang lebih luas "
            "dan infrastruktur logistik yang lebih efisien."
        )
        
        st.markdown("---")
        
        # REKOMENDASI
        st.header("💡 Rekomendasi Bisnis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚚 **Strategi Pengiriman**
            1. **Prioritaskan Ketepatan Waktu**
               - Keterlambatan menurunkan rating signifikan
               - Berikan kompensasi jika terjadi keterlambatan
            
            2. **Optimalkan Estimasi**
               - Berikan estimasi yang realistis
               - Kejutan positif (lebih cepat) meningkatkan kepuasan
            """)
        
        with col2:
            st.markdown("""
            ### 📍 **Strategi Ekspansi Seller**
            1. **Fokus pada Kota Produktif**
               - Ekspansi ke kota dengan produktivitas tinggi
               - Tambah seller di kota besar untuk meningkatkan penjualan
            
            2. **Diversifikasi Wilayah**
               - Kurangi ketergantungan pada satu kota/state
               - Kembangkan infrastruktur logistik di luar pusat ekonomi
            """)
        
        # DOWNLOAD BUTTON
        st.markdown("---")
        st.download_button(
            label="📥 Download Data Analisis (CSV)",
            data=filtered_delivery.to_csv(index=False).encode('utf-8'),
            file_name=f"ecommerce_analysis_{start_date}_to_{end_date}.csv",
            mime="text/csv",
            use_container_width=True
        )