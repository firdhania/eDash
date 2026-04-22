# Proyek Analisis Data

## Deskripsi Proyek
Dashboard interaktif untuk menganalisis hubungan antara waktu pengiriman dengan kepuasan pelanggan, serta performa seller berdasarkan lokasi geografis.

## Struktur Folder
SUBMISSION/
├── dataset/
│   ├── customers.csv
│   ├── geolocation.csv
│   ├── order_items.csv
│   ├── order_payments.csv
│   ├── order_reviews.csv
│   ├── orders.csv
│   ├── product_category.csv
│   ├── products.csv
│   ├── sellers.csv
├── venv/
│   ├── etc
│   ├── Include
│   ├── Lib
│   ├── Scripts
│   ├── share
│   ├── .gitignore
│   ├── pyvnev.cfg
│   ├── product_category_name.csv
│   ├── products_dataset.csv
│   ├── sales_df.csv
│   ├── sellers_dataset.csv
├── dashboardbaru.py
├── README.md
├── requirements.txt
└── url.txt

## Cara Menjalankan
1. **Pastikan lingkungan Python terinstal**
   - Gunakan Python versi 3.8 atau lebih baru

2. **Instal dependensi**

   ```sh
   pip install streamlit
   pip install -r requirements.txt
   ```

3. **Jalankan program**
   ```sh
   streamlit run SUBMISSION/dashboardbaru.py
   ```