import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk memuat semua data dari folder
@st.cache_data
def load_data():
    folder_path = "PRSA_Data_20130301-20170228"
    missing_value_format = ['N.A', 'na', 'n.a.', 'n/a', '?', '-']
    
    # Ambil semua file CSV di folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Gabungkan semua file CSV menjadi satu DataFrame
    df_list = [pd.read_csv(file, na_values=missing_value_format) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    return df

# Load data
df = load_data()

# Sidebar untuk navigasi
st.sidebar.header("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", [
    "Pilih Halaman",  # Tambahkan opsi default
    "PM2.5 Tertinggi per Lokasi", 
    "Tren Kualitas Udara Tahunan", 
    "Prediksi vs Aktual PM2.5"
])

# Filter Tahun hanya untuk halaman Prediksi vs Aktual PM2.5
if page == "Prediksi vs Aktual PM2.5":
    year = st.sidebar.selectbox("Pilih Tahun", sorted(df["year"].dropna().unique()))
    df_filtered = df[df["year"] == year]
else:
    df_filtered = df

# Fungsi untuk model prediksi PM2.5
def train_model(df):
    df = df.dropna()
    X = df[["TEMP", "PRES", "DEWP", "RAIN"]]
    y = df["PM2.5"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return X_test, y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

# Tampilan dashboard berdasarkan halaman yang dipilih
st.markdown("<h1 style='text-align: center;'> Dashboard Analisis Kualitas Udara</h1>", unsafe_allow_html=True)

# Tampilan awal (tanpa data)
if page == "Pilih Halaman":
    st.markdown("<h3 style='text-align: center;'>Selamat Datang di Dashboard Analisis Kualitas Udara!</h3>", unsafe_allow_html=True)

# Halaman PM2.5 Tertinggi per Lokasi
elif page == "PM2.5 Tertinggi per Lokasi":
    st.subheader("Tingkat PM2.5 Tertinggi di Setiap Lokasi")
    df_clean = df.dropna(subset=['station', 'PM2.5'])
    max_pm25_per_site = df_clean.groupby('station')['PM2.5'].max().reset_index()
    max_pm25_per_site['station'] = max_pm25_per_site['station'].str.strip()
    max_pm25_per_site = max_pm25_per_site.sort_values(by='PM2.5', ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='PM2.5', y='station', data=max_pm25_per_site, palette='Reds_r', ax=ax)
    ax.set_title('Tingkat PM2.5 Tertinggi di Setiap Lokasi')
    ax.set_xlabel('PM2.5 Tertinggi')
    ax.set_ylabel('Lokasi')
    plt.tight_layout()
    st.pyplot(fig)

# Halaman Tren Kualitas Udara Tahunan
elif page == "Tren Kualitas Udara Tahunan":
    st.subheader("Tren Kualitas Udara (PM2.5) dari Tahun ke Tahun")
    annual_trend = df.groupby('year')['PM2.5'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='year', y='PM2.5', data=annual_trend, marker='o', color='green', ax=ax)
    ax.set_title('Tren Kualitas Udara (PM2.5) dari Tahun ke Tahun')
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Rata-rata PM2.5')
    ax.grid(True)
    st.pyplot(fig)

# Halaman Prediksi vs Aktual PM2.5
elif page == "Prediksi vs Aktual PM2.5":
    st.subheader("Prediksi vs Aktual PM2.5")
    X_test, y_test, y_pred, _, _ = train_model(df_filtered)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='blue')
    ax.set_xlabel('Nilai Aktual PM2.5')
    ax.set_ylabel('Nilai Prediksi PM2.5')
    ax.set_title('Prediksi vs Aktual PM2.5')
    ax.grid()
    st.pyplot(fig)

# Menjalankan aplikasi Streamlit secara lokal
def main():
    st.write("<h4 style='text-align: center;'> Gunakan sidebar untuk memilih halaman dan melihat hasil analisis</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
