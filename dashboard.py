
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Penggunaan Layar Device Anak India 2025",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background-color: #e6f3ff;
        border: 1px solid #a7d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stSpinner > div {
        border-top-color: #1f77b4;
    }
    
    .visualization-section {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load dan preprocess data
@st.cache_data
def load_and_preprocess_data():
    """Load dan preprocess data dari file CSV."""
    with st.spinner('Memuat dan memproses data...'):
        try:
            # Menggunakan nama file yang sama seperti di Colab
            df = pd.read_csv('Penggunaan_Layar_Device_Anak_India_2025.csv')
            
            # Data cleaning dan preprocessing dari Colab
            df.columns = df.columns.str.strip()
            df = df.rename(columns={
                'Rata_rata_Waktu_Layar_per_Hari_(jam)': 'Rata_rata_Waktu_Layar_per_Hari_jam'
            })
            
            # Konversi boolean TRUE/FALSE ke Yes/No - tanpa menghapus baris
            if 'Melebihi_Batas_Waktu_yang_Direkomendasikan' in df.columns:
                df['Melebihi_Batas_Waktu_yang_Direkomendasikan'] = df['Melebihi_Batas_Waktu_yang_Direkomendasikan'].map({
                    True: 'Ya', False: 'Tidak', 'TRUE': 'Ya', 'FALSE': 'Tidak',
                    'Yes': 'Ya', 'No': 'Tidak'
                }).fillna('Tidak') # Isi nilai yang tidak dikenal dengan 'Tidak'
            
            # Konversi tipe data untuk kolom numerik dan isi NA dengan median
            numeric_cols = ['Usia', 'Rata_rata_Waktu_Layar_per_Hari_jam', 'Rasio_Edukasi_terhadap_Hiburan']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(df[col].median())
            
            # Hapus baris yang memiliki nilai null setelah pengisian numerik
            # Mengikuti langkah di Colab untuk dropna setelah mengisi numerik
            original_shape = df.shape
            df = df.dropna()
            st.info(f"Jumlah baris data setelah preprocessing dan penghapusan missing values: {len(df)} (dari {original_shape[0]})")

            return df
            
        except FileNotFoundError:
            st.error("File 'Penggunaan_Layar_Device_Anak_India_2025.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
            st.warning("Membuat data dummy untuk demo.")
            return create_dummy_data()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")
            st.warning("Membuat data dummy untuk demo.")
            return create_dummy_data()

def create_dummy_data():
    """Membuat data dummy untuk tujuan demo."""
    np.random.seed(42)
    n_samples = 9713
    
    data = {
        'Usia': np.random.randint(8, 18, n_samples),
        'Jenis_Kelamin': np.random.choice(['Male', 'Female'], n_samples, p=[0.515, 0.485]),
        'Perangkat_Utama': np.random.choice(['Smartphone', 'TV', 'Laptop', 'Tablet'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Rata_rata_Waktu_Layar_per_Hari_jam': np.round(np.random.normal(5, 2, n_samples), 1),
        'Rasio_Edukasi_terhadap_Hiburan': np.round(np.random.beta(2, 5, n_samples), 2),
        'Melebihi_Batas_Waktu_yang_Direkomendasikan': np.random.choice(['Ya', 'Tidak'], n_samples, p=[0.7, 0.3]),
        'Dampak_Kesehatan': np.random.choice([
            'None', 'Eye Strain', 'Poor Sleep', 'Eye Strain, Poor Sleep',
            'Eye Strain, Headaches', 'Poor Sleep, Behavioral Issues',
            'Eye Strain, Poor Sleep, Headaches', 'Eye Strain, Poor Sleep, Behavioral Issues'
        ], n_samples, p=[0.1, 0.25, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05]),
        'Tinggal_di_Perkotaan_atau_Pedesaan': np.random.choice(['Urban', 'Rural'], n_samples, p=[0.65, 0.35])
    }
    # Ensure Rata_rata_Waktu_Layar_per_Hari_jam is non-negative
    data['Rata_rata_Waktu_Layar_per_Hari_jam'] = np.maximum(0, data['Rata_rata_Waktu_Layar_per_Hari_jam'])
    return pd.DataFrame(data)

# Load data
df = load_and_preprocess_data()

# Header utama
st.markdown('<div class="main-header">📱 Dashboard Analisis Penggunaan Layar Device Pada Anak India 2025</div>', unsafe_allow_html=True)

# Sidebar untuk filter
st.sidebar.header("🔧 Filter Data")

# Filter berdasarkan usia
if 'Usia' in df.columns:
    age_range = st.sidebar.slider(
        "Rentang Usia",
        min_value=int(df['Usia'].min()),
        max_value=int(df['Usia'].max()),
        value=(int(df['Usia'].min()), int(df['Usia'].max()))
    )
    df_filtered = df[(df['Usia'] >= age_range[0]) & (df['Usia'] <= age_range[1])]
else:
    df_filtered = df

# Filter berdasarkan jenis kelamin
if 'Jenis_Kelamin' in df.columns:
    gender_options = st.sidebar.multiselect(
        "Jenis Kelamin",
        options=df['Jenis_Kelamin'].unique(),
        default=list(df['Jenis_Kelamin'].unique()) # Convert to list
    )
    df_filtered = df_filtered[df_filtered['Jenis_Kelamin'].isin(gender_options)]

# Filter berdasarkan perangkat
if 'Perangkat_Utama' in df.columns:
    device_options = st.sidebar.multiselect(
        "Perangkat Utama",
        options=df['Perangkat_Utama'].unique(),
        default=list(df['Perangkat_Utama'].unique()) # Convert to list
    )
    df_filtered = df_filtered[df_filtered['Perangkat_Utama'].isin(device_options)]

# ---
## Ringkasan Data
st.subheader("📊 Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_samples = len(df_filtered)
    st.metric("Total Sampel", f"{total_samples:,}")

with col2:
    if 'Rata_rata_Waktu_Layar_per_Hari_jam' in df_filtered.columns:
        avg_screen_time = df_filtered['Rata_rata_Waktu_Layar_per_Hari_jam'].mean()
        st.metric("Rata-rata Waktu Layar", f"{avg_screen_time:.1f} jam")

with col3:
    if 'Melebihi_Batas_Waktu_yang_Direkomendasikan' in df_filtered.columns:
        pct_exceeding = (df_filtered['Melebihi_Batas_Waktu_yang_Direkomendasikan'] == 'Ya').mean() * 100
        st.metric("Melebihi Batas Waktu", f"{pct_exceeding:.1f}%")

with col4:
    if 'Rasio_Edukasi_terhadap_Hiburan' in df_filtered.columns:
        avg_edu_ratio = df_filtered['Rasio_Edukasi_terhadap_Hiburan'].mean()
        st.metric("Rata-rata Rasio Edukasi", f"{avg_edu_ratio:.2f}")

# ---
## Visualisasi Data Eksploratori dengan Button Toggle
st.subheader("📈 Visualisasi Data Eksploratori")

# Tambahkan button untuk toggle visualisasi
show_visualizations = st.button(
    "🔍 Tampilkan/Sembunyikan Visualisasi",
    help="Klik untuk menampilkan atau menyembunyikan visualisasi data eksploratori"
)

# Gunakan session state untuk menyimpan status tampilan
if 'show_viz' not in st.session_state:
    st.session_state.show_viz = False

# Toggle status saat button diklik
if show_visualizations:
    st.session_state.show_viz = not st.session_state.show_viz

# Tampilkan visualisasi hanya jika status True
if st.session_state.show_viz:
    with st.container():
        st.markdown('<div class="visualization-section">', unsafe_allow_html=True)
        
        # Buat layout 3x3 yang lebih rapi
        st.markdown("#### 📊 Distribusi Variabel Utama")
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        
        # Row 1, Col 1: Distribusi Usia
        with row1_col1:
            if 'Usia' in df_filtered.columns:
                fig_age = px.histogram(
                    df_filtered, 
                    x='Usia', 
                    nbins=15,
                    title='Distribusi Usia',
                    color_discrete_sequence=['skyblue']
                )
                fig_age.add_vline(
                    x=df_filtered['Usia'].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Mean: {df_filtered['Usia'].mean():.1f}"
                )
                fig_age.update_layout(
                    xaxis_title="Usia (tahun)",
                    yaxis_title="Frekuensi",
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_age, use_container_width=True)

        # Row 1, Col 2: Distribusi Jenis Kelamin
        with row1_col2:
            if 'Jenis_Kelamin' in df_filtered.columns:
                gender_counts = df_filtered['Jenis_Kelamin'].value_counts()
                fig_gender = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title='Distribusi Jenis Kelamin',
                    color_discrete_sequence=['#FFB6C1', '#87CEEB']
                )
                fig_gender.update_traces(textposition='inside', textinfo='percent+label')
                fig_gender.update_layout(height=350, margin=dict(l=50, r=50, t=50, b=50))
                st.plotly_chart(fig_gender, use_container_width=True)

        # Row 1, Col 3: Distribusi Waktu Layar per Hari
        with row1_col3:
            if 'Rata_rata_Waktu_Layar_per_Hari_jam' in df_filtered.columns:
                fig_screen = px.histogram(
                    df_filtered,
                    x='Rata_rata_Waktu_Layar_per_Hari_jam',
                    nbins=20,
                    title='Distribusi Waktu Layar per Hari',
                    color_discrete_sequence=['lightgreen']
                )
                fig_screen.add_vline(
                    x=df_filtered['Rata_rata_Waktu_Layar_per_Hari_jam'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {df_filtered['Rata_rata_Waktu_Layar_per_Hari_jam'].mean():.1f}"
                )
                fig_screen.update_layout(
                    xaxis_title="Waktu Layar (jam)",
                    yaxis_title="Frekuensi",
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_screen, use_container_width=True)

        st.markdown("#### 📱 Analisis Perangkat dan Kebiasaan")
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # Row 2, Col 1: Distribusi Perangkat Utama
        with row2_col1:
            if 'Perangkat_Utama' in df_filtered.columns:
                device_counts = df_filtered['Perangkat_Utama'].value_counts()
                fig_device = px.bar(
                    x=device_counts.index,
                    y=device_counts.values,
                    title='Distribusi Perangkat Utama',
                    color=device_counts.values,
                    color_continuous_scale='plotly3'
                )
                fig_device.update_layout(
                    xaxis_title="Perangkat",
                    yaxis_title="Jumlah",
                    height=350,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_device, use_container_width=True)

        # Row 2, Col 2: Melebihi Batas Waktu yang Direkomendasikan
        with row2_col2:
            if 'Melebihi_Batas_Waktu_yang_Direkomendasikan' in df_filtered.columns:
                exceed_counts = df_filtered['Melebihi_Batas_Waktu_yang_Direkomendasikan'].value_counts()
                fig_exceed = px.pie(
                    values=exceed_counts.values,
                    names=exceed_counts.index,
                    title='Melebihi Batas Waktu yang Direkomendasikan',
                    color_discrete_sequence=['#98FB98', '#FFB6C1']
                )
                fig_exceed.update_traces(textposition='inside', textinfo='percent+label')
                fig_exceed.update_layout(height=350, margin=dict(l=50, r=50, t=50, b=50))
                st.plotly_chart(fig_exceed, use_container_width=True)

        # Row 2, Col 3: Distribusi Rasio Edukasi terhadap Hiburan
        with row2_col3:
            if 'Rasio_Edukasi_terhadap_Hiburan' in df_filtered.columns:
                fig_ratio = px.histogram(
                    df_filtered,
                    x='Rasio_Edukasi_terhadap_Hiburan',
                    nbins=20,
                    title='Distribusi Rasio Edukasi terhadap Hiburan',
                    color_discrete_sequence=['gold']
                )
                fig_ratio.add_vline(
                    x=df_filtered['Rasio_Edukasi_terhadap_Hiburan'].mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {df_filtered['Rasio_Edukasi_terhadap_Hiburan'].mean():.2f}"
                )
                fig_ratio.update_layout(
                    xaxis_title="Rasio Edukasi/Hiburan",
                    yaxis_title="Frekuensi",
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

        st.markdown("#### 🏥 Dampak Kesehatan dan Lokasi")
        row3_col1, row3_col2, row3_col3 = st.columns(3)

        # Row 3, Col 1: Distribusi Dampak Kesehatan
        with row3_col1:
            if 'Dampak_Kesehatan' in df_filtered.columns:
                health_counts = df_filtered['Dampak_Kesehatan'].value_counts()
                fig_health = px.bar(
                    x=health_counts.values,
                    y=health_counts.index,
                    orientation='h',
                    title='Distribusi Dampak Kesehatan',
                    color=health_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_health.update_layout(
                    xaxis_title="Jumlah",
                    yaxis_title="Dampak Kesehatan",
                    height=350,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_health, use_container_width=True)

        # Row 3, Col 2: Distribusi Lokasi Tinggal
        with row3_col2:
            if 'Tinggal_di_Perkotaan_atau_Pedesaan' in df_filtered.columns:
                location_counts = df_filtered['Tinggal_di_Perkotaan_atau_Pedesaan'].value_counts()
                fig_location = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title='Distribusi Lokasi Tinggal',
                    color=location_counts.values,
                    color_continuous_scale='Plasma'
                )
                fig_location.update_layout(
                    xaxis_title="Lokasi Tinggal",
                    yaxis_title="Jumlah",
                    height=350,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_location, use_container_width=True)

        # Row 3, Col 3: Waktu Layar vs Dampak Kesehatan (Boxplot)
        with row3_col3:
            if 'Rata_rata_Waktu_Layar_per_Hari_jam' in df_filtered.columns and 'Dampak_Kesehatan' in df_filtered.columns:
                # Simplify health categories as done in Colab
                df_filtered_copy = df_filtered.copy()
                df_filtered_copy['Dampak_Kesehatan_Simple'] = df_filtered_copy['Dampak_Kesehatan'].apply(lambda x: 
                    'None' if x == 'None' else 
                    'Low' if 'Eye Strain' in str(x) and ',' not in str(x) else
                    'Medium' if str(x).count(',') == 0 and 'Poor Sleep' in str(x) else # Adjusted from Colab
                    'High')
                
                # Sort categories for consistent order in plot
                health_order = ['None', 'Low', 'Medium', 'High']
                df_filtered_copy['Dampak_Kesehatan_Simple'] = pd.Categorical(df_filtered_copy['Dampak_Kesehatan_Simple'], categories=health_order, ordered=True)
                df_filtered_copy.sort_values('Dampak_Kesehatan_Simple', inplace=True)

                fig_box = px.box(
                    df_filtered_copy,
                    x='Dampak_Kesehatan_Simple',
                    y='Rata_rata_Waktu_Layar_per_Hari_jam',
                    title='Waktu Layar vs Dampak Kesehatan',
                    color='Dampak_Kesehatan_Simple',
                    color_discrete_sequence=['#98FB98', '#FFB6C1', '#87CEEB', '#DDA0DD'] # Adjusted colors
                )
                fig_box.update_layout(
                    xaxis_title="Dampak Kesehatan",
                    yaxis_title="Waktu Layar (jam)",
                    height=350,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💡 Klik tombol di atas untuk melihat visualisasi data eksploratori")

# Bagian yang diperbaiki untuk Analisis Korelasi
## Analisis Korelasi
st.subheader("🔗 Analisis Korelasi")

# Siapkan data untuk analisis korelasi
correlation_data = df_filtered.copy()

# Encode semua variabel kategorikal untuk analisis korelasi
le = LabelEncoder()
categorical_columns = ['Jenis_Kelamin', 'Perangkat_Utama', 'Melebihi_Batas_Waktu_yang_Direkomendasikan', 
                      'Dampak_Kesehatan', 'Tinggal_di_Perkotaan_atau_Pedesaan']

for col in categorical_columns:
    if col in correlation_data.columns:
        correlation_data[col + '_encoded'] = le.fit_transform(correlation_data[col].astype(str))

# Pilih kolom numerik untuk analisis korelasi
numeric_columns = ['Usia', 'Rata_rata_Waktu_Layar_per_Hari_jam', 'Rasio_Edukasi_terhadap_Hiburan']
encoded_columns = [col + '_encoded' for col in categorical_columns if col in correlation_data.columns]
analysis_columns = numeric_columns + encoded_columns

# Filter kolom yang tersedia
available_columns = [col for col in analysis_columns if col in correlation_data.columns]

if len(available_columns) > 1:
    # Hitung matriks korelasi
    correlation_matrix = correlation_data[available_columns].corr()
    
    # Visualisasi heatmap korelasi
    fig_corr = px.imshow(
        correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale='RdBu',
        aspect="auto",
        title="Heatmap Korelasi Antar Variabel",
        zmin=-1,
        zmax=1
    )
    
    # Add text annotations
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            fig_corr.add_annotation(
                x=i, y=j,
                text=str(round(correlation_matrix.iloc[j, i], 3)),
                showarrow=False,
                font=dict(
                    color="white" if abs(correlation_matrix.iloc[j, i]) > 0.5 else "black",
                    size=10
                )
            )
    
    fig_corr.update_layout(height=500, margin=dict(l=50, r=50, t=50, b=50))
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Analisis korelasi yang signifikan
    st.markdown("#### 🔍 Korelasi Signifikan (|r| > 0.3)")
    
    significant_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.3:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                significant_correlations.append((var1, var2, corr_value))
    
    if significant_correlations:
        for var1, var2, corr in significant_correlations:
            direction = "positif" if corr > 0 else "negatif"
            if abs(corr) > 0.7:
                strength = "kuat"
                box_class = "success-box"
            elif abs(corr) > 0.5:
                strength = "sedang"
                box_class = "warning-box"
            else:
                strength = "lemah"
                box_class = "info-box"
            
            st.markdown(f'<div class="{box_class}">• <b>{var1}</b> ↔ <b>{var2}</b>: r = {corr:.3f} ({direction}, {strength})</div>', unsafe_allow_html=True)
    else:
        st.info("Tidak ada korelasi signifikan yang ditemukan (|r| > 0.3)")
    
    # Fokus pada korelasi dengan waktu layar
    target_col = 'Rata_rata_Waktu_Layar_per_Hari_jam'
    if target_col in correlation_matrix.columns:
        st.markdown(f"#### 📱 Korelasi dengan {target_col}")
        
        waktu_layar_corr = correlation_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
        
        # Create DataFrame for better visualization
        corr_df = pd.DataFrame({
            'Variabel': waktu_layar_corr.index,
            'Korelasi': waktu_layar_corr.values,
            'Korelasi_Abs': abs(waktu_layar_corr.values)
        })
        
        # Filter correlation > 0.1
        significant_corr_df = corr_df[corr_df['Korelasi_Abs'] > 0.1].copy()
        
        if not significant_corr_df.empty:
            # Text summary
            st.markdown("**Korelasi dengan Waktu Layar:**")
            for _, row in significant_corr_df.iterrows():
                var = row['Variabel']
                corr = row['Korelasi']
                direction = "positif" if corr > 0 else "negatif"
                st.markdown(f'<div class="insight-box">• <b>{var}</b>: r = {corr:.3f} ({direction})</div>', unsafe_allow_html=True)
        else:
            st.info(f"Tidak ada korelasi signifikan dengan {target_col} yang ditemukan (|r| > 0.1)")

else:
    st.warning("Tidak cukup kolom numerik untuk analisis korelasi. Pastikan ada lebih dari satu kolom numerik yang tersedia.")

# ---
## Pemodelan Regresi Linear
st.subheader("🤖 Pemodelan Regresi Linear")

if 'Rata_rata_Waktu_Layar_per_Hari_jam' in df.columns:
    with st.spinner('Membangun dan mengevaluasi model regresi linear...'):
        # Siapkan data untuk modeling
        modeling_data = df.copy()

        # Target variable
        target = 'Rata_rata_Waktu_Layar_per_Hari_jam'
        
        # Feature selection
        feature_columns = []
        numerical_features = ['Usia', 'Rasio_Edukasi_terhadap_Hiburan']
        categorical_features = ['Jenis_Kelamin', 'Perangkat_Utama', 'Melebihi_Batas_Waktu_yang_Direkomendasikan', 
                               'Dampak_Kesehatan', 'Tinggal_di_Perkotaan_atau_Pedesaan']

        # Tambahkan numerical features yang ada
        for col in numerical_features:
            if col in modeling_data.columns:
                feature_columns.append(col)

        # Encode categorical features
        le_dict = {}
        for col in categorical_features:
            if col in modeling_data.columns:
                # Check if the column has enough non-NA values to encode
                if modeling_data[col].dropna().nunique() > 1:
                    le_dict[col] = LabelEncoder()
                    modeling_data[col + '_encoded'] = le_dict[col].fit_transform(modeling_data[col].astype(str))
                    feature_columns.append(col + '_encoded')
                else:
                    st.info(f"Kolom '{col}' memiliki nilai unik kurang dari 2 atau hanya NA, tidak akan digunakan sebagai fitur.")

        # Pastikan ada cukup fitur untuk membangun model
        if not feature_columns:
            st.warning("Tidak ada fitur yang tersedia untuk membangun model. Pastikan kolom-kolom data sudah benar.")
        elif len(modeling_data) < 2:
            st.warning("Jumlah data terlalu sedikit untuk membangun model.")
        else:
            X = modeling_data[feature_columns]
            y = modeling_data[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Buat dan latih model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            st.success("Model Regresi Linear berhasil dilatih!")

            st.markdown("#### Koefisien Model (Feature Importance)")
            feature_importance = pd.DataFrame({
                'Feature': feature_columns,
                'Coefficient': model.coef_,
                'Abs_Coefficient': np.abs(model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)

            fig_coef = px.bar(
                feature_importance,
                x='Feature',
                y='Coefficient',
                color='Coefficient',
                color_continuous_scale='RdBu',
                title='Koefisien Fitur Model Regresi Linear'
            )
            fig_coef.update_layout(xaxis_title="Fitur", yaxis_title="Koefisien", height=450)
            st.plotly_chart(fig_coef, use_container_width=True)

            st.markdown("#### Evaluasi Model")
            eval_col1, eval_col2 = st.columns(2)
            with eval_col1:
                st.info(f"**Training Set R²:** {train_r2:.4f}")
                st.info(f"**Training Set MAE:** {train_mae:.4f}")
                st.info(f"**Training Set MSE:** {train_mse:.4f}")
                st.info(f"**Training Set RMSE:** {train_rmse:.4f}")
            with eval_col2:
                st.info(f"**Test Set R²:** {test_r2:.4f}")
                st.info(f"**Test Set MAE:** {test_mae:.4f}")
                st.info(f"**Test Set MSE:** {test_mse:.4f}")
                st.info(f"**Test Set RMSE:** {test_rmse:.4f}")
            
            # Visualisasi Actual vs Predicted
            fig_pred = make_subplots(rows=1, cols=2, subplot_titles=('Actual vs Predicted (Training)', 'Actual vs Predicted (Test)'))

            fig_pred.add_trace(go.Scatter(x=y_train, y=y_train_pred, mode='markers', name='Training Data', 
                                        marker=dict(color='blue', opacity=0.6)), row=1, col=1)
            fig_pred.add_trace(go.Scatter(x=[y_train.min(), y_train.max()], y=[y_train.min(), y_train.max()], 
                                        mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')), row=1, col=1)
            
            fig_pred.add_trace(go.Scatter(x=y_test, y=y_test_pred, mode='markers', name='Test Data', 
                                        marker=dict(color='green', opacity=0.6)), row=1, col=2)
            fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                        mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')), row=1, col=2)

            fig_pred.update_layout(height=450, showlegend=False)
            fig_pred.update_xaxes(title_text="Nilai Aktual", row=1, col=1)
            fig_pred.update_yaxes(title_text="Nilai Prediksi", row=1, col=1)
            fig_pred.update_xaxes(title_text="Nilai Aktual", row=1, col=2)
            fig_pred.update_yaxes(title_text="Nilai Prediksi", row=1, col=2)
            st.plotly_chart(fig_pred, use_container_width=True)



# ---
## Ringkasan Data
st.subheader("📋 Ringkasan Data")
if st.checkbox("Tampilkan Data Mentah"):
    st.dataframe(df_filtered)

if st.checkbox("Tampilkan Statistik Deskriptif"):
    st.write(df_filtered.describe())

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📱 Dashboard Analisis Penggunaan Layar Device Pada Anak India 2025</p>
    <p>Dibuat dengan ❤️ menggunakan Streamlit</p>
</div>
""", unsafe_allow_html=True)