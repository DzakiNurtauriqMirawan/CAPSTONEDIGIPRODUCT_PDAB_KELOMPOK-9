import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Fungsi untuk memuat model yang telah dilatih
def load_model():
    model = joblib.load('dtc.pkl')
    return model

# Fungsi untuk melakukan prediksi
def predict(input_data, model):
    # Lakukan prediksi dengan model yang telah dilatih
    prediction = model.predict(input_data)
    return prediction

# Membaca data dari file CSV
df = pd.read_csv('data-before-mapping (1).csv')

# Judul halaman
st.image("https://2.bp.blogspot.com/-noBYeeAuxuo/T74s8CT3StI/AAAAAAAAAdo/RFxovPpsE6M/s1600/efek-rumah-kaca.jpg")
st.title("Analisis Emisi Gas Rumah Kaca di Kanada: Pendekatan Klasifikasi untuk Mengidentifikasi Sumber Utama Emisi")

# Judul sidebar
st.sidebar.title('Dashboard')

# Daftar navigasi
nav_selection = st.sidebar.selectbox("Go to", ["Home", "Distribusi", "Hubungan", "Perbandingan dan Komposisi", "Predict"])

# Jika pilihan di sidebar adalah "Home"
if nav_selection == "Home":
    st.header('Business Objective')
    st.write('Tujuan dari proyek ini adalah untuk membantu pemerintah Kanada dalam mengidentifikasi sumber utama emisi gas rumah kaca di negara mereka. Dengan pemahaman yang lebih baik tentang sumber-sumber utama emisi, pemerintah dapat merancang kebijakan yang lebih efektif untuk mengurangi emisi dan memenuhi komitmen mereka terhadap perubahan iklim global.')
    
    st.header('Asses Situations')
    st.write('Kanada adalah salah satu negara dengan emisi gas rumah kaca yang tinggi, terutama karena industri minyak dan gas serta sektor transportasi yang besar. Untuk mengurangi dampak perubahan iklim, pemerintah Kanada membutuhkan pemahaman yang lebih mendalam tentang sumber-sumber utama emisi tersebut. Saat ini, data yang ada mungkin tersebar dan sulit untuk diolah secara efektif. Oleh karena itu, diperlukan pendekatan klasifikasi yang tepat untuk mengidentifikasi sumber-sumber utama emisi ini.')
    
    st.header('Data Mining Goals')
    st.write('- Mengidentifikasi sumber-sumber utama emisi gas rumah kaca di Kanada.')
    st.write('- Membuat model klasifikasi yang dapat mengklasifikasikan jenis emisi berdasarkan data yang ada.')
    
    st.header('Project Plan')
    st.write('- Pengumpulan Data: Mengumpulkan data yang relevan tentang emisi gas rumah kaca di Kanada dari berbagai sumber seperti lembaga pemerintah, organisasi lingkungan, dan industri terkait.')
    st.write('- Persiapan Data: Membersihkan dan mempersiapkan data untuk analisis, termasuk penghapusan nilai yang hilang, normalisasi data, dan pemrosesan lainnya.')
    st.write('- Eksplorasi Data: Menganalisis dan memahami pola-pola dalam data, serta menjelajahi hubungan antara variabel-variabel yang relevan.')
    st.write('- Pembuatan Model: Membangun model klasifikasi menggunakan teknik seperti decision tree, GNB, atau KNN untuk mengidentifikasi sumber-sumber utama emisi.')
    st.write('- Evaluasi Model: Mengukur kinerja model menggunakan metrik yang sesuai seperti akurasi, presisi, recall, dan F1-score.')
    st.write('- Interpretasi Hasil: Menganalisis hasil dari model untuk mengidentifikasi sumber-sumber utama emisi dan menganalisis kontribusi sektor-sektor tertentu.')

    st.header('Tampilan Dataset')
    df
    
    st.write("Baris: ", df.shape[0])
    st.write("Kolom: ", df.shape[1])

    st.text(df.info())
    

# Jika pilihan di sidebar adalah "Distribusi"
elif nav_selection == "Distribusi":
    # Menyiapkan data
    data = df.groupby('Federal organization')['Emissions (kt)'].sum()
    data_sorted = data.sort_values(ascending=False).head(10)

    # Membuat bar plot menggunakan Matplotlib
    plt.figure(figsize=(12, 8))
    data_sorted.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Organisasi Federal dengan Total Emisi Tertinggi')
    plt.xlabel('Organisasi Federal')
    plt.ylabel('Total Emisi (kt)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')

    # Menampilkan bar plot menggunakan Streamlit
    st.pyplot(plt.gcf())
    st.write('Dari visualisasi ini, terlihat bahwa "National Defence" memiliki total emisi tertinggi di antara 10 organisasi federal yang ditampilkan dalam data. Sedangkan organisasi "Natural Resource Canada" memiliki total emisi terendah di antara 10 organisasi tersebut.')

# Jika pilihan di sidebar adalah "Hubungan"
elif nav_selection == "Hubungan":
    # Menyiapkan data (memilih hanya kolom numerik)
    numeric_df = df.select_dtypes(include=[np.number])

    # Membuat matriks korelasi
    correlation_matrix = numeric_df.corr()

    # Plotting heatmap untuk menampilkan korelasi menggunakan Seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Heatmap Korelasi Antar Kolom Numerik')
    

    # Menampilkan heatmap menggunakan Streamlit
    st.pyplot(plt.gcf())
    st.write('Visualisasi berikut adalah sebuah heatmap yang menggambarkan matriks korelasi antara kolom-kolom numerik dalam suatu dataset. Setiap sel dalam heatmap menunjukkan nilai korelasi antara dua kolom yang sesuai. Kolom-kolom yang dianalisis dalam heatmap ini adalah "GHG scope", "Energy use (GJ)", dan "Emissions (kt)". Korelasi antara kolom "GHG scope" dan "Energy use (GJ)" adalah 0.16, antara "GHG scope" dan "Emissions (kt)" adalah 0.13, dan antara "Energy use (GJ)" dan "Emissions (kt)" adalah 0.95. Warna dari setiap sel menunjukkan kekuatan korelasi, di mana warna biru menunjukkan korelasi negatif, warna merah menunjukkan korelasi positif, dan intensitas warna mencerminkan kekuatan korelasi. Dari heatmap ini, dapat dilihat bahwa "Energy use (GJ)" dan "Emissions (kt)" memiliki korelasi positif yang kuat, ditunjukkan oleh warna merah yang intens di antara kedua kolom tersebut. Sedangkan "GHG scope" memiliki korelasi yang lemah dengan kedua kolom lainnya, ditunjukkan oleh warna yang mendekati biru atau merah yang lebih gelap.')

# Jika pilihan di sidebar adalah "Perbandingan dan Komposisi"
elif nav_selection == "Perbandingan dan Komposisi":
    # Menghitung nilai counts
    from_counts = df['GHG source'].value_counts()

    # Membuat pie chart
    fig, ax = plt.subplots()
    from_counts.plot(kind='pie', autopct='%1.1f%%', startangle=360, ax=ax)
    ax.set_ylabel('')  # Hapus label pada sumbu y
    ax.set_title('Distribusi Sumber GHG')

    # Menampilkan pie chart menggunakan Streamlit
    st.pyplot(fig)
    st.write('Visualisasi tersebut adalah sebuah diagram lingkaran (pie chart) yang menunjukkan proporsi data antara dua kategori, yaitu "facilities" dan "fleet" dari kolom "GHG source". Setiap bagian dari diagram lingkaran mewakili persentase dari total data.Dari diagram ini, terlihat bahwa kategori "facilities" memegang proporsi lebih besar dari total data, sebesar 53%, sedangkan kategori "fleet" memiliki proporsi sebesar 47%. Diagram lingkaran digunakan untuk dengan jelas memvisualisasikan proporsi atau persentase dari suatu kategori dalam sebuah dataset. Dalam hal ini, warna biru mewakili kategori "facilities", sementara warna oranye mewakili kategori "fleet".')

# Jika pilihan di sidebar adalah "Predict"
elif nav_selection == "Predict":
    # Memuat model yang telah dilatih
    model = load_model()

    # Memuat data
    # Misalkan df adalah dataframe yang berisi data Federal organization, Fiscal year, GHG source, GHG scope,
    # Energy use (GJ), Emissions (kt), Energy use category, dan Data_present
    df = pd.read_csv('data-mapping.csv')

    # Tampilkan form input untuk pengguna
    st.subheader("Predict Emisi Gas Rumah Kaca di Kanada")
    col1, col2 = st.columns(2)
    with col1 :
        federal_organization = st.selectbox("Pilih Federal Organization", df['Federal organization'].unique())
    with col2:
        fiscal_year = st.number_input("Masukan Fiscal Year", min_value=0)
    # ghg_source = st.text_input("GHG Source")
    with col1:
        ghg_scope = st.selectbox("Pilih GHG Scope", df['GHG scope'].unique())
    with col2:
        energy_use_gj = st.number_input("Masukan Energy Use (GJ)", min_value=0)
    with col1:
        emissions_kt = st.number_input("Masukan Emissions (kt)", min_value=0)
    with col2:
        energy_use_category = st.selectbox("Pilih Energy Use Category", df['Energy use category'].unique())
    data_present = st.selectbox("Pilih Data Present", df['Data_present'].unique())

    # Jika tombol "Predict" ditekan
    # Jika tombol "Predict" ditekan
    if st.button("Predict"):
        # Ubah input pengguna menjadi DataFrame
        input_data = pd.DataFrame({
            "Federal organization": [federal_organization],
            "Fiscal year": [fiscal_year],
            # "GHG source": [ghg_source],
            "GHG scope": [ghg_scope],
            "Energy use (GJ)": [energy_use_gj],
            "Emissions (kt)": [emissions_kt],
            "Energy use category": [energy_use_category],
            "Data_present": [data_present],
        })

        # Lakukan prediksi dengan model yang telah dilatih
        prediction = predict(input_data, model)

        # Tampilkan hasil prediksi sebagai teks
        if prediction == 0:
            msg = 'This Company too much using: **Facilities**'
        else:
            msg = 'This Company too much using: **Fleet**'

        # Tampilkan hasil prediksi sebagai teks menggunakan markdown
        st.markdown(msg)
