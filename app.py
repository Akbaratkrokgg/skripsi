import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import davies_bouldin_score
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Variabel global untuk menyimpan kolom yang dipilih
selected_columns = []

# Fungsi untuk membaca data dari file CSV
def read_csv(file_path):
    return pd.read_csv(file_path)

def find_epsilon(data):
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, _ = nbrs.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    gradient = np.gradient(distances)
    elbow_index = np.argmax(gradient)
    epsilon = distances[elbow_index]
    return epsilon

# Fungsi untuk melakukan clustering menggunakan DBSCAN
def perform_clustering(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)
    return clusters

# Fungsi untuk mengevaluasi hasil clustering menggunakan Davies-Bouldin Index (DBI)
def evaluate_clustering(data, clusters):
    if len(set(clusters)) > 1:
        return davies_bouldin_score(data, clusters)
    else:
        return np.nan

# Fungsi untuk menampilkan informasi negara dalam suatu cluster
def show_cluster_info(data, cluster_id, string_column):
    cluster_data = data[data['Cluster'] == cluster_id]
    st.write(f"Cluster {cluster_id}")
    st.write("Number of Entries:", len(cluster_data))
    st.write(cluster_data[[string_column, 'Cluster']])

# Main function
def main():
    st.title("Clustering Wisatawan Mancanegara")
    st.sidebar.title("DBSCAN Parameters")

    # Input file CSV
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(data.head())

        # Identifikasi kolom string pertama dalam dataset
        string_columns = [col for col in data.columns if data[col].dtype == 'object']
        if not string_columns:
            st.error("Dataset tidak memiliki kolom string.")
            return
        string_column = string_columns[0]

        # Pilih kolom yang akan di-cluster
        selected_columns = st.multiselect("Select columns to cluster", data.columns)

        # Cek apakah ada kolom non-numerik yang dipilih
        non_numeric_columns = [column for column in selected_columns if not pd.api.types.is_numeric_dtype(data[column])]
        if non_numeric_columns:
            st.error(f"Kolom berikut tidak berisi nilai numerik: {', '.join(non_numeric_columns)}")
            return

        # Pastikan hanya kolom numerik yang dipilih
        selected_columns = [column for column in selected_columns if pd.api.types.is_numeric_dtype(data[column])]
        if len(selected_columns) == 0:
            st.warning("Pastikan Pilih Kolom Numeric.")
            return

        # Buat subset data berdasarkan kolom yang dipilih
        data_selected = data[selected_columns]

        # Normalisasi data menggunakan Min-Max scaler
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data_selected)

        # Tampilkan data setelah normalisasi
        st.write("Normalized Data:")
        st.write(pd.DataFrame(data_normalized, columns=data_selected.columns))
        st.write("Normalized Data Description:")
        st.write(pd.DataFrame(data_normalized, columns=data_selected.columns).describe())

        # Temukan nilai epsilon menggunakan metode siku (elbow method)
        st.write("Mencari Rekomendasi Nilai Epsilon...")
        epsilon = find_epsilon(data_normalized)

        # Rekomendasikan nilai epsilon kepada pengguna
        st.write("Rekomendasi Epsilon:", epsilon)

        # Input nilai eps dan min_samples sebanyak lima kali
        epsilons = st.sidebar.text_area("Masukkan 5 Nilai Epsilon (pisahkan dengan koma):")
        min_samples = st.sidebar.text_area("Masukkan 5 Nilai MinPts (pisahkan dengan koma):")
        eps_list = epsilons.split(',')
        min_samples_list = min_samples.split(',')

        # Tampilkan pesan jika nilai belum dimasukkan
        if not epsilons or not min_samples:
            st.warning("Masukkan nilai eps dan minpts")
            return
        
        if len(eps_list) != 5 or len(min_samples_list) != 5:
            st.error("Nilai yang dimasukkan tidak sesuai dengan perintah")
            return

        # Pastikan semua nilai adalah angka
        try:
            eps_list = [float(eps) for eps in eps_list]
            min_samples_list = [int(min_samples) for min_samples in min_samples_list]
        except ValueError:
            st.error("All values must be numeric.")
            return

        # Inisialisasi list untuk menyimpan nilai eps dan minPts
        eps_values = []
        min_samples_values = []

        # Inisialisasi list untuk menyimpan hasil clustering dan evaluasi DBI
        cluster_results = []
        dbi_results = []

        # Inisialisasi variabel untuk menyimpan informasi DBI score terbaik
        best_dbi_score = np.inf
        best_eps = None
        best_minPts = None
        best_num_clusters = None
        best_noise = None
        best_iteration = None

        best_cluster_data = None

        for eps in eps_list:
            for min_samples in min_samples_list:
                # Lakukan clustering
                clusters = perform_clustering(data_normalized, eps, min_samples)
                num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                cluster_results.append(num_clusters)

                # Evaluasi hasil clustering
                if num_clusters > 1:
                    dbi_score = evaluate_clustering(data_normalized, clusters)
                else:
                    dbi_score = np.nan
                dbi_results.append(dbi_score)

                # Tambahkan nilai eps dan minPts ke dalam list
                eps_values.append(eps)
                min_samples_values.append(min_samples)

                # Tambahkan kolom Cluster ke data asli
                data['Cluster'] = clusters

                # Tampilkan hasil clustering
                st.subheader(f"Eps: {eps}, MinPts: {min_samples}")
                st.write("Jumlah Cluster:", num_clusters)
                st.write("Nilai DBI:", dbi_score)

                # Tampilkan informasi noise
                noise_countries = data[data['Cluster'] == -1][string_column]
                st.write("Jumlah Noise:", len(noise_countries))
                st.write(noise_countries)

                # Tampilkan informasi cluster
                for cluster_id in range(num_clusters):
                    show_cluster_info(data, cluster_id, string_column)

                # Simpan informasi tentang DBI score terbaik
                if dbi_score < best_dbi_score:
                    best_dbi_score = dbi_score
                    best_eps = eps
                    best_minPts = min_samples
                    best_num_clusters = num_clusters
                    best_noise = noise_countries.tolist()
                    best_iteration = f"Eps: {eps}, MinPts: {min_samples}"
                    best_cluster_data = data.copy()

        # Tampilkan tabel rangkuman saat tombol "Result" diklik
        with st.expander("Hasil Clustering", expanded=False):
            results_df = pd.DataFrame({
                "Epsilon": eps_values,
                "MinPts": min_samples_values,
                "Jumlah Cluster": cluster_results,
                "Nilai DBI": dbi_results
            })
            st.write("Rangkuman:")
            st.write(results_df)

            # DBI Score terbaik dan informasi tentang epsilon, minPts, dan jumlah cluster
            st.write("DBI terbaik:", best_dbi_score)
            st.write("Jumlah clustering terbaik:", best_num_clusters)
            st.write("Noise pada hasil Iterasi terbaik:", best_noise)
            st.write("Epsilon dan MinPts terbaik:", best_iteration)

            # Menampilkan informasi tabel untuk semua cluster dari hasil terbaik
            if best_cluster_data is not None:
                st.write("Informasi Cluster dari DBI terbaik:")
                for cluster_id in range(best_num_clusters):
                    show_cluster_info(best_cluster_data, cluster_id, string_column)

if __name__ == "__main__":
    main()
