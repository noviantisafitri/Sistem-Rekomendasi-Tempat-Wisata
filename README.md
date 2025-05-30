# Laporan Proyek Machine Learning - Novianti Safitri

## Project Overview

Pariwisata merupakan salah satu sektor penting dalam pembangunan ekonomi nasional dan regional. Di Indonesia, industri pariwisata memiliki potensi besar untuk terus tumbuh, terutama didorong oleh keindahan alam, keberagaman budaya, dan peningkatan aksesibilitas digital. Namun, di tengah semakin banyaknya informasi destinasi yang tersedia secara daring, calon wisatawan sering mengalami kesulitan dalam menentukan tujuan wisata yang paling sesuai dengan preferensi mereka. Hal ini menciptakan kebutuhan akan sistem yang dapat memberikan rekomendasi tempat wisata secara personal dan relevan.

Sistem rekomendasi merupakan salah satu penerapan kecerdasan buatan yang digunakan untuk menyaring informasi berdasarkan preferensi pengguna. Dalam konteks pariwisata, sistem ini dapat membantu pengguna dalam memilih destinasi berdasarkan minat, riwayat kunjungan, atau perilaku pengguna lain. Menurut penelitian oleh Gavalas et al. (2014), sistem rekomendasi dalam industri pariwisata dapat meningkatkan kepuasan pengguna dan efisiensi dalam perencanaan perjalanan. Dengan demikian, pengembangan sistem rekomendasi yang efektif dan adaptif menjadi sangat penting, khususnya di era digital yang sarat dengan informasi.

Proyek ini bertujuan untuk membangun sistem rekomendasi tempat wisata di Indonesia dengan pendekatan machine learning, menggunakan dua metode utama yaitu Content-Based Filtering dan Collaborative Filtering. Dataset yang digunakan bersumber dari Kaggle, yang mencakup informasi tentang tempat wisata serta penilaian pengguna terhadap tempat-tempat tersebut. Pendekatan content-based digunakan untuk merekomendasikan tempat berdasarkan kemiripan kategori wisata, sementara collaborative filtering memanfaatkan pola perilaku pengguna untuk menghasilkan rekomendasi.

Dengan adanya sistem rekomendasi ini, pengguna dapat menerima saran destinasi yang lebih sesuai dan personal, tanpa harus melakukan pencarian manual yang melelahkan. Di sisi lain, platform penyedia wisata juga dapat meningkatkan engagement dan kepuasan pengguna. Proyek ini diharapkan menjadi kontribusi nyata dalam pemanfaatan teknologi machine learning untuk mendukung pengambilan keputusan dalam sektor pariwisata.

Referensi:
1. Gavalas, D., Konstantopoulos, C., Mastakas, K., & Pantziou, G. (2014). Mobile recommender systems in tourism. Journal of Network and Computer Applications, 39, 319–333. https://doi.org/10.1016/j.jnca.2013.04.006
2. Kaggle. (2021). Indonesia Tourism Destination Dataset. Retrieved from https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination
3. Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.
4. Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734–749. https://doi.org/10.1109/TKDE.2005.99

## Business Understanding

### Problem Statements

- Pengguna sering kesulitan menemukan destinasi wisata yang sesuai dengan minat dan preferensi mereka.
- Tidak adanya sistem yang secara otomatis merekomendasikan tempat wisata berdasarkan data historis pengguna.

### Goals

- Membangun sistem rekomendasi yang dapat menyarankan tempat wisata berdasarkan kategori yang disukai pengguna.
- Mengembangkan model yang mampu merekomendasikan tempat wisata berdasarkan pola rating dari pengguna lain.

### Solution Statements

- Content-Based Filtering: merekomendasikan tempat wisata berdasarkan kemiripan kategori wisata yang pernah disukai pengguna.
- Collaborative Filtering: mempelajari pola rating antar pengguna untuk merekomendasikan tempat yang disukai pengguna serupa.

## Data Understanding

Dataset yang digunakan terdiri dari dua file utama, yaitu:

1. **`tourism_with_id.csv`** – berisi informasi tentang 437 tempat wisata di Indonesia.
2. **`tourism_rating.csv`** – berisi 10.000 data penilaian dari pengguna terhadap tempat wisata tersebut.

Dataset dapat diunduh dari [Kaggle - Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination).

### Fitur pada `tourism_with_id.csv` (sebelum preprocessing):

| Nama Kolom     | Tipe Data | Deskripsi                                                                 |
| -------------- | --------- | ------------------------------------------------------------------------- |
| `Place_Id`     | int64     | ID unik untuk setiap tempat wisata.                                       |
| `Place_Name`   | object    | Nama tempat wisata.                                                       |
| `Description`  | object    | Deskripsi singkat tentang tempat wisata.                                  |
| `Category`     | object    | Kategori tempat wisata (misal: Budaya, Taman Hiburan, Bahari, dll).       |
| `City`         | object    | Kota lokasi tempat wisata.                                                |
| `Price`        | int64     | Harga tiket masuk tempat wisata.                                          |
| `Rating`       | float64   | Rata-rata penilaian pengguna terhadap tempat wisata.                      |
| `Time_Minutes` | float64   | Estimasi waktu kunjungan (hanya tersedia pada sebagian entri).            |
| `Coordinate`   | object    | Informasi koordinat dalam bentuk dictionary (lat, long).                  |
| `Lat`          | float64   | Lintang geografis tempat wisata.                                          |
| `Long`         | float64   | Bujur geografis tempat wisata.                                            |
| `Unnamed: 11`  | float64   | Kolom kosong (null seluruh baris), tidak digunakan dalam analisis.        |
| `Unnamed: 12`  | int64     | Salinan numerik dari `Place_Id`, kemungkinan hasil kesalahan ekspor data. |

### Fitur pada `tourism_rating.csv`:

| Nama Kolom      | Tipe Data | Deskripsi                                         |
| --------------- | --------- | ------------------------------------------------- |
| `User_Id`       | int64     | ID pengguna yang memberikan rating.               |
| `Place_Id`      | int64     | ID tempat wisata yang diberi rating.              |
| `Place_Ratings` | int64     | Nilai rating yang diberikan pengguna (skala 1–5). |

* Jumlah entri tempat wisata: **437**
* Jumlah data rating: **10.000**
* Rata-rata nilai rating: **3.07**
* Nilai rating berkisar dari **1 hingga 5**, dengan distribusi relatif merata.

## Data Preparation

Tahap persiapan data dilakukan untuk memastikan bahwa dataset bersih, relevan, dan siap digunakan dalam proses pelatihan model. Berikut adalah langkah-langkah yang dilakukan:

1. **Menghapus Kolom Tidak Relevan**
   Beberapa kolom seperti `Description`, `Coordinate`, `Unnamed: 11`, dan `Unnamed: 12` dihapus karena tidak memberikan kontribusi signifikan terhadap model atau mengandung nilai kosong sepenuhnya.

2. **Menangani Nilai Kosong dan Duplikat**
   Dataset dicek untuk nilai kosong dan duplikat. Pada `df_ratings`, ditemukan adanya data duplikat dan diatasi menggunakan fungsi `df_ratings.drop_duplicates()` untuk menghindari bias dalam pembelajaran model.

3. **Encoding Fitur Kategorikal**
   Kolom `User_Id` dan `Place_Id` diubah menjadi indeks numerik (integer encoding) agar dapat digunakan dalam model Collaborative Filtering berbasis embedding.

4. **TF-IDF Vectorization (Content-Based Filtering)**
   Fitur `Category` dari `df_places` diolah menggunakan teknik **TF-IDF Vectorization**. Proses ini mengubah data teks kategori menjadi representasi numerik berbasis frekuensi, yang kemudian digunakan dalam perhitungan cosine similarity antar tempat wisata.

5. **Normalisasi Rating**
   Nilai rating pada `Place_Ratings` dinormalisasi ke rentang 0–1 agar sesuai dengan fungsi aktivasi `sigmoid` pada model Collaborative Filtering.

6. **Pembagian Data Latih dan Validasi**
   Dataset `df_ratings` dibagi menjadi data latih dan data validasi dengan rasio 80:20 menggunakan indeks slicing. Ini bertujuan untuk mengevaluasi kemampuan generalisasi model terhadap data yang belum pernah dilihat sebelumnya.

## Modeling
Pada tahap ini, dilakukan pembangunan dua sistem rekomendasi berbeda: **Content-Based Filtering** dan **Collaborative Filtering**. Setiap pendekatan dirancang untuk menghasilkan rekomendasi tempat wisata yang relevan berdasarkan data yang tersedia.

### Content-Based Filtering

Sistem rekomendasi ini bekerja dengan cara menganalisis kemiripan antar tempat wisata berdasarkan fitur konten, yaitu **kategori wisata**. Sebelumnya, fitur `Category` telah dikonversi ke dalam representasi numerik menggunakan teknik **TF-IDF Vectorization** (proses ini dijelaskan pada bagian *Data Preparation*). Setelah itu, kemiripan antar tempat dihitung menggunakan metode **Cosine Similarity**.

Fungsi rekomendasi kemudian mengembalikan daftar Top-5 tempat yang paling mirip dengan tempat yang dipilih pengguna.

**Kelebihan:**

* Tidak memerlukan data pengguna lain.
* Cocok untuk pengguna baru (*cold-start* user) karena hanya bergantung pada konten.

**Kekurangan:**

* Rekomendasi terbatas pada informasi yang tersedia.
* Tidak mempertimbangkan preferensi pengguna lain.

**Contoh Output (Top-5 rekomendasi untuk tempat: "Monumen Nasional"):**

| No | Place\_Name                         | Category |
| -- | ----------------------------------- | -------- |
| 1  | Candi Sewu                          | Budaya   |
| 2  | Museum Benteng Vredeburg Yogyakarta | Budaya   |
| 3  | Museum Satria Mandala               | Budaya   |
| 4  | Kyotoku Floating Market             | Budaya   |
| 5  | Bandros City Tour                   | Budaya   |


### Collaborative Filtering

Sistem ini menggunakan pendekatan **deep learning berbasis embedding** untuk merepresentasikan pengguna dan tempat dalam bentuk vektor. Model dilatih menggunakan data `Place_Ratings` yang telah dinormalisasi dan menggunakan fungsi loss **BinaryCrossentropy**.

Setelah pelatihan, model digunakan untuk memprediksi tempat-tempat yang mungkin disukai oleh pengguna berdasarkan interaksi pengguna serupa.

**Kelebihan:**

* Dapat menemukan pola tersembunyi dari interaksi pengguna.
* Lebih fleksibel dalam memahami hubungan kompleks antara pengguna dan item.

**Kekurangan:**

* Membutuhkan banyak data interaksi untuk performa optimal.
* Kurang cocok untuk pengguna baru atau tempat baru (*cold-start* problem).

**Contoh Output (Top-10 rekomendasi untuk salah satu pengguna):**

| No | Place\_Name                      | Category      |
| -- | -------------------------------- | ------------- |
| 1  | Monumen Yogya Kembali            | Budaya        |
| 2  | Bukit Bintang Yogyakarta         | Taman Hiburan |
| 3  | Grojogan Watu Purbo Bangunrejo   | Taman Hiburan |
| 4  | Keraton Surabaya                 | Budaya        |
| 5  | Monumen Tugu Pahlawan            | Budaya        |
| 6  | House of Sampoerna               | Budaya        |
| 7  | Taman Hiburan Rakyat             | Taman Hiburan |
| 8  | Taman Mundu                      | Taman Hiburan |
| 9  | Museum Mpu Tantular              | Budaya        |
| 10 | Taman Air Mancur Menari Kenjeran | Taman Hiburan |

## Evaluation

Model **Collaborative Filtering** dievaluasi menggunakan dua metrik utama, yaitu:

1. **Root Mean Squared Error (RMSE)**
2. **Loss (Binary Crossentropy)**

### Penjelasan Metrik

**Root Mean Squared Error (RMSE)** digunakan untuk mengukur seberapa jauh prediksi model terhadap nilai rating yang sebenarnya. Rumus perhitungannya adalah:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_{true}^{(i)} - y_{pred}^{(i)})^2}
$$

Semakin kecil nilai RMSE, semakin baik kualitas prediksi model.

**Loss** dihitung menggunakan fungsi **Binary Crossentropy** karena output model berada dalam skala \[0,1] hasil dari aktivasi sigmoid. Loss ini menunjukkan seberapa besar kesalahan model dalam memetakan prediksi terhadap nilai target.

### Hasil Evaluasi

Berdasarkan pelatihan model dan proses validasi, diperoleh hasil akhir sebagai berikut:

* **Validation RMSE**: `0.3515`
* **Validation Loss**: `0.6962`

### Analisis Hasil

Nilai RMSE yang relatif rendah menunjukkan bahwa model mampu memprediksi rating dengan cukup akurat setelah proses normalisasi. Sementara itu, stabilnya nilai Loss pada data validasi dibandingkan dengan penurunan Loss pada data pelatihan menunjukkan adanya **overfitting ringan**, namun masih dalam batas yang dapat diterima. Secara keseluruhan, model Collaborative Filtering berhasil membangun sistem rekomendasi dengan performa yang baik untuk skala dataset ini.

---
