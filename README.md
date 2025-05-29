# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Dalam era digital saat ini, informasi mengenai destinasi wisata sangat melimpah dan dapat membingungkan pengguna dalam menentukan tujuan liburan yang sesuai dengan preferensi mereka. Untuk mengatasi masalah tersebut, dibutuhkan sistem rekomendasi yang dapat membantu pengguna memilih tempat wisata secara personal. Proyek ini bertujuan untuk membangun sistem rekomendasi tempat wisata di Indonesia dengan memanfaatkan pendekatan content-based filtering dan collaborative filtering. Dataset yang digunakan berasal dari Kaggle dengan judul "Indonesia Tourism Destination".

## Business Understanding

### Problem Statements

- Pengguna sering kesulitan menemukan destinasi wisata yang sesuai dengan minat dan preferensi mereka.
- Tidak adanya sistem yang secara otomatis merekomendasikan tempat wisata berdasarkan data historis pengguna.

### Goals

- Membangun sistem rekomendasi yang dapat menyarankan tempat wisata berdasarkan kategori yang disukai pengguna.
- Mengembangkan model yang mampu merekomendasikan tempat wisata berdasarkan pola rating dari pengguna lain.

### Solution Approach

#### Solution Statements

- Content-Based Filtering: merekomendasikan tempat wisata berdasarkan kemiripan kategori wisata yang pernah disukai pengguna.
- Collaborative Filtering: mempelajari pola rating antar pengguna untuk merekomendasikan tempat yang disukai pengguna serupa.

## Data Understanding

Dataset terdiri dari dua file utama:
- `tourism_with_id.csv`: berisi informasi tempat wisata (437 entri).
- `tourism_rating.csv`: berisi data rating pengguna terhadap tempat wisata (10.000 entri).

Link dataset: [Kaggle - Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

Fitur pada `tourism_with_id.csv`:
- Place_Id, Place_Name, Description, Category, City, Price, Rating, Time_Minutes, Coordinate, Lat, Long

Fitur pada `tourism_rating.csv`:
- User_Id, Place_Id, Place_Ratings

EDA menunjukkan bahwa rata-rata rating yang diberikan adalah 3.07 dengan sebaran yang relatif merata. Kolom kosong dan tidak relevan dihapus untuk meningkatkan kualitas data.

## Data Preparation

- Menghapus kolom yang tidak relevan seperti Description, Coordinate, dan kolom kosong.
- Mengecek nilai kosong dan duplikat pada dataset.
- Normalisasi rating untuk collaborative filtering agar sesuai dengan aktivasi sigmoid.
- Encoding User_Id dan Place_Id menjadi indeks numerik.

Tahapan ini penting untuk memastikan data bersih dan siap digunakan oleh model pembelajaran mesin.

## Modeling

### Content-Based Filtering
Menggunakan TF-IDF Vectorizer pada fitur Category untuk membentuk matriks fitur, lalu dihitung cosine similarity antar tempat. Fungsi rekomendasi dibuat untuk menghasilkan top-5 tempat terdekat berdasarkan kemiripan.

**Kelebihan:**
- Tidak bergantung pada data pengguna lain.
- Dapat bekerja meskipun pengguna baru (cold start).

**Kekurangan:**
- Rekomendasi terbatas pada informasi yang sudah dimiliki.

### Collaborative Filtering
Menggunakan model deep learning berbasis embedding untuk pengguna dan tempat. Model dilatih dengan data rating yang dinormalisasi dan menggunakan fungsi loss BinaryCrossentropy.

**Kelebihan:**
- Dapat menangkap pola preferensi pengguna dari data interaksi.

**Kekurangan:**
- Membutuhkan banyak data interaksi.
- Rentan terhadap cold-start user.

## Evaluation

Model dievaluasi menggunakan Root Mean Squared Error (RMSE) dan Loss pada data training dan validation.

- RMSE pada training terus menurun, sedangkan validation stabil, menandakan overfitting ringan.
- Visualisasi loss menunjukkan hasil serupa.

**Formula RMSE:**
RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))

RMSE mengukur seberapa jauh prediksi model dari nilai rating sebenarnya. Semakin rendah nilai RMSE, semakin baik performa model.

---

_Catatan:_
- Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan.
