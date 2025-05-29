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
