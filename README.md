# ✈️ Airline Ticket Price Analysis & Prediction

Analisis eksploratif dan model machine learning untuk memahami **apa yang membentuk harga tiket pesawat** di India, serta memprediksi harganya dengan **R² ≈ 0.985** pada data yang belum pernah dilihat model.

> **Ringkasan singkat:** Dari 300 ribu data tiket, saya menemukan bahwa harga dipengaruhi terutama oleh kelas kabin, maskapai, dan waktu pemesanan, dengan pola yang berbeda antara Economy dan Business. Random Forest dipilih sebagai model akhir dengan MAE sekitar ₹1.300, turun ±74% dibanding baseline sederhana.

---

## 1. Latar Belakang & Tujuan

Harga tiket pesawat sulit ditebak: rute yang sama bisa berbeda harga puluhan persen tergantung kapan dan di maskapai mana kita membeli. Project ini menjawab dua hal:

1. **Analitis:** aktor-faktor apa saja yang paling dominan memengaruhi harga tiket, dan bagaimana pola hubungan variabel tersebut terhadap tarif akhir?
2. **Prediktif:** Seberapa akurat kita dapat memperkirakan harga tiket pesawat berdasarkan informasi penerbangan yang tersedia sebelum pembelian?

Pertanyaan bisnis yang dijawab di EDA:

- Kapan waktu terbaik membeli tiket?
- Apakah penerbangan transit selalu lebih murah dari penerbangan langsung?
- Apakah Vistara benar-benar lebih mahal dari Air India, atau hanya karena komposisi kelasnya?
- Apakah durasi penerbangan menentukan harga?

## 2. Dataset

| Item | Detail |
|---|---|
| Jumlah data | 300.153 baris, 11 kolom |
| Missing value / duplikat | Tidak ada |
| Target | `price` (INR) — rata-rata ₹20.948, median ₹7.426 |
| Fitur numerik | `duration` (jam), `days_left` (hari sebelum berangkat, 1–49) |
| Fitur kategorikal | `airline` (6), `source_city` (6), `destination_city` (6), `departure_time` (6), `arrival_time` (6), `stops` (3), `class` (2), `flight` (1.544 kode) |

Sumber data: _(isi dengan link dataset, mis. Kaggle)_

Distribusi harga bersifat **bimodal** (kelompok harga rendah dan tinggi), karena kelas Economy dan Business berada di rentang yang sangat berbeda. Ini menjadi petunjuk awal bahwa `class` adalah fitur dominan.

## 3. Pendekatan

```
Raw data → Split (anti-leakage) → EDA (train only) → Preprocessing → Modeling → CV & Tuning → Evaluasi test → Export model
```

### 3.1 Pencegahan data leakage (bagian yang paling saya perhatikan)

Banyak penerbangan di dataset ini punya fitur identik tetapi harga berbeda. Jika baris-baris itu tersebar acak ke train dan test, model bisa "menghafal" dan skor test jadi terlalu optimis. Karena itu:

- **`GroupShuffleSplit`** dengan grup berupa hash seluruh fitur input, sehingga kombinasi input yang identik **selalu berada di satu partisi yang sama**. Ada `assert` yang memverifikasi tidak ada grup yang bocor antar partisi.
- Pembagian **60 / 20 / 20** → train 180.090, validation 60.050, test 60.013. Proporsi kelas tetap konsisten (±69% Economy, ±31% Business).
- **EDA hanya dilakukan pada data train**, sehingga tidak ada informasi dari validation/test yang memengaruhi keputusan fitur.
- Kolom `flight` dibuang dari model (kardinalitas tinggi, berisiko menjadi pengenal unik).
- **`GroupKFold`** dipakai saat cross-validation dengan grup yang sama.

### 3.2 EDA yang tidak berhenti di korelasi

Selain visualisasi standar, saya sengaja menguji temuan dengan **matched comparison**: membandingkan harga hanya pada kondisi yang setara (rute, kelas, jumlah transit, booking window, dsb.), lalu mengukur ketidakpastiannya dengan **bootstrap per rute** (bukan per baris, karena baris dalam satu rute saling berkaitan). Ini untuk menghindari kesimpulan keliru akibat confounding, misalnya "Vistara lebih mahal" yang bisa saja hanya karena Vistara punya lebih banyak penerbangan Business.

### 3.3 Preprocessing

`ColumnTransformer` di dalam `Pipeline` (sehingga fit hanya pada train):

- `StandardScaler` untuk `duration`, `days_left` (hanya untuk model linear)
- `OrdinalEncoder` untuk `stops` (zero < one < two_or_more)
- `OneHotEncoder` (`handle_unknown='ignore'`) untuk kolom kategorikal lainnya

## 4. Temuan Utama dari EDA

| # | Temuan | Bukti |
|---|---|---|
| 1 | **Kelas kabin adalah pembeda harga terbesar.** Median Business ≈ **9,2×** Economy. | Boxplot & median per kelas |
| 2 | **Waktu pemesanan sangat berpengaruh di Economy, nyaris tidak di Business.** | Economy: median ₹4.700 (31–49 hari) → ₹13.128 (1–3 hari). Pada penerbangan yang dicocokkan, premi beli mepet ≈ **+127%** (CI 95%: 117–139%). Business hanya ≈ **+1,7%** (CI 0–3,7%). |
| 3 | **Transit tidak otomatis lebih murah.** | Pada kondisi setara, 1 transit lebih mahal dari nonstop: ≈ **+₹1.742** (Economy) dan ≈ **+₹25.345** (Business). |
| 4 | **Vistara lebih mahal dari Air India, dan bukan hanya karena komposisi kelas.** | Selisih rata-rata mentah ₹6.843 turun menjadi ₹4.032 setelah standardisasi kelas, tetapi tidak hilang. Pada sel yang dicocokkan, Vistara lebih mahal terutama di Business. |
| 5 | **Durasi bukan penentu harga yang berdiri sendiri.** | Korelasi agregat durasi–harga +0,20, tetapi menjadi **−0,04** saat dibandingkan dalam itinerary yang sama. Hubungannya sebagian besar efek dari karakteristik penerbangan lain. |
| 6 | **Variasi harga antar rute berbeda.** | Rute Economy seperti Mumbai–Chennai dan Mumbai–Hyderabad punya dispersi relatif (IQR/median) tertinggi (~0,86). |

**Implikasi praktis:** untuk tiket Economy, beli minimal 2 minggu sebelumnya. Harga relatif stabil pada rentang 15–49 hari dan melonjak tajam di sekitar 7 hari terakhir.

> ⚠️ Semua temuan di atas bersifat **observasional**. Data ini menunjukkan hubungan, bukan bukti sebab-akibat.

## 5. Modeling

Lima model dibandingkan pada **validation set**, dengan baseline "rata-rata harga per kelas":

| Model | Val MAE (₹) | Val R² | Train R² | Waktu fit |
|---|---:|---:|---:|---:|
| **Random Forest** | 1.303 | **0,9851** | 0,9891 | 19 dtk |
| Decision Tree | **1.172** | 0,9773 | 0,9991 | 11 dtk |
| Gradient Boosting | 2.933 | 0,9527 | 0,9524 | 5 dtk |
| Ridge Regression | 4.527 | 0,9097 | 0,9097 | 0,1 dtk |
| Linear Regression | 4.527 | 0,9097 | 0,9097 | 0,2 dtk |
| _Baseline (mean per kelas)_ | _4.921_ | _0,8783_ | – | – |

**Catatan interpretasi:**

- Model berbasis pohon jauh mengungguli model linear → hubungan fitur dan harga **non-linear** (konsisten dengan temuan interaksi kelas × maskapai × waktu pemesanan di EDA).
- Decision Tree punya MAE validation sedikit lebih rendah, tetapi **overfit berat** (Train MAE ₹74 vs Val MAE ₹1.172). Random Forest dipilih karena R² tertinggi dan gap train–validation yang jauh lebih kecil.
- Baseline sudah kuat (R² 0,878) karena `class` saja menjelaskan sebagian besar variasi. Nilai tambah model terlihat di **MAE yang turun ±74%** (₹4.921 → ₹1.303).

### Tuning

`RandomizedSearchCV` (8 kandidat, `GroupKFold` 5-fold) pada Random Forest menghasilkan CV R² **0,9841**, **tidak lebih baik** dari konfigurasi default (0,9851). Maka model final memakai parameter default (`n_estimators=100`, `min_samples_leaf=5`, `max_features=0.5`).

## 6. Hasil Akhir pada Test Set

Model final di-fit ulang pada train + validation, lalu dievaluasi **sekali** pada test set yang terpisah dan bebas dari kebocoran grup:

| Metrik | Nilai |
|---|---:|
| R² | **0,9859** |
| MAE | **₹1.239** |
| RMSE | ₹2.685 |

(Sebelum di-fit ulang, performa test: MAE ₹1.307 dan R² 0,9851, hampir identik dengan validation → model stabil.)

Model diekspor sebagai `airline_price_model.joblib` beserta `model_metadata.json` (versi library, daftar kategori valid, rentang input, skor test) agar mudah dipakai di deployment.

## 7. Keterbatasan

- **Data adalah snapshot** dengan horizon pemesanan 1–49 hari dan hanya 6 kota. Model tidak boleh dipakai untuk memperkirakan harga di luar rentang itu.
- **Temuan EDA bersifat korelasional**, dan matched comparison hanya mencakup sel yang punya cukup observasi (cakupan baris dilaporkan di notebook).
- Beberapa input identik punya harga berbeda (ketidakpastian yang tidak bisa dijelaskan fitur yang tersedia), sehingga ada **batas atas akurasi**.
- MAE ₹1.300 terlihat kecil untuk Business, tetapi **relatif besar untuk tiket Economy** (median ±₹6.000–7.000). Error belum dipecah per kelas atau per rute.
- Data tidak memuat faktor seperti musim liburan, harga bahan bakar, atau ketersediaan kursi.

## 8. Tech Stack

Python · pandas · NumPy · scikit-learn (Pipeline, ColumnTransformer, GroupShuffleSplit, GroupKFold, RandomizedSearchCV) · matplotlib · seaborn · joblib

## 9. Struktur Repo & Cara Menjalankan

```
.
├── App / Feature/              # Folder untuk kode aplikasi (Streamlit / FastAPI)
├── Model/                      # Folder untuk model final dan metadata
│   ├── airline_price_model.joblib
│   └── model_metadata.json
├── raw_data/
│   └── airlines_flights_data.csv # Dataset 
├── Airline_Project_v2.ipynb    # Notebook analisis lengkap
└── README.md
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
jupyter notebook Airline_Project_v2.ipynb
```

Contoh memuat model:

```python
import joblib, pandas as pd

model = joblib.load("airline_price_model.joblib")
sample = pd.DataFrame([{
    "airline": "Vistara", "source_city": "Delhi", "departure_time": "Morning",
    "stops": "zero", "arrival_time": "Afternoon", "destination_city": "Mumbai",
    "class": "Economy", "duration": 2.17, "days_left": 20,
}])
print(model.predict(sample))  
```

---
