# Klasifikasi Gambar Menggunakan Convolutional Neural Network (CNN)

## Gambaran Umum Proyek

Proyek ini bertujuan untuk membangun model *deep learning* berbasis **Convolutional Neural Network (CNN)** yang dapat mengklasifikasikan gambar berwarna ke dalam 10 kategori objek menggunakan dataset CIFAR-10.

Tujuan utama proyek ini adalah menerapkan alur kerja *deep learning* untuk klasifikasi gambar, mulai dari eksplorasi data, *preprocessing*, augmentasi data, perancangan arsitektur CNN, pelatihan, hingga evaluasi performa model per kelas.

Dalam proyek ini, dibangun satu model CNN dengan arsitektur *Sequential* yang terdiri dari tiga blok konvolusi dan satu *head* klasifikasi.

## Tujuan Proyek

- Memahami struktur dan distribusi dataset CIFAR-10.
- Melakukan *preprocessing* data gambar (normalisasi piksel dan *one-hot encoding* label).
- Menerapkan augmentasi data untuk mengurangi risiko *overfitting*.
- Membangun arsitektur CNN menggunakan Keras (TensorFlow).
- Melatih model dengan bantuan *callback* (`EarlyStopping` dan `ReduceLROnPlateau`).
- Mengevaluasi performa model menggunakan akurasi, *classification report*, dan *confusion matrix*.
- Memvisualisasikan kurva pelatihan dan contoh prediksi model.
- Menyimpan model agar dapat digunakan kembali.

## Dataset

Dataset yang digunakan dalam proyek ini adalah **CIFAR-10**, dataset *benchmark* populer di bidang *Computer Vision* yang dapat dimuat langsung melalui Keras.

- **Sumber:** CIFAR-10 (Alex Krizhevsky, University of Toronto), dimuat melalui `tensorflow.keras.datasets.cifar10`
- **Jumlah Data Training Awal:** 50.000 gambar
- **Jumlah Data Testing:** 10.000 gambar
- **Dimensi Gambar:** 32 × 32 piksel, 3 channel warna (RGB)
- **Jumlah Kelas:** 10
- **Nilai Piksel Asli:** 0 sampai 255

Daftar kelas:

| Indeks | Kelas | Indeks | Kelas |
|:---:|---|:---:|---|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

Distribusi kelas pada data training seimbang, yaitu setiap kelas memiliki jumlah gambar yang sama. Pembagian data yang digunakan:

| Subset | Jumlah Gambar | Keterangan |
|---|---:|---|
| Training | 45.000 | Digunakan untuk melatih model (dengan augmentasi) |
| Validation | 5.000 | 5.000 gambar pertama dari data training, untuk memantau pelatihan |
| Testing | 10.000 | Digunakan hanya untuk evaluasi akhir |

Dataset diunduh secara otomatis oleh Keras saat pertama kali dijalankan.

## Tools dan Teknologi

Tools dan library yang digunakan dalam proyek ini:

- **Python 3.12:** Bahasa pemrograman utama.
- **TensorFlow 2.20 dan Keras:** Pembangunan dan pelatihan model *deep learning*.
- **NumPy:** Operasi numerik dan array.
- **Matplotlib dan Seaborn:** Visualisasi data dan hasil.
- **Scikit-learn:** *Classification report* dan *confusion matrix*.
- **Google Colab (GPU):** Lingkungan menjalankan notebook dan pelatihan model.

## Tahapan Pengerjaan Proyek

1. **Import Library:** Mengimpor library, mengatur *random seed* (42), dan mengecek versi TensorFlow serta ketersediaan GPU.
2. **Load & Eksplorasi Dataset:** Memuat CIFAR-10, melihat dimensi data, menampilkan 32 sampel gambar, dan memeriksa distribusi kelas.
3. **Normalisasi Piksel:** Mengubah nilai piksel dari rentang 0–255 menjadi 0,0–1,0 dengan membagi 255.
4. **One-Hot Encoding:** Mengubah label angka menjadi vektor biner sepanjang 10 elemen.
5. **Pembagian Validation Set:** Memisahkan 5.000 gambar dari data training sebagai data validasi.
6. **Augmentasi Data:** Menggunakan `ImageDataGenerator` untuk memperbanyak variasi data training secara buatan.
7. **Perancangan Arsitektur CNN:** Membangun model *Sequential* dengan tiga blok konvolusi dan *head* klasifikasi.
8. **Kompilasi dan Training:** Melatih model menggunakan optimizer Adam dengan *callback* `EarlyStopping` dan `ReduceLROnPlateau`.
9. **Evaluasi Model:** Mengevaluasi model pada data test menggunakan loss, akurasi, dan *classification report*.
10. **Visualisasi Hasil:** Membuat kurva akurasi dan loss, *confusion matrix* ternormalisasi, serta contoh prediksi dengan bar probabilitas.
11. **Penyimpanan Model:** Menyimpan model ke file `cnn_cifar10_model.h5`.

## Augmentasi Data

Augmentasi hanya diterapkan pada data training, sedangkan data validasi dan test tidak diubah.

| Teknik | Nilai |
|---|---|
| `rotation_range` | 15 derajat |
| `width_shift_range` | 0,1 |
| `height_shift_range` | 0,1 |
| `horizontal_flip` | True |
| `zoom_range` | 0,1 |
| `fill_mode` | nearest |

## Arsitektur Model

Model CNN dibangun dengan Sequential API (`CNN_CIFAR10`):

| Blok | Lapisan | Fungsi |
|---|---|---|
| Blok 1 | Conv2D(32) × 2 → BatchNorm → MaxPooling → Dropout(0,25) | Deteksi fitur dasar: tepi dan warna |
| Blok 2 | Conv2D(64) × 2 → BatchNorm → MaxPooling → Dropout(0,30) | Deteksi fitur menengah: bentuk dan tekstur |
| Blok 3 | Conv2D(128) × 2 → BatchNorm → MaxPooling → Dropout(0,40) | Deteksi fitur kompleks: bagian objek |
| Head | Flatten → Dense(256) → BatchNorm → Dropout(0,50) → Dense(10, softmax) | Klasifikasi akhir |

Setiap `Conv2D` menggunakan kernel 3×3, aktivasi ReLU, dan `padding='same'`.

Jumlah parameter model:

| Jenis Parameter | Jumlah |
|---|---:|
| Total | 816.938 (3,12 MB) |
| Trainable | 815.530 (3,11 MB) |
| Non-trainable | 1.408 (5,50 KB) |

## Konfigurasi Training

| Parameter | Nilai |
|---|---|
| Optimizer | Adam (learning rate 0,001) |
| Loss function | Categorical Crossentropy |
| Metrik | Accuracy |
| Batch size | 64 |
| Maksimum epoch | 60 |
| `EarlyStopping` | Memantau `val_accuracy`, patience 10, `restore_best_weights=True` |
| `ReduceLROnPlateau` | Memantau `val_loss`, faktor 0,5, patience 5, `min_lr=1e-6` |

Pada epoch pertama, akurasi training 38,05% dan akurasi validasi 46,72%. Akurasi validasi kemudian meningkat bertahap, misalnya menjadi 72,00% pada epoch 9 dan 76,46% pada epoch 17 (sesuai log yang tampil pada notebook).

## Performa Model

Evaluasi dilakukan menggunakan 10.000 gambar data test yang tidak digunakan selama pelatihan.

| Metrik Evaluasi | Nilai |
|---|---:|
| Test Loss | 0,4315 |
| Test Accuracy | 85,77% |
| Macro Avg Precision | 0,86 |
| Macro Avg Recall | 0,86 |
| Macro Avg F1-Score | 0,86 |

Laporan klasifikasi per kelas (masing-masing 1.000 gambar uji):

| Kelas | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| airplane | 0,90 | 0,85 | 0,88 |
| automobile | 0,92 | 0,95 | 0,93 |
| bird | 0,89 | 0,76 | 0,82 |
| cat | 0,82 | 0,67 | 0,73 |
| deer | 0,84 | 0,85 | 0,84 |
| dog | 0,83 | 0,75 | 0,79 |
| frog | 0,73 | 0,97 | 0,83 |
| horse | 0,88 | 0,92 | 0,90 |
| ship | 0,94 | 0,93 | 0,93 |
| truck | 0,87 | 0,94 | 0,91 |

### Interpretasi Hasil

- Model mencapai akurasi **85,77%** pada data test, jauh di atas tebakan acak untuk 10 kelas (10%).
- Kelas dengan performa terbaik adalah **automobile** dan **ship** (F1-Score 0,93), diikuti **truck** (0,91) dan **horse** (0,90).
- Kelas dengan performa terendah adalah **cat** (F1-Score 0,73, recall 0,67), diikuti **dog** (F1-Score 0,79) dan **bird** (F1-Score 0,82). Kucing dan anjing sama-sama hewan berkaki empat dengan bentuk yang mirip pada resolusi 32×32, sehingga sulit dibedakan.
- Kelas **frog** memiliki recall sangat tinggi (0,97) tetapi precision paling rendah (0,73). Artinya model cenderung terlalu sering memprediksi gambar sebagai *frog*, termasuk pada gambar dari kelas lain.

## Kesimpulan

Proyek ini menunjukkan penerapan *deep learning* untuk klasifikasi gambar menggunakan CNN pada dataset CIFAR-10, mulai dari *preprocessing*, augmentasi data, perancangan arsitektur, pelatihan dengan *callback*, hingga evaluasi per kelas.

Model CNN dengan tiga blok konvolusi, *Batch Normalization*, dan *Dropout* mencapai akurasi **85,77%** dan loss 0,4315 pada data test. Performa terbaik terdapat pada kelas kendaraan seperti *automobile* dan *ship*, sedangkan kelas hewan seperti *cat* dan *dog* masih menjadi tantangan utama.

Secara keseluruhan, proyek ini memberikan pengalaman dalam membangun dan mengevaluasi model *deep learning* secara menyeluruh, termasuk teknik regularisasi untuk mengurangi *overfitting*.
