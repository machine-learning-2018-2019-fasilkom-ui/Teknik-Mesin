# Ide Proyek Akhir
- Topik : aplikasi prediksi genre musik
- ~~Deskripsi : memprediksi genre musik country, classical, jazz, metal, pop, blues, rock, reggae, disco, dan hip-hop~~
- Deskripsi : memprediksi genre musik country, classical, metal, dan hip-hop
- Data : ekstraksi manual fitur - fitur dari data audio pada http://marsyas.info/downloads/datasets.html
- Feature extraction : menggunakan library librosa untuk mengekstrak fitur - fitur audio (cthnya MFCC)
- Feature selection : menggunakan PCA
- Algoritma yang dibandingkan : Random Forest, Decision Tree, k-NN

**Referensi** : http://cs229.stanford.edu/proj2016/poster/BurlinCremeLenain-MusicGenreClassification-poster.pdf

## Dataset

Fitur Musik yang digunakan adalah :
- MFCC (12 fitur)
- Chroma stft (12 fitur)
- Chroma cqt (12 fitur)
- Spectral Bandwidth (1 fitur)
- Spectral Rolloff (1 fitur)
- Spectral Contrast (7 fitur)

Sehingga terdapat 45 fitur audio/musik, masing - masing dihitung statistiknya (mean dan standard deviation) menghasilkan total fitur sebesar **90 fitur**

Prediksi hanya dilakukan pada genre musik country, classical, metal, dan hip-hop ---- [Ekperimen Pengurangan Genre](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/Experiments_Data_Reducing.ipynb) ----

Dataset akhir dihasilkan 400 baris (100 baris masing - masing genre) dan 91 kolom (1 kolom untuk label/target/genre dan 90 kolom untuk fitur yang sudah diekstraksi) ---- [final_genres.csv](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/data/final_genres.csv) ----

## Implementasi Algoritma Klasifikasi
---- [classifier.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn/classifier.py) ---- 
1. Decision Tree dengan algoritma C4.5
   - Menggunakan gain ratio untuk mencari variabel terbaik pada split sebagai pengembangan dari ID3
   - Dapat dilakukan pada variabel kontinu atau numerik
   - Penetuan threshold pada variable kontinu dibandingkan nilai gain dari nilai pada 30%, 50%, 70% dan mean
   - Kemudian threshold dengan gain tertinggi diambil, sehingga pada sebuah variabel kontinu (sebuah kolom) hanya terdapat 2 nilai yaitu <= threshold dan > threshold
   - Melakukan pruning saat tree sudah dibuat
2. K-Nearest Neighbors
   - Menghitung jarak sebuah vektor input ke vektor yang ada pada data train
   - Sebuah Vektor merupakan representasi dari sebuah baris pada csv
   - Diambil vektor - vektor yang paling dekat dengan vektor input sebanyak k buah
   - Banyak k sama dengan banyak genre musik yang ada (k=4)
3. Random Forest
   - Implementasi menggunakan decision tree dengan algoritma c4.5
   - Kompleksitas masih tinggi kemungkinan karena menggunakan algoritma c4.5
   
## Lain - Lain
Library luar yang digunakan adalah numpy, pandas dan sklearn
- numpy digunakan pada semua implementasi klasifikasi
- pandas hanya digunakan untuk membuka csv dan memisahkan kolom variabel/fitur dengan kolom output
- sklearn digunakan untuk memecah data ke data training dan testing, sklearn juga digunakan sebagai pembanding algoritma yang diimplementasikan sendiri 
- Perbandingan algoritma sklearn dan implementasi sendiri terdapat pada [Eksperimen Perbandingan](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/Experiments_Classifier_Testing.ipynb)

Kami mencoba untuk mengimplementasikan dari awal beberapa teknik yang dibutuhkan (walaupun sudah ada dalam sklearn), terletak pada folder [learn](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn), contohnya
- PCA (Principal Component Analysis), untuk mereduksi jumlah fitur ---- [decomposition.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn/decomposition.py) -----
- MinMaxScaler, untuk normalisai min-max terhadap data ---- [preprocessing.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn/preprocessing.py) -----
- metrics, untuk melihat accuracy dan confusion matrix ---- [metrics.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn/metrics.py) ----

PCA dan MinMaxScaler berguna dalam fit pada algoritma k-NN
