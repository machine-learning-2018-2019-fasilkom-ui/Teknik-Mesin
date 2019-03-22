# Ide Proyek Akhir
- Topik : aplikasi prediksi genre musik
- Deskripsi : memprediksi genre musik country, classical, jazz, metal, pop, blues, rock, reggae, disco, dan hip-hop
- Data : ekstraksi manual fitur - fitur dari data audio pada http://marsyas.info/downloads/datasets.html
- Feature extraction : menggunakan library librosa untuk mengekstrak fitur - fitur audio (cthnya MFCC)
- Feature selection : menggunakan PCA
- Algoritma yang dibandingkan : Random Forest, Decision Tree, k-NN

**Referensi** : http://cs229.stanford.edu/proj2016/poster/BurlinCremeLenain-MusicGenreClassification-poster.pdf

# Progress 1
- Dataset hasil ekstraksi : [data_all.csv](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/data/data_all.csv)
  - Terdapat 50 fitur (mfcc, chroma_stft, rmse dengan ukuran statistik masing - masing mean dan std)
  - Terdiri dari 1000 musik
  - Masing - masing genre musik (10 genre) terdiri dari 100 data musik

- Implementasi Decision Tree C4.5 : [DecisionTree.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/algoritma_decision_tree/DecisionTree.py)
  - Sudah dapat membentuk tree dengan atribut/variabel yang kontinu
  - Kompleksitas split atribut kontinu masih tinggi
  - Belum melakukan pruning
