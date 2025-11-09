**CARA RUN SISTEM & ASUMSI**

Repositori ini berisi implementasi Sistem Temu Kembali Informasi (STKI) menggunakan dua model pencarian, yaitu **Boolean Retrieval Model** dan **Vector Space Model (TF-IDF + Cosine Similarity)**. Sistem dilengkapi antarmuka pencarian menggunakan Streamlit.


**CARA MENJALANKAN SISTEM**

1. Install Dependensi
Pastikan Python 3.9+ telah terpasang. Kemudian jalankan:

pip install -r requirements.txt

2. (Opsional) Menjalankan Preprocessing
Jika ingin memproses ulang dokumen dari data/raw ke data/processed:

python -c "from src.preprocess import process_folder; process_folder('data/raw','data/processed')"

3. Menjalankan Aplikasi Streamlit

streamlit run app/main.py

Aplikasi akan terbuka otomatis di browser.


**ASUMSI SISTEM**

1. Dokumen yang digunakan berformat .txt dan berbahasa Indonesia.

2. Sistem menggunakan tahapan preprocessing berikut:
	- case folding (mengubah huruf menjadi kecil)
	- tokenization (memecah teks menjadi kata)
	- stopword removal menggunakan NLTK
	- stemming menggunakan Sastrawi

3. Boolean Model mendukung operator: AND, OR, NOT

4. VSM melakukan pembobotan menggunakan TF-IDF dan mengukur kedekatan menggunakan cosine similarity.

5. Sistem bersifat bag-of-words, sehingga tidak memahami sinonim atau konteks makna kata.

6. Sistem berjalan secara lokal tanpa koneksi internet, dan dapat dideploy ke Streamlit Cloud.

