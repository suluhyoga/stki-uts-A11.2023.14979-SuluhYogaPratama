# ============================
#  PREPROCESSING TEKS DOKUMEN
# ============================
# Melakukan: case folding, tokenisasi, stopword removal, stemming

import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')

# Inisialisasi stopwords & stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words("indonesian"))

def clean(text):
    """Case folding + hapus angka, tanda baca, dan spasi ganda"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """Tokenisasi teks"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Hapus stopwords umum bahasa Indonesia"""
    return [t for t in tokens if t not in stop_words]

def stem(tokens):
    """Lakukan stemming (mengubah ke kata dasar)"""
    return [stemmer.stem(t) for t in tokens]

def preprocess_text(text):
    """Pipeline preprocessing lengkap"""
    text = clean(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens

def process_folder(input_folder, output_folder):
    """Proses semua file .txt dari data/raw → data/processed"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                text = f.read()

            tokens = preprocess_text(text)

            with open(os.path.join(output_folder, filename), "w", encoding="utf-8") as f:
                f.write(" ".join(tokens))

            print(f"[OK] {filename} → processed ({len(tokens)} tokens)")