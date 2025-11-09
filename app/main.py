import streamlit as st
from src.search import search_boolean, search_vector

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Pencarian Artikel Akademik", 
    layout="centered" 
)

# --- Judul dan Deskripsi Utama ---
st.title("Sistem Pencarian Artikel Akademik")
st.markdown(
    """
    Aplikasi ini mengimplementasikan Sistem Temu Kembali Informasi (STKI) menggunakan:
    * **Boolean Model** (pencocokan kata kunci)
    * **Vector Space Model (TF-IDF)** (perankingan relevansi)
    """
)

# --- Focused Input Form ---
st.divider()

with st.container(border=True):
    st.subheader("⚙️ Masukkan Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Query Pencarian:", 
            placeholder="Contoh: informasi sistem dokumen",
            label_visibility="collapsed" 
        )

    with col2:
        model = st.selectbox(
            "Model:",
            ["Vector Space Model", "Boolean Model"],
            index=0, 
            label_visibility="collapsed" 
        )
    
    search_button = st.button("Cari Dokumen 🔎", type="primary", use_container_width=True)

st.divider()

# --- Proses Pencarian dan Tampilan Hasil ---

if search_button:
    if not query.strip():
        st.error("⚠️ Query wajib diisi. Silakan masukkan kata kunci pencarian.")
    else:
        with st.spinner(f"Mencari menggunakan **{model}**..."):
            
            # --- Tampilan Hasil Boolean Model ---
            if model == "Boolean Model":
                results = search_boolean(query)
                st.header(f"📌 Hasil Boolean Model")
                st.caption(f"Query: **'{query}'**")
                
                if not results:
                    st.warning("❌ Tidak ada dokumen yang cocok 100%.")
                else:
                    st.success(f"✅ Ditemukan **{len(results)}** dokumen yang cocok.")
                    
                    # Penomoran Dipisah
                    for i, doc in enumerate(results):
                        # Membagi kolom untuk Nomor (0.1 bagian) dan Dokumen (0.9 bagian)
                        cols_doc = st.columns([0.1, 0.9])
                        with cols_doc[0]:
                            # Nomor dibuat bold dan ditempel ke kiri
                            st.markdown(f"**{i+1}.**") 
                        with cols_doc[1]:
                            # Ikon dan Nama dokumen dipisah dari nomor
                            st.markdown(f"📄 **{doc}**") 

            # --- Tampilan Hasil Vector Space Model (TF-IDF) ---
            else:
                results = search_vector(query)
                st.header(f"🏆 Hasil Vector Space Model")
                st.caption(f"Query: **'{query}'**")
                
                if not results:
                    st.warning("❌ Tidak ada dokumen yang relevan ditemukan.")
                else:
                    st.success(f"Ditemukan **{len(results)}** dokumen relevan. Menampilkan 10 Teratas:")
                    
                    # Tampilan unik: Menggunakan st.expander dengan detail visual
                    for i, (doc, score) in enumerate(results[:10]):
                        # Penomoran di Judul Expander (Dipisah Jelas)
                        with st.expander(f"**#{i+1}** &nbsp;&nbsp;&nbsp; **{doc}**"): 
                            
                            col_score, col_progress = st.columns([1, 4])
                            
                            with col_score:
                                st.markdown(f"**Skor:** `{score:.4f}`")
                            with col_progress:
                                st.progress(score) 
                                
                            st.markdown(
                                "Dokumen ini diurutkan berdasarkan perhitungan **Cosine Similarity** tertinggi."
                            )

# --- Footer ---
st.divider()
st.caption("Project UTS Sistem Temu Kembali Informasi (STKI)")