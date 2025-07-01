import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# Load data dan model
df = pd.read_csv("netflix_titles.csv")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("tfidf_matrix.pkl")
indices = joblib.load("indices.pkl")

st.title("🎬 Sistem Rekomendasi Film Netflix")
st.markdown("Masukkan judul film atau serial, lalu sistem akan merekomendasikan konten serupa.")

# Input judul
title_input = st.text_input("Masukkan judul film / serial:")

# Fungsi rekomendasi
def recommend(title, cosine_sim=tfidf_matrix):
    if title not in indices:
        return ["Judul tidak ditemukan dalam database."]
    
    idx = indices[title]
    sim_scores = list(enumerate(linear_kernel(cosine_sim[idx:idx+1], cosine_sim).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Proses rekomendasi
if title_input:
    recommendations = recommend(title_input)
    st.subheader("Rekomendasi:")
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
