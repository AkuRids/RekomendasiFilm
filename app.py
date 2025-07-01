import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# Load dataset
file_path = "movie.csv"
movies_df = pd.read_csv(file_path, encoding="utf-16")

# Preprocessing
movies_df['genre'] = movies_df['genre'].fillna("")
movies_df['introduction'] = movies_df['introduction'].fillna("")
movies_df['combined_features'] = movies_df['genre'] + " " + movies_df['introduction']

# Vectorization
vectorizer = CountVectorizer(stop_words='english', binary=True)
features_matrix = vectorizer.fit_transform(movies_df['combined_features'])
similarity_matrix = 1 - pairwise_distances(features_matrix.toarray().astype(bool), metric='jaccard')

# Rekomendasi
def get_recommendations(title):
    try:
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]
        similarity_scores = similarity_matrix[idx]
        similar_indices = similarity_scores.argsort()[::-1][1:11]
        recommendations = movies_df.iloc[similar_indices]
        return recommendations
    except IndexError:
        return pd.DataFrame()

# Streamlit UI
st.title("Sistem Rekomendasi Film")

movie_list = movies_df['title'].tolist()
selected_movie = st.selectbox("Pilih judul film:", movie_list)

if st.button("Tampilkan Rekomendasi"):
    results = get_recommendations(selected_movie)
    if not results.empty:
        st.subheader(f"Rekomendasi untuk '{selected_movie}':")
        for i, row in results.iterrows():
            st.markdown(f"**{row['title']}** - *{row['genre']}*")
            st.write(row['introduction'])
            st.markdown("---")
    else:
        st.warning("Judul film tidak ditemukan.")
