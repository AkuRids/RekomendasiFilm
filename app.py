import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances

# Membaca dan memproses data
file_path = "movie.csv"
movies_df = pd.read_csv(file_path, encoding="utf-16")

# Preprocessing: Mengganti nilai kosong di kolom 'genre' dan 'introduction' dengan string kosong
movies_df['genre'] = movies_df['genre'].fillna("")
movies_df['introduction'] = movies_df['introduction'].fillna("")

# Membuat kolom 'combined_features' untuk menyimpan gabungan genre dan deskripsi
movies_df['combined_features'] = movies_df['genre'] + " " + movies_df['introduction']

# Mengubah teks menjadi representasi numerik menggunakan CountVectorizer
vectorizer = CountVectorizer(stop_words='english', binary=True)
features_matrix = vectorizer.fit_transform(movies_df['combined_features'])

# Menghitung Jaccard Similarity (1 - Jaccard Distance)
similarity_matrix = 1 - pairwise_distances(features_matrix.toarray().astype(bool), metric='jaccard')

# Fungsi untuk mendapatkan rekomendasi berdasarkan judul film
def get_recommendations(title):
    try:
        # Mendapatkan indeks film berdasarkan judul
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index[0]

        # Mengambil skor kesamaan untuk film tersebut
        similarity_scores = similarity_matrix[idx]

        # Mengurutkan skor kesamaan dan memilih 10 film teratas (kecuali film input)
        similar_indices = similarity_scores.argsort()[::-1][1:11]
        recommendations = movies_df.iloc[similar_indices]
        return recommendations
    except IndexError:
        return pd.DataFrame()  # Jika judul tidak ditemukan atau terjadi kesalahan

# Membuat aplikasi Flask
app = Flask(__name__)

@app.route('/')
def index():
    # Mengambil daftar judul film unik
    movie_titles = movies_df['title'].tolist()
    return render_template('index.html', movies=movie_titles)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    movie_title = request.form['movie']
    recommendations = get_recommendations(movie_title)
    return render_template(
        'recommendations.html',
        movie_title=movie_title,
        recommendations=recommendations.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
