import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path="movies_data.pkl", vectors_path="movie_vectors.npz"):
        self.movies = pd.read_pickle(data_path)
        self.vectors = sparse.load_npz(vectors_path)

    def search(self, keyword, top_n=10):
        matches = self.movies[
            self.movies["title"].str.contains(keyword, case=False, na=False)
        ]

        if matches.empty:
            print(f"No movies found containing: '{keyword}'")
            return pd.DataFrame()

        cols = ["title", "original_title", "release_date", "imdb_rating", "popularity", "vote_count"]
        return (
            matches[cols]
            .sort_values("vote_count", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def recommend(self, movie_title, top_n=5, sort_by="imdb_rating", candidate_pool=50):
        match = self.movies[
            self.movies["title"].str.lower() == movie_title.strip().lower()
        ]

        if match.empty:
            print(f"Movie '{movie_title}' not found in the dataset.")
            return pd.DataFrame()

        idx = match.index[0]

        sim_scores = cosine_similarity(self.vectors[idx], self.vectors).flatten()

        pool_size = max(top_n, candidate_pool)
        top_candidates = np.argsort(sim_scores)[::-1][1 : pool_size + 1]

        cols = ["title", "original_title", "release_date", "imdb_rating", "popularity"]
        rec_df = self.movies.iloc[top_candidates][cols].copy()
        rec_df["similarity"] = sim_scores[top_candidates].round(4)

        sort_cols = ["similarity"] if sort_by == "similarity" else [sort_by, "similarity"]
        rec_df = rec_df.sort_values(by=sort_cols, ascending=False)

        return rec_df.head(top_n).reset_index(drop=True)