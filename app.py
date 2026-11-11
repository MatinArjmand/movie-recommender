import streamlit as st
from recommender import MovieRecommender

# Initial page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Load the recommendation engine once into cache
@st.cache_resource
def load_engine():
    return MovieRecommender()

engine = load_engine()

st.title("🎬 Movie Search & Recommendation System")

# --- Section 1: Movie Search ---
st.subheader("1. Search Reference Movie")
keyword = st.text_input(
    "Enter part of the movie's original title (e.g., Batman):",
    value="Batman",
)

selected_movie = None

if keyword.strip():
    search_results = engine.search(keyword, top_n=10)

    if not search_results.empty:
        st.dataframe(search_results, width="stretch", hide_index=True)

        # Select exact movie title from search results to avoid typos
        selected_movie = st.selectbox(
            "Select a movie to get similar recommendations:",
            options=search_results["original_title"].tolist(),
        )
    else:
        st.warning("No movies found matching that title.")

st.divider()

# --- Section 2: Settings & Recommendations ---
st.subheader("2. Similar Movie Recommendations")

col1, col2 = st.columns([2, 1])

with col1:
    sort_options = {
        "IMDb Rating (then Similarity)": "imdb_rating",
        "Release Date (Newest, then Similarity)": "release_date",
        "Popularity (then Similarity)": "popularity",
        "Content Similarity Only": "similarity",
    }
    chosen_label = st.selectbox("Sort results by:", list(sort_options.keys()))
    sort_by = sort_options[chosen_label]

with col2:
    top_n = st.slider("Number of recommendations:", min_value=3, max_value=20, value=5)

if selected_movie:
    recommendations = engine.recommend(
        movie_title=selected_movie,
        top_n=top_n,
        sort_by=sort_by,
        candidate_pool=50,
    )

    if not recommendations.empty:
        st.dataframe(recommendations, width="stretch", hide_index=True)