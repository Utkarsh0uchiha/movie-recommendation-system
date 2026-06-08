import streamlit as st
from PIL import Image
import json
import base64
import io
import requests
from classifier import KNearestNeighbours
import re
from dotenv import load_dotenv
import os

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────
with open("movie_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("movie_titles.json", "r", encoding="utf-8") as f:
    movie_titles = json.load(f)

OMDB_API_KEY = os.getenv("OMDB_API_KEY")
OMDB_URL = "https://www.omdbapi.com/"

# ── CSS (decorative only — no dynamic content injected here) ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e3da;
}
[data-testid="stSidebar"] { background: #111111 !important; border-right: 1px solid #1e1e1e; }
[data-testid="stSidebar"] label { color: #888 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* rank badge */
.rank-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.6rem;
    color: #e8c97a;
    line-height: 1;
    padding-top: 4px;
}
/* title link */
.movie-title-link a {
    font-size: 1.1rem;
    font-weight: 500;
    color: #f0ebe0 !important;
    text-decoration: none !important;
}
.movie-title-link a:hover { color: #e8c97a !important; }

/* rating pill */
.rating-pill {
    display: inline-block;
    background: #1e1a0e;
    border: 1px solid #3a3010;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.8rem;
    color: #e8c97a;
    letter-spacing: 0.04em;
    margin-top: 4px;
}

/* card separator */
.card-sep { border: none; border-top: 1px solid #1e1e1e; margin: 0.2rem 0 0.8rem 0; }

/* app title */
.app-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 0.15em;
    color: #e8c97a;
    line-height: 1;
}
.app-sub {
    font-size: 0.75rem;
    color: #444;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* button */
.stButton > button {
    background: #e8c97a !important;
    color: #0d0d0d !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    width: 100%;
    padding: 0.6rem 1rem !important;
}
.stButton > button:hover { opacity: 0.8 !important; }

/* poster placeholder */
.no-poster {
    width: 90px; height: 135px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_movie_info(movie_title):
    try:
        response = requests.get(
            OMDB_URL,
            params={
                "apikey": OMDB_API_KEY,
                "t": movie_title
            },
            timeout=8
        )

        response.raise_for_status()

        movie = response.json()

        if movie.get("Response") == "False":
            return 0.0, None

        try:
            rating = float(movie.get("imdbRating", "0"))
        except:
            rating = 0.0

        poster_url = movie.get("Poster")

        if poster_url == "N/A":
            poster_url = None

        return rating, poster_url

    except Exception:
        return 0.0, None


def fetch_poster_pil(url: str):
    """Download poster → PIL Image, resized."""
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return img.resize((180, 270))
    except Exception:
        return None


def recommend_movies(test_point, k: int):
    target = [0] * len(movie_titles)

    model = KNearestNeighbours(
        data,
        target,
        test_point,
        k=k
    )

    model.fit()

    results = []

    for i in model.indices:
        title = movie_titles[i][0]
        link = movie_titles[i][2]

        rating, poster_url = get_movie_info(title)

        results.append(
            (
                title,
                link,
                rating,
                poster_url
            )
        )

    results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return results

# ── Card renderer (uses native Streamlit — no HTML injection for content) ─────


def render_card(rank: int, title: str, link: str, rating: float,
                poster_url: str | None, show_posters: bool):
    stars = "★" * round(rating / 2) + "☆" * (5 - round(rating / 2))

    if show_posters:
        # 3 columns: rank | poster | info
        c_rank, c_poster, c_info = st.columns([0.8, 1.2, 6])
    else:
        c_rank, c_info = st.columns([0.8, 7])

    with c_rank:
        st.markdown(
            f"<div class='rank-badge'>{rank:02d}</div>", unsafe_allow_html=True)

    if show_posters:
        with c_poster:
            img = fetch_poster_pil(poster_url) if poster_url else None
            if img:
                st.image(img, width=90)
            else:
                st.markdown("<div class='no-poster'>🎬</div>",
                            unsafe_allow_html=True)

    with c_info:
        # Use native markdown for the clickable title — safe, no injection
        st.markdown(f"### [{title}]({link})")
        st.markdown(
            f"<span class='rating-pill'>⭐ {rating:.1f} &nbsp;·&nbsp; {stars}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='card-sep'/>", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    genres_list = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Game-Show",
        "History", "Horror", "Music", "Musical", "Mystery", "News", "Reality-TV",
        "Romance", "Sci-Fi", "Short", "Sport", "Thriller", "War", "Western",
    ]
    movies = [t[0] for t in movie_titles]
    category = ["-- Select --", "Movie Based", "Genre Based"]

    # Sidebar brand
    st.sidebar.markdown(
        "<div style='font-family:Bebas Neue,sans-serif;font-size:1.6rem;"
        "color:#e8c97a;letter-spacing:0.12em'>🎬 CineMatch</div>"
        "<div style='font-size:0.68rem;color:#444;text-transform:uppercase;"
        "letter-spacing:0.16em;margin-bottom:1.2rem'>IMDB 5000 dataset</div>",
        unsafe_allow_html=True,
    )

    cat_op = st.sidebar.selectbox("Recommendation Type", category)

    # Main header
    st.markdown("<div class='app-title'>CineMatch</div>"
                "<div class='app-sub'>Intelligent movie recommendations · IMDB 5000</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Movie Based
    if cat_op == category[1]:
        select_movie = st.sidebar.selectbox(
            "Select a seed movie", ["-- Select --"] + movies)
        if select_movie == "-- Select --":
            st.sidebar.warning("Pick a movie to continue.")
            st.info("👈 Choose a seed movie from the sidebar.")
        else:
            show_posters = st.sidebar.radio(
                "Show posters?", ("Yes", "No")) == "Yes"
            n = st.sidebar.slider("Number of recommendations", 5, 20, 10)
            if st.sidebar.button("Get Recommendations"):
                with st.spinner("Fetching recommendations & posters…"):
                    test_point = data[movies.index(select_movie)]
                    table = recommend_movies(test_point, n + 1)
                    table.pop(0)

                st.success(
                    f"Top {len(table)} picks based on **{select_movie}**")
                for rank, (title, link, rating, poster_url) in enumerate(table, 1):
                    render_card(rank, title, link, rating,
                                poster_url, show_posters)

    # ── Genre Based
    elif cat_op == category[2]:
        sel_gen = st.sidebar.multiselect("Select genres", genres_list)
        if sel_gen:
            show_posters = st.sidebar.radio(
                "Show posters?", ("Yes", "No")) == "Yes"
            imdb_score = st.sidebar.slider("Minimum IMDb score", 1, 10, 7)
            n = st.sidebar.number_input(
                "Number of recommendations", 5, 20, 10, step=1)
            if st.sidebar.button("Get Recommendations"):
                with st.spinner("Fetching recommendations & posters…"):
                    test_point = [
                        1 if g in sel_gen else 0 for g in genres_list]
                    test_point.append(imdb_score)
                    table = recommend_movies(test_point, int(n))

                st.success(f"Top picks for **{', '.join(sel_gen)}**")
                for rank, (title, link, rating, poster_url) in enumerate(table, 1):
                    render_card(rank, title, link, rating,
                                poster_url, show_posters)
        else:
            st.sidebar.warning("Select at least one genre.")
            st.info("👈 Select genres from the sidebar.")

    else:
        st.info("👈 Choose a recommendation type from the sidebar to get started.")


if __name__ == "__main__":
    run()
