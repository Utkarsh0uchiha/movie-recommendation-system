# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using Python and Streamlit that recommends movies based on genre similarity and IMDb ratings. The system uses a custom implementation of the K-Nearest Neighbors (KNN) algorithm to identify movies that are most similar to a user's selected movie or preferred genres.

## Features

* Movie-based recommendations
* Genre-based recommendations
* Custom K-Nearest Neighbors implementation
* IMDb rating integration
* Movie poster retrieval
* Interactive Streamlit web interface
* Support for multiple recommendation results

## Tech Stack

* Python
* Streamlit
* NumPy
* BeautifulSoup
* Requests
* IMDbPy
* PIL (Pillow)

## Dataset

This project uses movie metadata derived from the IMDb 5000 Movie Dataset.

The dataset is transformed into numerical feature vectors containing:

* Genre information
* IMDb ratings
* Encoded movie attributes

These vectors are used to calculate similarity between movies.

## Machine Learning Approach

### Content-Based Filtering

The recommendation engine follows a content-based filtering approach.

Each movie is represented as a feature vector containing genre information and rating-related features.

### K-Nearest Neighbors (KNN)

A custom KNN implementation is used to:

1. Calculate Euclidean distance between movies.
2. Find the K most similar movies.
3. Return recommendations based on nearest neighbors.

Distance Formula:

```text
distance = √((x₁ − y₁)² + (x₂ − y₂)² + ... + (xₙ − yₙ)²)
```

## Project Structure

```text
movie-recommendation-system/
│
├── maincode.py              # Streamlit application
├── classifier.py            # Custom KNN implementation
├── movie_data.json          # Movie feature vectors
├── movie_titles.json        # Movie metadata and IMDb links
├── logo.jpg                 # Application logo
├── README.md
```

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run maincode.py
```

The application will launch in your default browser.

## How It Works

### Movie-Based Recommendation

1. Select a movie.
2. The system retrieves its feature vector.
3. KNN identifies similar movies.
4. Recommended movies are displayed along with IMDb ratings.

### Genre-Based Recommendation

1. Select one or more genres.
2. Choose a preferred IMDb score.
3. The system constructs a query feature vector.
4. KNN retrieves the closest matching movies.

## Future Improvements

* TF-IDF based movie description analysis
* Cosine similarity search
* Collaborative filtering
* User accounts and watchlists
* Hybrid recommendation system
* Deep learning based recommendation models

## Learning Outcomes

Through this project, I gained experience with:

* Recommendation systems
* Content-based filtering
* K-Nearest Neighbors (KNN)
* Feature engineering
* Data preprocessing
* Streamlit application development
* API integration
* Web scraping and data retrieval

## Author

**Utkarsh Raj**

* GitHub: https://github.com/Utkarsh0uchiha
* LinkedIn: https://linkedin.com/in/Utkarsh0uchiha
