# Movie Recommendation System

A content-based movie recommendation engine that suggests similar movies based on genre and description tags — the same core idea used by real recommendation engines like Netflix, on a smaller scale.

## Features
- Converts movie tags into TF-IDF vectors
- Computes similarity between movies using Cosine Similarity
- Returns the top N most similar movies for any given title
- Interactive CLI — type a movie you like and get recommendations instantly

## Tech Stack
- Python
- scikit-learn

## How to Run
```bash
pip install scikit-learn
python movie_recommender.py
```

## What I Learned
- How **content-based filtering** works — recommending items based on their attributes rather than other users' behavior
- How to use TF-IDF to turn text (genre/tags) into a format a machine learning model can compare
- How Cosine Similarity measures how "close" two items are in vector space

## Example
```
Enter movie title: The Dark Knight
Because you liked 'The Dark Knight', you might also enjoy:
  - The Avengers  (similarity: 0.42)
  - Iron Man  (similarity: 0.38)
  - Sherlock Holmes  (similarity: 0.21)
```
