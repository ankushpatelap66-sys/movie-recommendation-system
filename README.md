"""
Movie Recommendation System (Content-Based Filtering)
--------------------------------------------------------
Recommends movies similar to one the user likes, based on genre
and description text. Uses TF-IDF (to convert text into numeric
vectors) and Cosine Similarity (to measure how similar two movies
are) - the same core idea used by real recommendation engines
like Netflix and Amazon, just on a smaller scale.

This script uses a small built-in dataset so it runs out of the
box with no internet/downloads needed. For a bigger, more
realistic project, replace the `movies` list below with a full
dataset (e.g. the "TMDB 5000 Movies" dataset from Kaggle) loaded
using pandas.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# 1. Sample dataset: movie title + genre/description tags
# ---------------------------------------------------------
movies = [
    {"title": "The Dark Knight", "tags": "action crime drama superhero batman gotham"},
    {"title": "Inception", "tags": "action sci-fi thriller dreams mind-bending heist"},
    {"title": "Interstellar", "tags": "sci-fi drama space time travel adventure"},
    {"title": "The Avengers", "tags": "action superhero adventure team sci-fi"},
    {"title": "Iron Man", "tags": "action superhero sci-fi adventure technology"},
    {"title": "3 Idiots", "tags": "comedy drama friendship college inspirational"},
    {"title": "Dangal", "tags": "drama sports biography inspirational family"},
    {"title": "Zindagi Na Milegi Dobara", "tags": "comedy drama friendship travel adventure"},
    {"title": "PK", "tags": "comedy drama sci-fi social satire"},
    {"title": "The Conjuring", "tags": "horror thriller supernatural mystery"},
    {"title": "A Quiet Place", "tags": "horror thriller sci-fi survival suspense"},
    {"title": "The Hangover", "tags": "comedy friendship party adventure"},
    {"title": "Titanic", "tags": "romance drama tragedy history"},
    {"title": "The Notebook", "tags": "romance drama love story"},
    {"title": "Sherlock Holmes", "tags": "mystery crime action detective thriller"},
]


def build_similarity_matrix(movie_list):
    """Convert movie tags into TF-IDF vectors and compute similarity between all pairs."""
    tags = [movie["tags"] for movie in movie_list]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tags)
    similarity = cosine_similarity(tfidf_matrix)
    return similarity


def recommend(movie_title, movie_list, similarity_matrix, top_n=3):
    """Find the top N movies most similar to the given movie title."""
    titles = [m["title"] for m in movie_list]

    if movie_title not in titles:
        return None

    index = titles.index(movie_title)
    scores = list(enumerate(similarity_matrix[index]))

    # Sort by similarity score, excluding the movie itself
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != index]

    top_matches = scores[:top_n]
    return [(titles[i], round(score, 2)) for i, score in top_matches]


def main():
    print("===== Movie Recommendation System =====\n")
    print("Available movies:")
    for movie in movies:
        print(f"  - {movie['title']}")

    similarity_matrix = build_similarity_matrix(movies)

    while True:
        print("\nType a movie name from the list above to get recommendations.")
        choice = input("Enter movie title (or 'exit' to quit): ").strip()

        if choice.lower() == "exit":
            print("Goodbye!")
            break

        results = recommend(choice, movies, similarity_matrix)

        if results is None:
            print("Movie not found in the list. Please check spelling and try again.")
            continue

        print(f"\nBecause you liked '{choice}', you might also enjoy:")
        for title, score in results:
            print(f"  - {title}  (similarity: {score})")


if __name__ == "__main__":
    main()
