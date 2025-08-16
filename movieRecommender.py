import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

movieMatrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

ratingSimilarity = cosine_similarity(movieMatrix.T)
ratingSimilarity = pd.DataFrame(ratingSimilarity, index=movieMatrix.columns, columns=movieMatrix.columns)
tagData = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movieTags = movies.merge(tagData, on='movieId', how='left').fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tagMatrix = tfidf.fit_transform(movieTags['tag'])

tagSimilarity = cosine_similarity(tagMatrix)
tagSimilarity = pd.DataFrame(tagSimilarity, index=movieTags['movieId'], columns=movieTags['movieId'])

ratingWeight = 0.6
tagWeight = 0.4
similarity = ratingWeight * ratingSimilarity + tagWeight * tagSimilarity

def recommend(movieTitle, top=5):
    try:
        movieId = movies[movies['title'] == movieTitle]['movieId'].values[0]
    except IndexError:
        return ["Movie not found"]
    similarScore = similarity[movieId]
    topMov = similarScore.sort_values(ascending=False).head(top+1)
    topMov = topMov.drop(movieId, errors="ignore")
    recommendedTitle = movies[movies['movieId'].isin(topMov.index)]['title'].values
    return recommendedTitle

st.title("Movie Recommendation")
selectedMovie = st.selectbox("Pick a movie you like:", movies['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selectedMovie)
    st.write("You might also like:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")
