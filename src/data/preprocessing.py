""" Preprocessing module for the MovieLens dataset """

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(ratings_path: str, movies_path: str, sample_frac):
    """
    Load and preprocess MovieLens dataset.
    
    Args:
        ratings_path: Path to ratings.csv
        movies_path: Path to movies.csv
        sample_frac: Fraction of data to sample (default: 0.1)
    
    Returns:
        dict containing processed dataframes and encoders
    """
    # Load data
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Sample some fraction of data to speed up computations
    ratings = ratings.sample(frac=sample_frac, random_state=42)  
    movies = movies.sample(frac=sample_frac, random_state=42)

    # Check for missing values
    if ratings.isnull().sum().any() or movies.isnull().sum().any():
        raise ValueError("Missing values in dataset")

    # Normalise ratings 0-1
    scaler = MinMaxScaler()
    ratings['scaled_rating'] = scaler.fit_transform(ratings[['rating']])

    # Encode user and movie IDs
    # Note encode first then explode to avoid multiple encodings for each user/movie pair
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    
    ratings['userId'] = user_encoder.fit_transform(ratings['userId'])
    ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

    # Merge ratings with movies
    full_ratings = pd.merge(ratings, movies, on='movieId')
    
    # Process genres
    full_ratings['genres'] = full_ratings['genres'].str.split('|')
    exploded_ratings = full_ratings.explode('genres')

    # Split data into train and test sets
    train_data, test_data = train_test_split(
        exploded_ratings, 
        test_size=0.2, 
        random_state=42
    )

    return {
        'train_data': train_data,
        'test_data': test_data,
        'user_encoder': user_encoder,
        'movie_encoder': movie_encoder,
        'rating_scaler': scaler,
        'num_users': len(user_encoder.classes_),
        'num_movies': len(movie_encoder.classes_)
    }