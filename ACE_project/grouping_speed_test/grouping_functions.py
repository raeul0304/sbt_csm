import pandas as pd
import numpy as np
import time
import os

# 현재 파일의 디렉토리 경로를 가져옵니다
current_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 파일 경로 설정
movie_df = pd.read_csv(os.path.join(current_dir, 'data', 'movies.csv'))
rating_df = pd.read_csv(os.path.join(current_dir, 'data', 'ratings.csv'))

#print(movie_df.head(5))
#print(rating_df.head(5))

movie_rating_df = pd.merge(movie_df, rating_df, on='movieId', how='inner')

movie_rating_df.drop(columns=['timestamp'], inplace=True)

genre_grouped_df = movie_rating_df.groupby('genres').agg({'rating': 'mean'}).reset_index()
#print(genre_grouped_df.head(5))

movie_rating_df['genres'] = movie_rating_df['genres'].str.split('|')
print(movie_rating_df['genres'].groupyby(movie_rating_df['title']).head(5))





