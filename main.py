from fastapi import FastAPI, HTTPException, Query, APIRouter
from fastapi.openapi.models import OpenAPI, Info
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = FastAPI()    # http://127.0.0.1:8000

df = pd.read_csv('final_data.csv')

@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):
    # Filter the dataset by the specified genre
    genre_data = df[df['genres'] == genre]

    # Check if the filtered DataFrame is empty
    if genre_data.empty:
        return {"genre": genre, "most_played_year": None}

    # Use .loc to set the 'release_year' column
    genre_data.loc[:, 'release_year'] = pd.to_datetime(genre_data['release_date']).dt.year

    # Group the filtered data by release year and calculate total playtime
    year_playtime = genre_data.groupby('release_year')['playtime_forever'].sum()

    # Find the year with the highest total playtime
    most_played_year = year_playtime.idxmax()

    return {"genre": genre, "most_played_year": int(most_played_year)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/UsersForGenre/{genre}")
def UserForGenre(genre: str):
    # Filter the dataset by the specified genre
    genre_data = df[df['genres'] == genre]

    # Convert the 'posted' column to datetime when filtering the data
    genre_data['posted'] = pd.to_datetime(genre_data['posted'])

    # Group the filtered data by user, item_id (or item_name), and release year, and calculate total playtime
    user_item_year_playtime = genre_data.groupby(['user_id', 'item_id', genre_data['posted'].dt.year])['playtime_forever'].sum().reset_index()

    # Find the user with the highest total playtime for that genre
    most_played_user = user_item_year_playtime.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Filter data for the most played user
    most_played_user_data = user_item_year_playtime[user_item_year_playtime['user_id'] == most_played_user]

    # Create a list of accumulated playtime by year for the most played user
    year_sum_list = most_played_user_data.rename(columns={'posted': 'Año', 'playtime_forever': 'Horas'}).to_dict(orient='records')

    return {"Usuario con más horas jugadas para Género " + genre: most_played_user, "Horas jugadas": year_sum_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

@app.get("/UsersRecommend/{year}")
def UsersRecommend(year: int):
    # Filter the DataFrame for the given year and recommended reviews with positive/neutral sentiment
    filtered_df = df[(df['recommend'] == True) & (df['sentiment_analysis'] >= 1) & (pd.to_datetime(df['posted']).dt.year == year)]

    if filtered_df.empty:
        return {"error": "No recommended games found for the given year"}

    # Group by game and calculate the number of recommendations
    game_recommendations = filtered_df.groupby('item_name')['recommend'].sum()

    if game_recommendations.empty:
        return {"error": "No recommended games found for the given year"}

    # Sort the games by the number of recommendations in descending order
    sorted_games = game_recommendations.sort_values(ascending=False)

    # Take the top 3 games
    top_3_games = sorted_games.head(3)

    # Create the result in the specified format
    result = [{"1st Place": top_3_games.index[0]}, {"2nd Place": top_3_games.index[1]}, {"3rd Place": top_3_games.index[2]}]

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/UsersNotRecommend/{year}")
def UsersNotRecommend(year: int):
    # Filter the DataFrame for the given year and not recommended reviews with negative sentiment
    filtered_df = df[(df['recommend'] == False) & (df['sentiment_analysis'] == 0) & (pd.to_datetime(df['posted']).dt.year == year)]

    if filtered_df.empty:
        return {"error": "No least recommended games found for the given year"}

    # Group by game and calculate the number of not recommended reviews
    game_not_recommendations = filtered_df.groupby('item_name')['recommend'].count()

    if game_not_recommendations.empty:
        return {"error": "No least recommended games found for the given year"}

    # Sort the games by the number of not recommended reviews in ascending order
    sorted_games = game_not_recommendations.sort_values(ascending=True)

    # Take the top 3 games
    top_3_games = sorted_games.head(3)

    # Create the result in the specified format
    result = [{"1st Place": top_3_games.index[0]}, {"2nd Place": top_3_games.index[1]}, {"3rd Place": top_3_games.index[2]}]

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/sentiment_analysis/{year}")
def sentiment_analysis(year: int):

    # Filter the DataFrame for the given year
    filtered_df = df[pd.to_datetime(df['release_date']).dt.year == year]

    if filtered_df.empty:
        return {"error": "No data found for the given year"}

    # Group by 'sentiment_analysis' and calculate the sum of each sentiment
    grouped_sentiments = filtered_df.groupby(['sentiment_analysis'])['user_id'].nunique().reset_index()

    # Rename the columns and map sentiment values to labels
    grouped_sentiments.columns = ['Sentiment', 'Count']
    sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    grouped_sentiments['Sentiment'] = grouped_sentiments['Sentiment'].map(sentiment_labels)

    # Convert the DataFrame to a dictionary
    sentiment_counts = grouped_sentiments.set_index('Sentiment')['Count'].to_dict()

    return sentiment_counts

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Function for user-item recommendation
def user_recommendation(user_id):
    user_history = df[df['user_id'] == user_id]
    user_preferences = user_history.groupby('item_name')['playtime_forever'].mean().reset_index()
    user_preferences = user_preferences.sort_values(by='playtime_forever', ascending=False)
    recommended_games = user_preferences['item_name'].head(5).tolist()
    return recommended_games

# User-item recommendation endpoint
@app.get("/UserRecommendation/{user_id}")
def user_item_recommendation(user_id: str):
    recommended_games = user_recommendation(user_id)
    return {"Recommended Games": recommended_games}

# Encode the 'item_name' column into numerical values
label_encoder = LabelEncoder()
df['item_name_encoded'] = label_encoder.fit_transform(df['item_name'])

# Standardize the 'playtime_forever' column (optional but recommended)
scaler = StandardScaler()
df['playtime_forever_st'] = scaler.fit_transform(df['playtime_forever'].values.reshape(-1, 1))

# Create a DataFrame with the features for K-nearest neighbors
item_features = df[['item_name_encoded', 'playtime_forever_st']]

# Initialize the K-nearest neighbors model
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')

# Fit the model to your item features
knn_model.fit(item_features)

# Item-item recommendation
def game_recommendation_knn(item_id):
    # Check if the item_id exists in the dataset
    if item_id not in df['item_id'].values:
        return {"Error": f"Item ID {item_id} not found in the dataset"}

    # Find the index of the provided item_id in the dataset
    item_index = df[df['item_id'] == item_id].index[0]

    # Find the K-nearest neighbors
    distances, indices = knn_model.kneighbors([item_features.iloc[item_index]], n_neighbors=6)

    # Extract recommended games using inverse_transform
    recommended_games = list(set(df.iloc[indices[0][1:5]]['item_name']))

    return {"Recommended Games": recommended_games}

@app.get("/game-recommendation/{item_id}")
def get_game_recommendation_knn(item_id: str):
    recommended_games = game_recommendation_knn(item_id)
    return {"Recommended Games": recommended_games}