from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# http://127.0.0.1:8000

@app.get("/")
def read_root():
    return {"Hello": "World"}

df = pd.read_csv('final_data.csv')

@app.get("/PlayTimeGenre/{genre}")
def PlayTimeGenre(genre: str):
    # Filter the dataset by the specified genre
    genre_data = df[df['genres'] == genre]

    # Group the filtered data by release year and calculate total playtime
    year_playtime = genre_data.groupby('release_date')['playtime_forever'].sum()

    # Find the year with the highest total playtime
    most_played_year = year_playtime.idxmax()

    return {"genre": genre, "most_played_year": most_played_year}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/UserForGenre/{genre}")
def UserForGenre(genre: str):
    # Filter the dataset by the specified genre
    genre_data = final_df[final_df['genres'] == genre]

    # Group the filtered data by user and release year, calculating total playtime
    user_year_playtime = genre_data.groupby(['user_id', 'release_date'])['playtime_forever'].sum().reset_index()

    # Find the user with the highest total playtime for that genre
    most_played_user = user_year_playtime.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Get the sum of hours for each individual year as a list of dictionaries
    year_sum_list = user_year_playtime.groupby('release_date')['playtime_forever'].sum().reset_index().to_dict(orient='records')

    return {"genre": genre, "most_played_user": most_played_user, "year_sum_list": year_sum_list}

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

    # Count the reviews by sentiment analysis
    sentiment_counts = filtered_df['sentiment_analysis'].value_counts().to_dict()

    # Map sentiment values to labels
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_counts_with_labels = {sentiment_labels[key]: value for key, value in sentiment_counts.items()}

    return sentiment_counts_with_labels

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

