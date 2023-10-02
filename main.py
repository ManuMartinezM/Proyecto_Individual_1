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
    # Filter the DataFrame based on the given genre
    genre_df = df[df['genres'].str.contains(genre)]

    if genre_df.empty:
        return {"error": "Genre not found"}

    # Group by release year and calculate total playtime for each year
    year_playtime = genre_df.groupby(pd.to_datetime(genre_df['release_date']).dt.year)['playtime_forever'].sum()

    # Find the year with the maximum playtime
    max_playtime_year = year_playtime.idxmax()

    return {"Release year with most hours played for genre": max_playtime_year}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


@app.get("/UserForGenre/{genre}")
def UserForGenre(genre: str):
    # Filter the DataFrame based on the given genre
    genre_df = df[df['genres'].str.contains(genre)]

    if genre_df.empty:
        return {"error": "Genre not found"}

    # Group by user and release year and calculate total playtime for each year
    user_year_playtime = genre_df.groupby(['user_id', pd.to_datetime(genre_df['posted']).dt.year])['playtime_forever'].sum()

    if user_year_playtime.empty:
        return {"error": "No user data for the given genre"}

    # Find the user with the maximum playtime overall
    max_playtime_user = user_year_playtime.groupby('user_id').sum().idxmax()

    # Get the accumulated hours played for each year for the user with max playtime
    user_max_playtime_data = user_year_playtime.loc[max_playtime_user].reset_index()

    # Filter the data for the specific years (2013, 2012, and 2011)
    specific_years_data = user_max_playtime_data[user_max_playtime_data['posted']]

    # Convert the data to a list of dictionaries
    specific_years_data = [{"Year": int(row['posted']), "Hours": int(row['playtime_forever'])} for index, row in specific_years_data.iterrows()]

    return {
        "User with most hours played overall": max_playtime_user,
        "Hours played for specific years": specific_years_data
        }

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

