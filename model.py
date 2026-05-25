import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge both datasets into one
movies = movies.merge(credits, on='title')

# Keep only useful columns
movies = movies[['title', 'genres', 'keywords', 'cast', 'crew', 'overview']]


# this function converts that messy JSON list into plain text
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# apply it to genres and keywords columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


# for cast - only get top 3 actors
def convert_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

# for crew - only get the director
def convert_crew(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(convert_crew)


# convert overview from sentence to list of words
movies['overview'] = movies['overview'].fillna('').apply(lambda x: x.split())


# remove spaces from names so they're treated as one word
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])


# combine all columns into one 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# keep only title and tags
movies = movies[['title', 'tags']]

# convert list to a single string
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))

#to convert everything in lower case 
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# convert text into numbers
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# find similarity between all movies
similarity = cosine_similarity(vectors)

def recommend(movie):
    # find the index of the movie in our dataset
    movie_index = movies[movies['title'] == movie].index[0]
    
    # get similarity scores of this movie with all others
    distances = similarity[movie_index]
    
    # sort movies by similarity score and get top 5
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    # print the recommended movies
    for i in movies_list:
        print(movies.iloc[i[0]].title)


# save the model
pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print('model saved!')