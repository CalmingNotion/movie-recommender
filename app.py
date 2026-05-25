from flask import Flask, render_template, request
import pickle
import requests

app = Flask(__name__)

# load our saved ML model
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

def fetch_poster(title):
    api_key = '8b18680e'  # replace with your OMDb key
    url = f'http://www.omdbapi.com/?t={title}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    return data.get('Poster', '')

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    recommended_posters = []
    
    for i in movies_list:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        recommended_posters.append(fetch_poster(title))
    
    return recommended_movies, recommended_posters

@app.route('/', methods=['GET', 'POST'])
def home():
    movies_list = movies['title'].tolist()
    recommended_movies = []
    recommended_posters = []
    selected_movie = ''
    
    if request.method == 'POST':
        selected_movie = request.form.get('movie')
        recommended_movies, recommended_posters = recommend(selected_movie)
    
    return render_template('index.html', 
                         movies=movies_list,
                         recommended_movies=recommended_movies,
                         recommended_posters=recommended_posters,
                         selected_movie=selected_movie)

if __name__ == '__main__':
    app.run(debug=True)