import flask
import random
from flask import send_from_directory, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__)
random_sample = pd.read_csv("Final_output.csv", sep=";")
random_sample['Book Title'] = random_sample['Book Title'].str.lower()
random_sample['Genres'] = random_sample['Genres'].str.lower()
train_data = pd.read_csv("Embeddings.csv", sep=";")
print(random_sample.columns)
embeddings_dict = {}
for i in range(1000):
    book_title = random_sample["Book Title"].iloc[i]
    embedding = train_data.iloc[i].values  # Assuming 'train_data' contains embeddings in each row

    embeddings_dict[book_title] = embedding


def get_similar_books_in_same_genre(dataframe,input_book_title, book_embeddings, genre_mapping, num_similar_books=5):
    input_genre = genre_mapping.get(input_book_title)
    genres = dataframe["Genres"].unique()
    if input_book_title in book_embeddings:
        input_embedding = book_embeddings[input_book_title]

        similarities = {}
        for book_title, embedding in book_embeddings.items():
            if book_title != input_book_title and genre_mapping.get(book_title) == input_genre:
                similarity = cosine_similarity([input_embedding], [embedding])[0][0]
                similarities[book_title] = similarity

        similar_books = sorted(similarities, key=similarities.get, reverse=True)
        similar_books = [f'"{book}"' for book in similar_books if book != input_book_title][:num_similar_books]
        return  f'Here are the most similar books for you :  {" - ".join(similar_books)} ........ Enter a different genre or book for another recommendation!.'
    else:
        random_books = []
        for genre in genres:
            genre_books = dataframe[dataframe["Genres"] == genre]
            random_book_title = random.choice(genre_books["Book Title"].tolist())
            random_books.append(
                f"'{random_book_title}' from the '{genre}' genre."
            )

        return f"I'm sorry, I don't have any recommendation for this but you can Try {random_books}"


genre_mapping = dict(zip(random_sample["Book Title"], random_sample["Genres"]))

def recommend_books_by_genre(dataframe, input_genre):
    genres = dataframe["Genres"].unique()
    if input_genre in ['crime', 'suspense', 'detective', 'thriller', 'spy fiction',  'hardboiled', 'whodunnit', 'music', 'cabal', 'conspiracy', 'cyberpunk' ]:
        input_genre = 'mystery'
    elif input_genre in ['war', 'novel', 'ya', 'non-fiction', 'comic', 'novella', 'utopian', 'dystopian', 'action']:
        input_genre = 'fiction'
    elif input_genre in ['adventure', 'horror', 'comedy', 'historical', 'parallel','humour']:
        input_genre = 'fantasy'
    elif input_genre in ['satire', 'short story', 'romance', 'autobiography', 'biography', 'gothic', 'goth',"children's literature"]:
        input_genre = "children's literature"
    elif input_genre in ['speculative fiction']:
        input_genre = 'science fiction'
    if input_genre in genres:
        genre_books = dataframe[dataframe["Genres"] == input_genre]
        recommended_books = genre_books.sample(n=5)
        return f'{" - ".join(recommended_books["Book Title"].tolist())} ............... Enter a different genre or book for another recommendation!.'
    else:
        random_books = []
        for genre in genres:
            genre_books = dataframe[dataframe["Genres"] == genre]
            random_book_title = random.choice(genre_books["Book Title"].tolist())
            random_books.append(
                f"'{random_book_title}' from the '{genre}' genre."
            )

        return f"I'm sorry, I don't have any recommendation for this type of books but you can Try {random_books}"


@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    req = request.get_json(force=True)
    action = req["queryResult"]["action"]
    print(req)
    if action == "ActionGenre":
        rec_genre = recommend_books_by_genre(random_sample, req["queryResult"]["parameters"]["genre"].lower())
        return {'fulfillmentText': f"{rec_genre}"}
    # ActionTitle
    if action == "NameAction":
        rec_title = get_similar_books_in_same_genre(random_sample,req["queryResult"]["parameters"]["t"].lower(), embeddings_dict,
                                                    genre_mapping)
        return {'fulfillmentText': f"{rec_title}"}

    return {'fulfillmentText': "Please en"}


if __name__ == "__main__":
    app.secret_key = 'ItIsASecret'
    app.debug = True
    app.run()