import os
import pandas as pd
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from utils.predict_genres import predict_genres
from utils.rerank_and_explain import rerank_and_explain_with_llm
import uuid

# Set the path for persistent Chroma DB
CHROMA_DIR = "./chroma_db_movies"

# Read the main cleaned dataset
movies = pd.read_csv('./notebooks/cleaned_movies_dataset.csv')

# Genre columns for metadata
genre_cols = [col for col in movies.columns if movies[col].isin([0,1]).all()]

# Prepare the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load or create Chroma DB
if os.path.exists(CHROMA_DIR):
    print("🔁 Loading existing Chroma DB...")
    chroma_db_movies = Chroma(persist_directory=CHROMA_DIR,embedding_function=embeddings)
else:
    print("🆕 Creating new Chroma DB from documents...")
    docs = []
    for _, row in movies.iterrows():
        title = row['Title']
        date = row['Release_Date']
        overview = row['Overview']
        genres = [genre for genre in genre_cols if row[genre] == 1]

        docs.append(Document(
            page_content=overview,
            metadata={
                "title": title,
                "release_date": date,
                "genre": ", ".join(genres)
            }
        ))

    chroma_db_movies = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("✅ Chroma DB created and saved.")


# 🎯 Recommendation Pipeline
def retrieve_semantic_recommendations(query, top_k=5):
    predicted = predict_genres(query)
    print("🎯 Predicted Genres:", predicted)

    results = chroma_db_movies.similarity_search(query, k=top_k)

    matched_titles = [doc.metadata['title'] for doc in results]
    matched_df = movies[movies['Title'].isin(matched_titles)].copy()

    final_df = rerank_and_explain_with_llm(query, matched_df, predicted)
    return final_df


# Optional test run
if __name__ == "__main__":
    print("🚀 Testing Semantic Movie Recommendation")
    sample = retrieve_semantic_recommendations("A horror movie where the hero loses his mind.")
    print(sample[['Title', 'Overview', 'Explanation']])
