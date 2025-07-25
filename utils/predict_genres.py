from transformers import pipeline

classifier = pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

unique_genres=['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

def predict_genres(query):
    result=classifier(query,unique_genres,multi_label=True)
    predicted = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.4]
    return predicted