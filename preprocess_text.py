
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess_text(texts, labels=None):
    cleaned_texts = [text.lower().replace("\n", " ").strip() for text in texts]
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(cleaned_texts).toarray()
    if labels is not None:
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        return text_features, encoded_labels
    return text_features
