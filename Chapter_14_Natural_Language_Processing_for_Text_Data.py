
from sklearn.feature_extraction.text import CountVectorizer

# Example of text preprocessing and vectorization
corpus = ['This is a positive review.', 'This is a negative review.']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# Display feature names
print(vectorizer.get_feature_names_out())
