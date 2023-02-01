# conda install -c conda-forge vadersentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def getSentimentScores():
    reviews = ["this is nice", "i hate this product!", "its ok", "love it!!!", "i like it but its smells bad", "its funny looking", ":("]
    scores = {}

    analyzer = SentimentIntensityAnalyzer()
    for review in reviews:
        vs = analyzer.polarity_scores(review)
        scores[review] = vs['compound']
        # print("{:-<65} {}".format(review, str(vs['compound'])))

    print(scores)

def getCosineSimilarityScores():
    descriptions = ['CSGO is an online first person shooter game.', 
    'Yugioh Master Duel is a card game where you can face enemies online.', 
    'EA Sports FIFA 23 is a football game that you play with your friends locally or online!']

    vectorizer = TfidfVectorizer(stop_words='english')

    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # print(tfidf_matrix.shape)

    cosine_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print(cosine_similarity_matrix)

if __name__ == '__main__':
    getSentimentScores()
    getCosineSimilarityScores()