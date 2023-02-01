# conda install -c conda-forge vadersentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
    pass