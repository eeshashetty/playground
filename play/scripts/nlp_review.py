
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

from nltk.corpus import stopwords, movie_reviews
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()

def create_features(words):
    new_words = [lm.lemmatize(word) for word in words]
    new_words = [word for word in new_words if word not in stopwords.words('english')]
    words_dict = dict([(word, True) for word in new_words])
    return words_dict

pos_revs = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_revs.append((create_features(words), 'positive'))

neg_revs = []
for fileid in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileid)
    neg_revs.append((create_features(words), 'negative'))

train = pos_revs[750:] + neg_revs[:750]
test = pos_revs[:750] + neg_revs[750:]

classifier = NaiveBayesClassifier.train(train)

accuracy = nltk.classify.util.accuracy(classifier, test)
print(accuracy*100)

from sklearn.externals import joblib
joblib.dump(classifier, 'nlpmodel.pkl')

'''
# testing the model
input_text = input('Review: ')
text = create_features(text)
print('\nsentiment: ', end='')
print(classifier.classify(text))

'''