from django.shortcuts import render
from django.http import HttpResponse
from .forms import InputText

from sklearn.externals import joblib
# from .nlp_review import create_features
from nltk.tokenize import word_tokenize


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()

def create_features(words):
    new_words = [lm.lemmatize(word) for word in words]
    new_words = [word for word in new_words if word not in stopwords.words('english')]
    words_dict = dict([(word, True) for word in new_words])
    return words_dict

def predict(text):
    model = joblib.load('play/nlpmodel.pkl')
    text = word_tokenize(text)
    text = create_features(text)
    return model.classify(text)

# Create your views here.
def index(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = InputText(request.POST)
        # check whether it's valid:
        if form.is_valid():
            text = form.cleaned_data['text']
            res = predict(text)
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return render(request, 'play/index.html', {'res':res, 'form':form})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = InputText()

    return render(request, 'play/index.html', {'form': form})