from django.shortcuts import render
from .models import News
from datetime import datetime as dt
import joblib
#from sklearn.feature_extraction.text import CountVectorizer
from .utils import normalize_text
import nltk
nltk.download('stopwords')

# Create your views here.
def home(request):
    qs = News.objects.all().values().order_by('-datetime')
    text = ""
    err = ''

    if request.method == 'POST':
        text = request.POST['comment']
        
        if text.strip() != '':
            model = joblib.load("lr_cv_model.sav")
            stop_words = nltk.corpus.stopwords.words('english')
            stop_words.remove('no')
            stop_words.remove('but')
            stop_words.remove('not') 
            cv = joblib.load("cv_vector.model")
            norm_text = normalize_text([text])
            cv_text = cv.transform(norm_text)
            scores = model.predict_proba(cv_text)
            print(scores[0,1])
            if scores[0,1] > 0.5:
                err = "This comment is not appropriate on this page"
            else:
                news = News(comment=text, label=1, datetime=dt.now())
                news.save()
                err = ''

    context = {
        'data': qs,
        'error': err
    }
    return render(request,'news/index.html', context)