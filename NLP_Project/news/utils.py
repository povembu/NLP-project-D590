import spacy
import re
from nltk.tokenize.toktok import ToktokTokenizer
#import en_core_web_sm
import nltk
nltk.download('stopwords')

stop_words = nltk.corpus.stopwords.words('english')
stop_words.remove('no')
stop_words.remove('but')
stop_words.remove('not')

nlp = spacy.load("en_core_web_sm")
tokenizer = ToktokTokenizer()

def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stop_words):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_text(corpus, text_lower_case=True,
                     text_stemming=False, text_lemmatization=True,
                     special_char_removal=True, remove_digits=True,
                     stopword_removal=True,stopwords=stop_words):

    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # lemmatize text
        if text_lemmatization:
          doc = lemmatize_text(doc)

        # stem text
        if text_stemming and not text_lemmatization:
          doc = simple_porter_stemming(doc)

        # remove special characters and\or digits
        if special_char_removal:
          # insert spaces between special characters to isolate them
          special_char_pattern = re.compile(r'([{.(-)!}])')
          doc = special_char_pattern.sub(" \\1 ", doc)
          doc = remove_special_characters(doc, remove_digits=remove_digits)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

         # lowercase the text
        if text_lower_case:
          doc = doc.lower()

        # remove stopwords
        if stopword_removal:
          doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords=stopwords)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus