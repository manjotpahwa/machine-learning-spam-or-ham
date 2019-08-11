# Extremely simple text classifier in Python. This can be used to determine
# whether an email text (or any other text for that matter) is spam or ham.
# Algorithm:
# Cleanse document by converting to lowercase, lemmatising, removing stop
# words, etc.
# Vectorize on the basis of frequency of the words using a bag of words.



import pandas as pd
import nltk
from nltk import stem
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix


stemmer = stem.SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
#stopwords = set(stopwords.words('english'))
stopwords = nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


data = pd.read_csv("spam.csv", encoding = "latin-1")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})


def cleanup_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
#    nltk_pos = [tag[1] for tag in pos_tag(word_tokenize(msg))]
#    msg = [tag[0] for tag in pos_tag(word_tokenize(msg))]
#    wnpos = ['a' if tag[0] == 'J' else tag[0].lower() if tag[0] in
#            ['N', 'R', 'V'] else 'n' for tag in nltk_pos]
#    msg = " ".join([lemmatizer.lemmatize(word, wnpos[i]) for i,
#        word in enumerate(msg)])
#    msg = [word for word in msg.split() if word not in stopwords]


    return msg

data['text'] = data['text'].apply(cleanup_messages)

# vectorize_text
X_train, X_test, y_train, y_test = train_test_split(data['text'],
        data['label'], test_size = 0.1, random_state = 1)
# training the vectorizer 
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)


# build SVM classifier
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)

X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test) 
print(confusion_matrix(y_test, y_pred))


def predict(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]
