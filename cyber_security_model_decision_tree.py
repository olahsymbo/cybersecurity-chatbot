import os, sys, inspect, json
import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  
from sklearn.neural_network import MLPClassifier 
from sklearn import tree
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, KeyedVectors
from gensim.models import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from re import sub
from sklearn.metrics import accuracy_score

from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings('ignore')

glove = api.load("glove-wiki-gigaword-50")    

DATA = 'Cyber_security_Database.xlsx'
 
threats = pd.read_excel(DATA, sheet_name = 'threats')


qa = pd.read_excel(DATA, sheet_name = 'questions_answers')
 
data_matrix = pd.read_excel(DATA, sheet_name = 'data_matrix')

labels = data_matrix['label']
Y = pd.get_dummies(labels).values
Ynames = pd.get_dummies(labels).columns

le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
# labels = labels.reshape(-1,1)


data_matrix.drop('label', axis=1, inplace=True)

data_matrixn = data_matrix[data_matrix.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1)

num_words = 100
epoch = 30
batch = 20
max_len = 200

data_matrixn = data_matrixn.values
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data_matrixn)

def tokenizer_text(dat, num_words, tokenizer): 
    
    xdata = tokenizer.texts_to_sequences(dat)
    xdata = pad_sequences(xdata, maxlen=max_len)
    return xdata

X = tokenizer_text(data_matrixn, num_words, tokenizer)

  
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.3, random_state = 42)

n_samples =  X_train.shape[0]


#-----------train Decision Tree ---------------------------------------#

clf = tree.DecisionTreeClassifier(max_depth = 15)
clf.fit(X_train, y_train) 
output = clf.predict(X_test)
print("Test Accuracy of Classifier :: ", accuracy_score(y_test, output))

#-------------------Ask users for their experience----------------------#
text_length = 10
text = str(input("what are you experiencing: "))
if not text.replace(' ','').isalpha():
    print("Error! Only letters (a-z) are allowed!")
    sys.exit()
elif len(text) < text_length:
    print("Error! Only 10 or more characters allowed!")
    sys.exit()


#---- Matching Input Text simialarity with Questions using word Embedding ----#
stopwords = ['the', 'and', 'are', 'a']

def preprocess(doc):
    # Tokenize, clean up input document string
    doc = sub(r'<img[^<>]+(>|$)', " image_token ", doc)
    doc = sub(r'<[^<>]+(>|$)', " ", doc)
    doc = sub(r'\[img_assist[^]]*?\]', " ", doc)
    doc = sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', " url_token ", doc)
    return [token for token in simple_preprocess(doc, min_len=0, max_len=float("inf")) if token not in stopwords]

# Preprocess the documents, including the query string
corpus = [preprocess(document) for document in qa['question']]
query = preprocess(text)
documents = qa['question']
# Load the model: this is a big file, can take a while to download and open
similarity_index = WordEmbeddingSimilarityIndex(glove)

# Build the term dictionary, TF-idf model
dictionary = Dictionary(corpus+[query])
tfidf = TfidfModel(dictionary=dictionary)

# Create the term similarity matrix.  
similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary, tfidf)

query_tf = tfidf[dictionary.doc2bow(query)]

index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(document) for document in corpus]],
            similarity_matrix)

doc_similarity_scores = index[query_tf]

# Output the sorted similarity scores and documents
sorted_indexes = np.argsort(doc_similarity_scores)[::-1] 
 
#----------- Select the question with closest similarity after word embedding----#
th = qa.loc[sorted_indexes[0]]

submatrix = qa[qa['threat_id'] == th['threat_id']]
submatrix = submatrix.sample(n=3, replace=False)

dt = []
resp = []
for i in submatrix['question']: 
    print(i)
    reponse = input()
    while reponse.lower() not in ("yes","no"):
        reponse = input("enter yes/no: ")
    dt.append(i + ',' + reponse + ',')
         

data = " ".join(dt)

data = data[:-1]

Xt = tokenizer_text([data], num_words, tokenizer) 

hn = clf.predict(Xt) 

print("Now testing your input with the model \n") 

prediction = le.inverse_transform([hn])
# prediction = gnb.predict(Xt)

print("This is",prediction[0], '\n')
if prediction == 'not cyberattack':
    print("no recommended actions")
else:
    threats_submatrix = threats.loc[threats['term'] == prediction[0]]
    
    print("Recommended solution: ", threats_submatrix['protection method'])
    