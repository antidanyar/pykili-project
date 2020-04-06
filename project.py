import numpy as np 
import string 
import word2vec
from nltk.corpus import stopwords  
filename = 'embeddings.txt'

textname = 'text.txt'

def preprocessing(corpus):
    stop_words = set(stopwords.words('english'))     
    training_data = [] 
    sentences = corpus.split(".") 
    for i in range(len(sentences)): 
        sentences[i] = sentences[i].strip() 
        sentence = sentences[i].split() 
        x = [word.strip(string.punctuation) for word in sentence if word not in stop_words] 
        x = [word.lower() for word in x] 
        training_data.append(x) 
    return training_data 
       
   
def prepare_data_for_training(sentences,w2v): 
    dataset = set()
    for sentence in sentences: 
        for word in sentence: 
            dataset.add(word)
    V = len(dataset) 
    data = sorted(list(dataset))
    vocab = {} 
    for i in range(len(data)): 
        vocab[data[i]] = i 
    for sentence in sentences: 
        for i in range(len(sentence)): 
            center_word = [0 for x in range(V)] 
            center_word[vocab[sentence[i]]] = 1
            context = [0 for x in range(V)] 
            for j in range(max(0, i-w2v.window_size), min(len(sentence),i+w2v.window_size)): 
                if i!=j: 
                    context[vocab[sentence[j]]] += 1
            w2v.X_train.append(center_word) 
            w2v.y_train.append(context) 
    w2v.initialize(V,data) 
   
    return w2v.X_train,w2v.y_train

corpus = "" 
with open(textname, 'r', encoding= 'utf-8') as text:
    for line in text.readlines():
        corpus += line
corpus = corpus[1:]
epochs = 500
  
training_data = preprocessing(corpus) 
w2v = word2vec.word2vec() 
  
prepare_data_for_training(training_data,w2v) 
print('started training')
w2v.train(epochs)  
print('finished training')

w2v.save()
