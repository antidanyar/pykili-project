import sys, zipfile, csv, json
import gensim, wget
from nltk.corpus import stopwords
import text_preprocess as tp

stopwords_rus = stopwords.words('russian')
settings_filename = 'settings.json'
with open(settings_filename, 'r', encoding= 'utf-8') as setting:
    settings = json.load(setting)
topics = settings['topics']
topipm = settings['topipm']
freqdict_file = settings['freqdict']

def get_top_freq(freqdict_file, topipm):
    top_freq = []
    with open(freqdict_file, newline= '', encoding= 'utf-8') as freqfile:
        freqreader = csv.reader(freqfile, delimiter= '\t')
        for row in freqreader:
            word, ipm = row[0], float(row[2])
            if ipm < topipm:
                break
            top_freq.append(word)
    return top_freq

#по умолчанию считает среднее по всем словам в топике. если иметь сравнительно большое количество топик-слов, то можно поиграться с num
def sem_sim_3max(topic_words, text_words, w2v_model, error_coef = 0.0, num = 100): 
    topic_sims = [] 
    for topic_word in topic_words:
        t_sum = 0
        for text_word in text_words:
            try:
                t_sum += w2v_model.similarity(topic_word, text_word)
            except Exception:
                t_sum += error_coef
        t_sum /= len(text_words)
        topic_sims.append(t_sum)
    if num < len(topic_sims):
        topic_sims.sort(reverse= True)
        return sum(topic_sims[:num]) / num
    else:
        return sum(topic_sims) / len(topic_sims)

def get_model():
    #model_url = 'http://vectors.nlpl.eu/repository/11/182.zip' - этот кусок кода использовать если зип с моделью не скачан
    #m = wget.download(model_url)
    #model_file = model_url.split('/')[-1]
    model_file = '182.zip'
    with zipfile.ZipFile(model_file, 'r') as archive:
        stream = archive.open('model.bin')
        return gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

def get_tokens(filename):
    text = ''
    with open(filename, 'r', encoding= 'utf-8') as f:
        for line in f.readlines():
            text += line
    text = text.lower()
    old_text_words = tp.text_preprocessing(text)
    top_freq = get_top_freq(freqdict_file, topipm)
    text_words = []
    for word in old_text_words:
        if not(word.split('_')[0] in stopwords_rus or word.split('_')[0] in top_freq):
            text_words.append(word)
    return text_words

def get_percentage(topic_file, model, topics):
    text_words = get_tokens(topic_file)
    topic_similarity = {}
    for topic in topics: 
        topic_similarity[topic] = sem_sim_3max(topics[topic], text_words, model)
    return topic_similarity.items()

def demonstration():
    print('чем ближе к 1.0, тем ближе текст к теме')
    test_files = ['CHEM_text.txt', 'LING_text.txt', 'CS_text.txt', 'PHYS_text.txt']
    model = get_model()
    for test_file in test_files:
        print(test_file)
        for item in get_percentage(test_file, model, topics):
            print(item)

demonstration()
    


