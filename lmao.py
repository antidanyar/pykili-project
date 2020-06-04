import sys, zipfile, csv, json
import gensim, wget
from nltk.corpus import stopwords
import text_preprocess as tp

settings = 'settings.json'
freqdict_file = 'freqdict.csv'
topipm = 75.0
stopwords_rus = stopwords.words('russian')
with open(settings, 'r', encoding= 'utf-8') as setting:
    topics = json.load(setting)

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

def semantic_similarity(topic_words, text_words, w2v_model, error_coef=0.0):
    sim_sum = 0.0
    for topic_word in topic_words:
        for text_word in text_words:
            try:
                sim_sum += w2v_model.similarity(topic_word, text_word)
            except Exception:
                sim_sum += error_coef
    return sim_sum / (len(topic_words) * len(text_words))

def sem_sim_3max(topic_words, text_words, w2v_model, error_coef = 0.0, num = 5):
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

#model_url = 'http://vectors.nlpl.eu/repository/11/182.zip' - этот кусок кода использовать если зип с моделью не скачан
#m = wget.download(model_url)
#model_file = model_url.split('/')[-1]
model_file = '182.zip'
with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)

text = ''
with open('text.txt', 'r', encoding= 'utf-8') as f:
    for line in f.readlines():
        text += line
text = text.lower()
old_text_words = tp.text_preprocessing(text)

top_freq = get_top_freq(freqdict_file, topipm)
text_words = []
for word in old_text_words:
    if not(word.split('_')[0] in stopwords_rus or word.split('_')[0] in top_freq):
        text_words.append(word)
with open('tagged_text.txt', 'w', encoding= 'utf-8') as f:
    for word in text_words:
        f.write(word + '\n')

topic_similarity = {}
print('algo_1')
for topic in topics: 
    topic_similarity[topic] = semantic_similarity(topics[topic], text_words, model)
for item in topic_similarity.items():
    print(item)

print('algo_3')
topic_similarity = {}
for topic in topics: 
    topic_similarity[topic] = sem_sim_3max(topics[topic], text_words, model)
for item in topic_similarity.items():
    print(item)
