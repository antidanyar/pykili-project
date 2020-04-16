import sys, zipfile, csv, logging
import gensim
from nltk.corpus import stopwords
import text_preprocess as tp

freqdict_file = 'freqdict.csv'
topipm = 50.0
stopwords_rus = stopwords.words('russian')
topics = {
    'лингвистика': ['лингвистика_NOUN', 'язык_NOUN', 'фонетика_NOUN', 'фонология_NOUN', 'морфология_NOUN', 'синтаксис_NOUN', 'семантика_NOUN'],
    'математика': ['математика_NOUN', 'алгебра_NOUN', 'геометрия_NOUN', 'топология_NOUN', 'дискретная_ADJ', 'дифференциальный_ADJ'],
    'физика': ['физика_NOUN','механика_NOUN', 'электромагнетизм_NOUN', 'термодинамика_NOUN', 'квантовый_ADJ', 'относительность_NOUN']
}

def make_log():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

model_file = '182.zip'
with zipfile.ZipFile(model_file, 'r') as archive:
    stream = archive.open('model.bin')
    make_log()
    model = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=True)
    make_log()

text = ''
with open('text.txt', 'r', encoding= 'utf-8') as f:
    for line in f.readlines():
        text += line
text = text.lower()
old_text_words = tp.text_preprocessing(text)

top_freq = get_top_freq(freqdict_file, topipm)
text_words = []
for word in old_text_words: #возможно следует также удалять частотные слова
    if not(word.split('_')[0] in stopwords_rus or word.split('_')[0] in top_freq):
        text_words.append(word)
with open('tagged_text.txt', 'w', encoding= 'utf-8') as f:
    for word in text_words:
        f.write(word + '\n')

topic_similarity = {}
max_similarity = -1000.0
similarity = 0
for topic in topics: #возможно следует считать максимальный среди тематических слов
    topic_similarity[topic] = semantic_similarity(topics[topic], text_words, model)

for item in topic_similarity.items():
    print(item)



