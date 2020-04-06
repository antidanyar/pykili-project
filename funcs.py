import numpy as np
import word2vec

def cosine_similarity(a, b):
    x = 0.0
    y = 0.0
    z = 0.0
    for i in range(len(a)):
        x += a[i] * b[i]
        y += a[i] ** 2
        z += b[i] ** 2
    y = y ** 0.5
    z = z ** 0.5
    return x / (y * z) - 1.0

w1_filename = 'w1_file.txt'
w_filename = 'w_file.txt'
words_filename = 'word_file.txt'

w = []
w1 = []
word_index = {}

with open(w_filename, 'r', encoding= 'utf-8') as f:
    for line in f.readlines():
        newline = [float(i) for i in line.split()]
        w.append(newline)

with open(w1_filename, 'r', encoding= 'utf-8') as f:
    for line in f.readlines():
        newline = [float(i) for i in line.split()]
        w1.append(newline)

w1 = np.array(w1)
w = np.array(w)

with open(words_filename, 'r', encoding= 'utf-8') as f:
    for line in f.readlines():
        word, index = line.split(' ')
        word_index[word] = int(index)

w2v = word2vec.word2vec()
w2v.init_with_given(word_index, w, w1)

print(w2v.predict('love',5))
