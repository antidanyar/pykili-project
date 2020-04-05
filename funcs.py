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

def closest_vector(w2v, word1, word2, word3, w, words, num):
    word_vector = w2v.get_word_vector(word1) - w2v.get_word_vector(word2) + w2v.get_word_vector(word3)
    word_vector = word_vector.T
    t = 0.0
    closest_words = {}
    for i in range(len(w)):
        t = cosine_similarity(word_vector, w[i])
        closest_words[i] = t
    return [words[i] for i in sorted(closest_words, key=closest_words.get, reverse= True)[:num]]


w1_filename = 'w1_file.txt'
w_filename = 'w_file.txt'
words_filename = 'word_file.txt'

w = []
w1 = []
word_index = {}
words = []

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
        words.append(word)
        word_index[word] = int(index)

w2v = word2vec.word2vec()
w2v.init_with_given(word_index, w, w1)

operation = int(input('Найти подходящее слово: введите 1; Вывести возможное окружение: введите 2'))
if operation == 1:
    is_aight = False
    while not(is_aight):
        word1, word2, word3 = input('Введите X,Y,Z в желаемом X-Y+Z = ?').lower().split()
        if word1 in word_index and word2 in word_index and word3 in word_index:
            is_aight = True
        else:
            print('не все слова есть, попробуйте ещё раз')
    print(closest_vector(w2v, word1, word2, word3, w, words, 5))
elif operation == 2:
    is_aight = False
    while not(is_aight):
        word = input('Введите желаемое слово: ')
        if word in word_index:
            is_aight = True
        else:
            print('такого слова нет, попробуйте ещё раз')
    print(w2v.predict(word,5))
