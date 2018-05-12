# coding=utf-8
import pickle

import gensim
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

import config

print u"Загрузка сообщений."
dialogs = pickle.load(open(config.processed_messages_file, 'rb'))
print u"Загрузка представлений слов."
word_model = gensim.models.Word2Vec.load("data/generated/words_vector.gensim")

print u"Предобработка обучающих данных."

X = []
Y = []

total_dialogs = len(dialogs)
for i, dialog in enumerate(dialogs):
    X.append(word_model.wv[dialog[0]])
    Y.append(word_model.wv[dialog[1]])
    if i % 100 == 0:
        print i, "/", total_dialogs

X = np.array(X)[0]
Y = np.array(Y)[0]

# plt.imshow(X.reshape(X.shape[0], -1), aspect="auto")
# plt.show()


for x in X:
    plt.imshow(x.reshape(x.shape[0], -1), aspect="auto")
    plt.pause(0.0001)

# plt.imshow(Y.reshape(Y.shape[0], -1), aspect="auto")
plt.show()
