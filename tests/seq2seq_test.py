# coding=utf-8
import pickle
import sys

import numpy as np

import gensim
import keras

from util_preprocessing import message2vector, word2vector, add_service_tokens
from seq2seq.models import SimpleSeq2Seq, Seq2Seq

print u"Загрузка сообщений."
dialogs = pickle.load(open('data/generated/processed_messages.txt', 'rb'))[:35000]
print u"Загрузка представлений слов."
word_model = gensim.models.Word2Vec.load("data/generated/words_vector.gensim")

print u"Предобработка обучающих данных."

X = []
Y = []

total_dialogs = len(dialogs)
for i, dialog in enumerate(dialogs):
    x_sequence = []
    y_sequence = []

    for word in add_service_tokens(dialog[0], 20):
        x_sequence.append(list(reversed(word2vector(word_model, word)[0])))

    for word in add_service_tokens(dialog[1], 20):
        y_sequence.append(word2vector(word_model, word)[0])

    if i % 100 == 0:
        print i, "/", total_dialogs

    X.append(x_sequence)
    Y.append(y_sequence)

X = np.array(X)
Y = np.array(Y)

num_ex, sequence_length, vec_size = X.shape
print X.shape

word_model = gensim.models.Word2Vec.load("data/generated/words_vector.gensim")

model = Seq2Seq(input_dim=vec_size,
                hidden_dim=vec_size,
                output_length=sequence_length,
                output_dim=vec_size,
                depth=2,
                dropout=0.3)

model.load_weights("data/generated/2018-05-07 02:01:09/iter_51.h5")

print "Testing.."

while True:
    msg = raw_input().decode("utf-8")
    vector_input = message2vector(word_model, sequence_length - 2, msg)
    vector_output = model.predict(np.array([vector_input]))[0]
    words_output = [word_model.similar_by_vector(vector_word)[0][0] for vector_word in vector_output]
    msg_output = " ".join(words_output)

    print msg_output
    print ""
