# coding=utf-8
import logging
import pickle

import gensim

import config
from util_preprocessing import visualize

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

with open(config.processed_messages_file, 'rb') as f:
    messages = pickle.load(f)

documents = []

for msg in messages:
    documents.append(msg[0] + msg[1])


word_vector_size = 500
model = gensim.models.Word2Vec(size=word_vector_size, window=5, min_count=0, workers=8, sample=0.0005, seed=42)
model.build_vocab(documents)
model.train(documents, total_examples=len(documents), epochs=200)
model.save(config.words_embedding_file)

visualize(model, "data/generated", vector_word_len=word_vector_size)

words_to_test = [u"холод", u"мать", u"привет", u"дела", u"ночь", u"мы", u"я", u"он"]
for word in words_to_test:
    if word in model.wv.vocab:
        similar = model.wv.similar_by_word(word, topn=4)
        print u"для слова:", word
        for pair in similar:
            k, v = pair
            print k, v
        print u""
    else:
        print u"Слово:", word, u" не находится в словаре"

print "Done."
