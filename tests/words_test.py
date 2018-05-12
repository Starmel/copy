# coding=utf-8
import gensim

model = gensim.models.Word2Vec.load("data/generated/words_vector.gensim")

print "Testing.."

while True:
    word = raw_input().decode("utf-8")
    if word in model.wv.vocab:
        similar = model.similar_by_word(word)
        # print "для", word, model.wv[word]
        for pair in similar:
            k, v = pair
            print k, v
    print ""
