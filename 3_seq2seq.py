# coding=utf-8
import datetime
import pickle
import shutil
import sys
import time

import gensim
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from config import *
from seq2seq.models import SimpleSeq2Seq
from util_preprocessing import *

reload(sys)
sys.setdefaultencoding("utf-8")

print u"Предварительная настройка."
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=tf_config))

print u"Загрузка сообщений."
dialogs = pickle.load(open(config.processed_messages_file, 'rb'))
print u"Загрузка представлений слов."
word_model = gensim.models.Word2Vec.load(config.words_embedding_file)

print u"Предобработка обучающих данных."

X = []
Y = []

total_dialogs = len(dialogs)
for i, dialog in enumerate(dialogs):
    X.append(word_model.wv[dialog[0]])
    Y.append(word_model.wv[dialog[1]])
    if i % 100 == 0:
        print i, "/", total_dialogs

X = np.array(X)
Y = np.array(Y)

num_ex, sequence_length, vec_size = X.shape
print X.shape
print u"Построение модели."
# vec_size - длина векторного слова
# sequence_length - длина предложения

model = SimpleSeq2Seq(input_dim=vec_size,
                      hidden_dim=vec_size,
                      output_length=sequence_length,
                      output_dim=vec_size,
                      depth=2)
model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
model.summary()

plot_model(model, to_file='model.png')


print u"Настройка tensorboard."
tensorboard = TensorBoard(log_dir='data/generated/tensorboard', histogram_freq=0, write_graph=True, write_images=False)
tensorboard.set_model(model)

total_iterations = 100000

print u"Настройка логирования."
run_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
save_dir = "data/generated/" + run_timestamp
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
shutil.copy("3_seq2seq.py", save_dir + "/3_seq2seq.py")

print u"Начало обучения."

for iteration in range(total_iterations):
    model.fit(X, Y, epochs=100, batch_size=128 * 6, callbacks=[tensorboard])
    print iteration, '/', total_iterations
    for msg in get_test_messages():
        vector_input = np.array([message2vector(word_model, msg, max_words)])
        vector_output = model.predict(vector_input)[0]
        words_output = [word_model.similar_by_vector(vector_word)[0][0] for vector_word in vector_output]
        msg_output = " ".join(words_output)
        print msg.ljust(40, "."), msg_output

    iteration_file = save_dir + "/iter_" + str(iteration)
    model.save(iteration_file + ".h5")
    model.save_weights(iteration_file + ".w")

print "хоба"
