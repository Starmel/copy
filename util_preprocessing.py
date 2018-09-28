# coding=utf-8
import os

import numpy as np
import tensorflow as tf
from gensim.utils import simple_preprocess
from tensorflow.contrib.tensorboard.plugins import projector
import codecs

import config


def add_service_tokens(msg_words, max_words, end=False):
    """
    Добавляет служебные токены к сообщению
    ["привет"], max_words = 3   -> ["привет", "pad", "pad"]
    :param msg_words: набор слов, которые нужно выравнить
    :param max_words: по какое количество символов нужно выравнить сообщение
    :param end: добавлять ли символ конца сообщения
    :return: массив слов с символами выравнивания
    """
    msg_words_count = len(msg_words)
    out = msg_words
    if msg_words_count < max_words:
        if not end:
            out += ["pad"] * (max_words - msg_words_count)
        else:
            out += ["eos"]
            out += ["pad"] * (max_words - msg_words_count - 1)
    elif msg_words_count > max_words:
        raise Exception("Message out max words count, message =", msg_words, "except words =", max_words, "found =",
                        msg_words_count)
    return out if end else list(reversed(out))


def word2vector(word2vec_model, x):
    """
    Превращает слово в набор векторов
    :param word2vec_model: модель в которой брать представление
    :param x: слово
    :return: векторное представление слова
    """
    try:
        return word2vec_model.wv[tokenize(x)]
    except:
        return word2vec_model.wv[[u"я"]]


def vector2word(word2vec_model, x):
    """
    Превращает набор векторов в слово
    :param word2vec_model: модель в которой брать представление
    :param x: вектор
    :return: слово
    """
    x = word2vec_model.most_similar(positive=[x])[0]
    return x


def message2vector(word2vec_model, msg, max_words):
    """
    Представляет указанное сообщение как набор векторов
    :param word2vec_model: модель в которой брать представление
    :param msg: строка сообщения
    :param max_words: по какой границе нужно выравнивать количество слов
    :return: массив векторов слов
    """
    return [word2vec_model.wv[word]
            for word in add_service_tokens(tokenize(msg), max_words) if word in word2vec_model.wv.vocab]


def visualize(model, output_path, vector_word_len):
    """
    Представляет слова в 3-х мерном пространстве
    :param model: модель в которой брать представление
    :param output_path: куда сохранить файл визуаализации
    :param vector_word_len: длина векторного слова
    """
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), vector_word_len))

    with open(os.path.join(output_path, meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            if word == '':
                file_metadata.write(u"{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write(u"{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable=False, name='w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(output_path, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(output_path, 'w2x_metadata.ckpt'))
    print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))


def get_test_messages():
    """
    Берет строки из файла, чтобы можно было менять тестовые данные во время обучения модели.
    :return: вовзвращает массив тестоввых строк
    """
    test_messages = codecs.open(config.test_messages_file, 'r', encoding="utf-8").readlines()
    test_messages = map(lambda x: x.rstrip(), test_messages)
    return test_messages


def tokenize(message):
    """
    Разбиввает строку на слова, убирает лишние символы
    :param message: строка
    :return: массив слов
    """
    return simple_preprocess(message, min_len=0)
