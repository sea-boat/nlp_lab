import os
import jieba
import re
import itertools
from collections import Counter
import numpy as np

base_path = 'D:/nlp_lab/'
classes = ['finance', 'it', 'sports']
padding_word = "<PAD/>"


def load_data():
    sentences, labels = laod_data_label()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def pad_sentences(sentences):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def laod_data_label():
    post_list = []
    label_list = []
    labels = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    i = 0
    for c in classes:
        file_list = os.listdir(base_path + '/data/cnn_train/' + c)
        for files in file_list:
            f = open(base_path + '/data/cnn_train/' + c + '/' + files, 'r', encoding='gbk', errors='ignore')
            temp = f.read().replace('nbsp', '')
            data = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、：“”~@#￥%……&*（）]+", "", temp)
            # data = ''.join(re.findall(u'[\u4e00-\u9fff]+', text))
            _data = list(jieba.cut(data))
            post_list.append(_data)
            label_list.append(labels[i])
            f.close()
        i += 1
    return post_list, label_list


def get_batch(data, batch_size, num_epochs):
    data = list(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data_shuffled = np.array(data)[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data_shuffled[start_index:end_index]
