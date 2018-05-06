import os
import codecs
import json
from six import iteritems
import random

train_folder = 'D:\data/train'
tag2label = {"U": 0, "B": 1, "M": 2, "E": 3, "S": 4}


def read_train_data():
    data = []
    file_list = os.listdir(train_folder)
    for name in file_list:
        f = open(train_folder + '/' + name, 'r', encoding='utf-8', errors='ignore')
        sentences = []
        labels = []
        while 1:
            line = f.readline()
            if not line: break
            if line != '\n':
                [word, label] = line.split()
                sentences.append(word)
                labels.append(label)
            else:
                data.append((sentences, labels))
                sentences, labels = [], []
        f.close()
    return data


def create_vocab(text):
    unique_chars = ['<NUM>', '<UNK>', '<ENG>'] + list(set(text))
    print(unique_chars)
    vocab_size = len(unique_chars)
    vocab_index_dict = {}
    index_vocab_dict = {}
    for i, char in enumerate(unique_chars):
        vocab_index_dict[char] = i
        index_vocab_dict[i] = char
    return vocab_index_dict, index_vocab_dict, vocab_size


def load_vocab(vocab_file):
    with codecs.open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_index_dict = json.load(f)
    index_vocab_dict = {}
    vocab_size = 0
    for char, index in iteritems(vocab_index_dict):
        index_vocab_dict[index] = char
        vocab_size += 1
    return vocab_index_dict, index_vocab_dict, vocab_size


def save_vocab(vocab_index_dict, vocab_file):
    with codecs.open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_index_dict, f, indent=2, sort_keys=True)


def get_data():
    text = []
    file_list = os.listdir(train_folder)
    for name in file_list:
        f = open(train_folder + '/' + name, 'r', encoding='utf-8', errors='ignore')
        while 1:
            line = f.readline()
            if not line:
                break
            if line != '\n':
                w_l = line.split('\t')
                text.append((w_l[0].encode('utf-8').decode('utf-8-sig').strip()))
            else:
                continue
        f.close()
    return text


def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)
        label_ = [tag2label[tag] for tag in tag_]
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []
        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x: len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


if __name__ == '__main__':
    vocab_index_dict, index_vocab_dict, vocab_size = create_vocab(get_data())
    save_vocab(vocab_index_dict, 'vocab.json')
