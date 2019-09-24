# -*- coding: utf-8 -*-
import json
import os, codecs
import pickle
import numpy as np
import jieba
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import sys
import requests
import time


def generate_embedding(level):
    data_path = 'data/%s_level' % level

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # prepare corpus
    sentences = LineSentence(os.path.join(data_path, 'corpus_all.txt'))
    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))

    # run model
    model = Word2Vec(sentences, size=300, min_count=1, window=5, sg=1, iter=10)
    # model.wv.save_word2vec_format('data/word_level/word2vec.txt', binary=False)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab)+2, 300), dtype='float32')

    for w, i in vocab.items():
        if w not in d:
            continue
        # print(d)
        emb[i, :] = weights[d[w], :]

    np.save(open(os.path.join(data_path, 'cail_300_dim.embeddings'), 'wb'), emb)


def generate_embedding(level):
    data_path = 'data/%s_level' % level

    # configure logging
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # prepare corpus
    sentences = LineSentence(os.path.join(data_path, 'corpus_all.txt'))
    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary_all.pkl'), 'rb'))

    # run model
    model = Word2Vec(sentences, size=300, min_count=1, window=5, sg=1, iter=10)
    # model.wv.save_word2vec_format('data/word_level/word2vec.txt', binary=False)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab)+2, 300), dtype='float32')

    for w, i in vocab.items():
        if w not in d:
            continue
        # print(d)
        emb[i, :] = weights[d[w], :]

    np.save(open(os.path.join(data_path, 'cail_300_dim.embeddings'), 'wb'), emb)


def build_word_level_corpus_all(train_file, valid_file=None, test_file=None):
    sentences = list()

    with codecs.open(train_file, "r", encoding="utf8") as f_train:
        for line in f_train:
            x = json.loads(line)
            sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    if valid_file:
        with codecs.open(valid_file, encoding='utf-8') as f_valid:
            for line in f_valid:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    if test_file:
        with codecs.open(test_file, encoding='utf-8') as f_test:
            for line in f_test:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    target_lines = [' '.join(jieba.cut(sentence)) + '\n' for sentence in sentences]

    with codecs.open('data/word_level/corpus_all.txt', 'w', encoding='utf-8') as f_corpus:
        f_corpus.writelines(target_lines)


def build_char_level_corpus_all(train_file, valid_file=None, test_file=None):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        for line in f_train:
            x = json.loads(line)
            sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    if valid_file:
        with codecs.open(valid_file, encoding='utf-8') as f_valid:
            for line in f_valid:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    if test_file:
        with codecs.open(test_file, encoding='utf-8') as f_test:
            for line in f_test:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    target_lines = list()
    for sentence in sentences:
        target_lines.append(' '.join([char for char in sentence]) + '\n')

    with codecs.open('data/char_level/corpus_all.txt', 'w', encoding='utf-8') as f_corpus:
        f_corpus.writelines(target_lines)


def build_word_level_vocabulary_all(train_file, valid_file=None, test_file=None):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        for line in f_train:
            x = json.loads(line)
            sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])
    if valid_file:
        with codecs.open(valid_file, encoding='utf-8') as f_valid:
            for line in f_valid:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])
    if test_file:
        with codecs.open(test_file, encoding='utf-8') as f_test:
            for line in f_test:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])
    corpus = u''.join(sentences)
    word_list = list(set([tk[0] for tk in jieba.tokenize(corpus)]))

    return dict((word, idx+1) for idx, word in enumerate(word_list))


def build_char_level_vocabulary_all(train_file, valid_file=None, test_file=None):
    sentences = list()

    with codecs.open(train_file, encoding='utf-8') as f_train:
        for line in f_train:
            x = json.loads(line)
            sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])
    if valid_file:
        with codecs.open(valid_file, encoding='utf-8') as f_valid:
            for line in f_valid:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])
    if test_file:
        with codecs.open(test_file, encoding='utf-8') as f_test:
            for line in f_test:
                x = json.loads(line)
                sentences.extend([x['A'].strip(), x['B'].strip(), x['C'].strip()])

    corpus = u''.join(sentences)
    char_list = list(set([char for char in corpus]))

    return dict((char, idx+1) for idx, char in enumerate(char_list))


def send(sentences):
    try:
        url = "http://localhost:8080/server/hello"
        headers = {
            'Connection': 'keep-alive',
            'Content-Type': 'application/json; charset=UTF-8'
        }
        payload = {'context': sentences.strip("\n")}
        r = requests.post(url, json=payload, headers=headers, verify=False)
        print(r)
    except Exception as e:
        pass
    finally:
        pass


def load_data(raw_file, level='word', test=False):
    if test:
        if level == 'word':
            with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
                vocabulary = pickle.load(f_vocabulary)
            print('vocab_len_word:', len(vocabulary))
            x_a = list()
            x_b = list()
            with codecs.open(raw_file, encoding='utf-8') as f:
                for line in f:
                    # send(line)
                    # time.sleep(0.01)
                    x = json.loads(line)
                    input_a = x['A']
                    input_b = x['B']
                    input_c = x['C']
                    words_a = jieba.cut(input_a)
                    words_b = jieba.cut(input_b)
                    words_c = jieba.cut(input_c)
                    x_a.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_a])
                    x_a.append(x_a[-1])
                    x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_b])
                    x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_c])
            return x_a, x_b, vocabulary
        else:
            with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
                vocabulary = pickle.load(f_vocabulary)
            print('vocab_len_word:', len(vocabulary))
            with open('data/char_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
                vocabulary_char = pickle.load(f_vocabulary)
            print('vocab_len_char:', len(vocabulary_char))
            x_a = list()
            x_b = list()
            x_a_char = list()
            x_b_char = list()
            with codecs.open(raw_file, encoding='utf-8') as f:
                for line in f:
                    x = json.loads(line)
                    # send(line)
                    # time.sleep(0.01)
                    input_a = x['A']
                    input_b = x['B']
                    input_c = x['C']
                    words_a = jieba.cut(input_a)
                    words_b = jieba.cut(input_b)
                    words_c = jieba.cut(input_c)
                    x_a.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_a])
                    x_a.append(x_a[-1])
                    x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_b])
                    x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_c])
                    x_a_char.append(
                        [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_a])
                    x_a_char.append(x_a_char[-1])
                    x_b_char.append(
                        [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_b])
                    x_b_char.append(
                        [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_c])
            return x_a, x_b, x_a_char, x_b_char, vocabulary
    if level == 'word':
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        x_a = list()
        x_b = list()
        x_c = list()
        with codecs.open(raw_file, encoding='utf-8') as f:
            for line in f:
                x = json.loads(line)
                # send(line)
                # time.sleep(0.01)
                input_a = x['A']
                input_b = x['B']
                input_c = x['C']
                words_a = jieba.cut(input_a)
                words_b = jieba.cut(input_b)
                words_c = jieba.cut(input_c)
                x_a.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_a])
                x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_b])
                x_c.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_c])
        return x_a, x_b, x_c, vocabulary
    else:
        with open('data/word_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary = pickle.load(f_vocabulary)
        print('vocab_len_word:', len(vocabulary))
        with open('data/char_level/vocabulary_all.pkl', 'rb') as f_vocabulary:
            vocabulary_char = pickle.load(f_vocabulary)
        print('vocab_len_char:', len(vocabulary_char))
        x_a = list()
        x_b = list()
        x_c = list()
        x_a_char = list()
        x_b_char = list()
        x_c_char = list()
        # max_len = 0.
        # max_char_len = 0.
        # avg_len = 0.
        # avg_char_len = 0.
        with codecs.open(raw_file, encoding='utf-8') as f:
            for line in f:
                x = json.loads(line)
                # send(line)
                # time.sleep(0.01)
                input_a = x['A']
                input_b = x['B']
                input_c = x['C']
                words_a = jieba.lcut(input_a)
                words_b = jieba.lcut(input_b)
                words_c = jieba.lcut(input_c)
                x_a.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_a])
                x_b.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_b])
                x_c.append([vocabulary.get(word, len(vocabulary) + 1) for word in words_c])
                # if len(words_a) > max_len:
                #     max_len = len(words_a)
                # if len(words_b) > max_len:
                #     max_len = len(words_b)
                # if len(words_c) > max_len:
                #     max_len = len(words_c)
                # avg_len += len(words_a) + len(words_b) + len(words_c)
                # l = 0.
                # for word in words_a + words_b + words_c:
                #     if len(word) > max_char_len:
                #         max_char_len = len(word)
                #     l += len(word)
                # avg_char_len += l / (len(words_a) + len(words_b) + len(words_c))
                x_a_char.append(
                    [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_a])
                x_b_char.append(
                    [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_b])
                x_c_char.append(
                    [[vocabulary_char.get(char, len(vocabulary_char) + 1) for char in word] for word in words_c])
        # print 'max_len:', max_len, 'max_char_len:', max_char_len
        # print 'avg_len:', avg_len / (len(x_a) * 3), 'avg_char_len:', avg_char_len / len(x_a)
        return x_a, x_b, x_c, x_a_char, x_b_char, x_c_char, vocabulary


if __name__ == '__main__':
    vocab = build_word_level_vocabulary_all('data/input.txt')
    with open('data/word_level/vocabulary_all.pkl', 'wb') as vocabulary_pkl:
        pickle.dump(vocab, vocabulary_pkl, -1)
        print(len(vocab))  # 4536
    build_word_level_corpus_all('data/input.txt')
    generate_embedding('word')
    # generate_fasttext_embedding('word')
    vocab = build_char_level_vocabulary_all('data/input.txt')
    with open('data/char_level/vocabulary_all.pkl', 'wb') as vocabulary_pkl:
        pickle.dump(vocab, vocabulary_pkl, -1)
        print(len(vocab))  # 1541
    with open('data/bert_level/vocabulary_all.pkl', 'wb') as vocabulary_pkl:
        pickle.dump(vocab, vocabulary_pkl, -1)
        print(len(vocab))  # 1541

    build_char_level_corpus_all('data/input.txt')
    generate_embedding('char')
    # generate_fasttext_embedding('char')
    load_data('data/input.txt', level='word_level')