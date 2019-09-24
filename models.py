# -*- coding: utf-8 -*-

from abc import abstractmethod

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Conv1D, MaxPooling1D, Flatten, Lambda, LSTM, CuDNNLSTM, Dense, \
    concatenate, TimeDistributed, GlobalMaxPool1D, Reshape, Permute, Dot, Add, Multiply, Conv2D, MaxPooling2D, \
    BatchNormalization, Bidirectional, GlobalAvgPool1D, GlobalMaxPooling1D, Activation, SpatialDropout1D
from keras import backend as K, initializers, regularizers, constraints, activations
from keras.activations import softmax
from keras.models import Model
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from layers.similarity import Highway
import numpy as np
import os
from scipy.stats import rankdata

from layers.cnn import AttMatchingConv1D, AttConv1D
from layers.BiMPM import FullMatching, MaxMatching, AttentiveMatching, MaxAttentiveMatching
from callbacks.ensemble import *


class LanguageModel:
    def __init__(self, config, vocab=None):
        self.config = config
        self.vocab = vocab
        self.a = Input(shape=(self.config.max_len,), dtype='int32', name='A_base')
        self.b = Input(shape=(self.config.max_len,), dtype='int32', name='B_base')
        self.c = Input(shape=(self.config.max_len,), dtype='int32', name='C_base')

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._b = None
        self._qa_model = None

        self.training_model = None
        self.prediction_model = None

        self.callbacks = []

    def get_b(self):
        if self._b is None:
            self._b = Input(shape=(self.config.max_len,), dtype='int32', name='B')
        return self._b

    @abstractmethod
    def build(self):
        return

    def cosine(self):
        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        return lambda x: dot(K.l2_normalize(x[0], axis=-1), K.l2_normalize(x[1], axis=-1))

    def get_qa_model(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model is None:
            simi = self._models

            self._qa_model = Model(inputs=[self.a, self.get_b()], outputs=simi,
                                   name='qa_model')
        return self._qa_model

    def compile(self, **kwargs):
        qa_model = self.get_qa_model()

        good_similarity = qa_model([self.a, self.b])
        bad_similarity = qa_model([self.a, self.c])

        loss = Lambda(lambda x: K.relu(self.config.margin - x[0] + x[1]),
                      output_shape=lambda x: x[0])([good_similarity, bad_similarity])

        self.prediction_model = Model(inputs=[self.a, self.b],
                                      outputs=good_similarity, name='prediction_model')
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=self.config.optimizer, **kwargs)

        self.training_model = Model(inputs=[self.a, self.b, self.c], outputs=loss, name='training_model')
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=self.config.optimizer, **kwargs)

    def pad(self, x):
        return pad_sequences(x, maxlen=self.config.max_len, padding='post', truncating='post')

    def pad_char(self, x):
        for i in range(len(x)):
            x[i] = pad_sequences(x[i], maxlen=self.config.char_per_word, padding='post', truncating='post')
            if len(x[i]) <= self.config.max_len:
                pad_width = ((0, self.config.max_len - len(x[i])), (0, 0))
                x[i] = np.pad(x[i], pad_width=pad_width, mode='constant', constant_values=0)
            else:
                x[i] = x[i][:self.config.max_len]
        return np.asarray(x)

    def fit(self, x_a_train, x_b_train, x_c_train, train_idx, valid_idx,
            x_a_char_train=None, x_b_char_train=None, x_c_char_train=None,
            swa=False, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'

        x_a_train = self.pad(x_a_train)
        x_b_train = self.pad(x_b_train)
        x_c_train = self.pad(x_c_train)
        input_train = [x_a_train[train_idx], x_b_train[train_idx], x_c_train[train_idx]]
        input_valid = [x_a_train[valid_idx], x_b_train[valid_idx], x_c_train[valid_idx]]

        if x_a_char_train:
            x_a_char_train = self.pad_char(x_a_char_train)
            x_b_char_train = self.pad_char(x_b_char_train)
            x_c_char_train = self.pad_char(x_c_char_train)
            input_train += [x_a_char_train[train_idx], x_b_char_train[train_idx], x_c_char_train[train_idx]]
            input_valid += [x_a_char_train[valid_idx], x_b_char_train[valid_idx], x_c_char_train[valid_idx]]

        y_train = np.zeros(shape=(len(train_idx),))  # doesn't get used
        y_valid = np.zeros(shape=(len(valid_idx),))  # doesn't get used

        if swa:
            self.add_swa()

        return self.training_model.fit(input_train, y_train,
                                       validation_data=(input_valid, y_valid),
                                       epochs=1,
                                       verbose=self.config.verbose_training,
                                       batch_size=self.config.batch_size,
                                       callbacks=self.callbacks,
                                       **kwargs)

    def add_swa(self, swa_start=5):
        self.callbacks.append(SWA(self.config.checkpoint_dir, self.config.exp_name, swa_start=swa_start))
        print('Logging Info - Callback Added: SWA')

    def predict(self, a, b, a_char=None, b_char=None):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        a = self.pad(a)
        b = self.pad(b)
        inputs = [a, b]
        if a_char is not None:
            a_char = self.pad_char(a_char)
            b_char = self.pad_char(b_char)
            inputs += [a_char, b_char]
        results = self.prediction_model.predict(inputs, verbose=1)
        results = [r[0] for r in results]
        return results

    def predict_dev(self, a, b, c, valid_idx, a_char=None, b_char=None, c_char=None):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        a = self.pad(a)[valid_idx]
        b = self.pad(b)[valid_idx]
        c = self.pad(c)[valid_idx]
        inputs_b = [a, b]
        inputs_c = [a, c]
        if a_char is not None:
            a_char = self.pad_char(a_char)[valid_idx]
            b_char = self.pad_char(b_char)[valid_idx]
            c_char = self.pad_char(c_char)[valid_idx]
            inputs_b += [a_char, b_char]
            inputs_c += [a_char, c_char]
        results_b = self.prediction_model.predict(inputs_b, verbose=1)
        print(results_b.shape)
        results_b = [r[0] for r in results_b]
        results_c = self.prediction_model.predict(inputs_c, verbose=1)
        print(results_c.shape)
        results_c = [r[0] for r in results_c]
        num = len(results_b)
        correct = 0.
        for i in range(len(results_b)):
            if results_b[i] > results_c[i]:
                correct += 1
        acc = correct / num
        print('val_acc:', acc)
        return acc

    def evaluate(self, results):
        num = 0.
        correct = 0.
        good = 0.
        for i in range(len(results)):
            if i % 2 == 0:
                good = results[i]
            else:
                bad = results[i]
                if good > bad:
                    correct += 1
                num += 1
        acc = correct / num

        print('\n- **Evaluation results of %s model**' % self.config.exp_name)
        print('acc:', acc)
        return acc

    def save_weights(self, cv, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, '%s_%d.hdf5' % (self.config.exp_name, cv))
        print('save %s' % checkpoint_path)
        self.prediction_model.save_weights(checkpoint_path, **kwargs)

    def load_weights(self, cv, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        checkpoint_path = os.path.join(self.config.checkpoint_dir, '%s_%d.hdf5' % (self.config.exp_name, cv))
        self.prediction_model.load_weights(checkpoint_path, **kwargs)

    def load_swa_weight(self):
        print('Logging Info - Loading SWA model checkpoint: %s_swa.hdf5\n' % self.config.exp_name)
        self.prediction_model.load_weights(
            os.path.join(self.config.checkpoint_dir, '%s_swa.hdf5' % self.config.exp_name))
        print('Logging Info - SWA Model loaded')


class SiameseCNN(LanguageModel):

    def char_embedding(self, weights):
        sent_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32')

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='char_embedding_layer', trainable=True)
        sent_char_embedding = TimeDistributed(embedding_layer)(sent_char)
        conv_layer = Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1)
        sent_conv = TimeDistributed(conv_layer)(sent_char_embedding)
        max_pooling = GlobalMaxPooling1D()
        sent_mp = TimeDistributed(max_pooling)(sent_conv)
        return Model(sent_char, sent_mp)

    def build(self):
        a = self.a
        b = self.get_b()

        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        a_embedding = embedding_layer(a)
        b_embedding = embedding_layer(b)

        if self.config.level == 'word_level':
            a_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='a_char_base')
            b_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='b_char_base')
            weights_char = np.load(os.path.join(self.config.embedding_path, 'char_level', self.config.embedding_file))
            char_emb = self.char_embedding(weights_char)
            a_char_embedding = char_emb(a_char)
            a_embedding = concatenate([a_embedding, a_char_embedding])
            b_char_embedding = char_emb(b_char)
            b_embedding = concatenate([b_embedding, b_char_embedding])

        filter_lengths = [2, 3, 4, 5]
        a_conv_layers = []
        b_conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=200, kernel_size=filter_length, padding='same',
                                activation='relu', strides=1)
            a_c = conv_layer(a_embedding)
            b_c = conv_layer(b_embedding)
            a_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(a_c)
            a_flatten = Flatten()(a_maxpooling)
            a_conv_layers.append(a_flatten)
            b_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(b_c)
            b_flatten = Flatten()(b_maxpooling)
            b_conv_layers.append(b_flatten)
        a_conv = concatenate(inputs=a_conv_layers)
        b_conv = concatenate(inputs=b_conv_layers)

        similarity = self.cosine()
        dropout = Dropout(self.config.dropout)

        simi = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(a_conv),
                                                                     dropout(b_conv)])
        return simi


class DPCNN(LanguageModel):

    def char_embedding(self, weights):
        sent_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32')

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='char_embedding_layer', trainable=True)
        sent_char_embedding = TimeDistributed(embedding_layer)(sent_char)
        conv_layer = Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1)
        sent_conv = TimeDistributed(conv_layer)(sent_char_embedding)
        max_pooling = GlobalMaxPooling1D()
        sent_mp = TimeDistributed(max_pooling)(sent_conv)
        return Model(sent_char, sent_mp)

    def build(self):
        a = self.a
        b = self.get_b()

        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        a_embedding = embedding_layer(a)
        b_embedding = embedding_layer(b)

        if self.config.level == 'word_level':
            a_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='a_char_base')
            b_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='b_char_base')
            weights_char = np.load(os.path.join(self.config.embedding_path, 'char_level', self.config.embedding_file))
            char_emb = self.char_embedding(weights_char)
            a_char_embedding = char_emb(a_char)
            a_embedding = concatenate([a_embedding, a_char_embedding])
            b_char_embedding = char_emb(b_char)
            b_embedding = concatenate([b_embedding, b_char_embedding])

        # highway = Highway()
        # sent_embedding = TimeDistributed(highway)(sent_embedding)

        conv_1 = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)
        region_x_a = conv_1(a_embedding)
        region_x_b = conv_1(b_embedding)

        act_1 = Activation(activation='relu')
        region_x_a = act_1(region_x_a)
        region_x_b = act_1(region_x_b)

        # for f in filter_lengths:
        repeat = 3
        # size = self.config.max_len[self.tag]
        conv_2 = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)
        x_a = conv_2(region_x_a)
        x_b = conv_2(region_x_b)

        act_2 = Activation(activation='relu')
        x_a = act_2(x_a)
        x_b = act_2(x_b)

        conv_3 = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)
        x_a = conv_3(x_a)
        x_b = conv_3(x_b)

        add_2 = Add()
        x_a = add_2([x_a, region_x_a])
        x_b = add_2([x_b, region_x_b])

        for _ in range(repeat):
            maxp_1 = MaxPooling1D(pool_size=3, strides=2, padding='same')
            px_a = maxp_1(x_a)
            px_b = maxp_1(x_b)

            act_3 = Activation(activation='relu')
            x_a = act_3(px_a)
            x_b = act_3(px_b)

            conv_4 = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)
            x_a = conv_4(x_a)
            x_b = conv_4(x_b)

            act_4 = Activation(activation='relu')
            x_a = act_4(x_a)
            x_b = act_4(x_b)

            conv_5 = Conv1D(filters=250, kernel_size=3, padding='same', strides=1)
            x_a = conv_5(x_a)
            x_b = conv_5(x_b)

            add_3 = Add()
            x_a = add_3([x_a, px_a])
            x_b = add_3([x_b, px_b])

        size = K.int_shape(x_a)[1]
        maxp_2 = MaxPooling1D(pool_size=size)
        x_a = maxp_2(x_a)
        x_b = maxp_2(x_b)

        flatten1 = Flatten()
        a_sent = flatten1(x_a)
        b_sent = flatten1(x_b)

        dens1 = Dense(400, activation='relu')
        a_sent = dens1(a_sent)
        b_sent = dens1(b_sent)

        bn_1 = BatchNormalization()
        a_sent = bn_1(a_sent)
        b_sent = bn_1(b_sent)

        similarity = self.cosine()
        dropout = Dropout(self.config.dropout)

        simi = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(a_sent),
                                                                     dropout(b_sent)])

        return simi


class AttCNN(LanguageModel):

    def char_embedding(self, weights):
        sent_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32')

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='char_embedding_layer', trainable=True)
        sent_char_embedding = TimeDistributed(embedding_layer)(sent_char)
        conv_layer = Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1)
        sent_conv = TimeDistributed(conv_layer)(sent_char_embedding)
        max_pooling = GlobalMaxPooling1D()
        sent_mp = TimeDistributed(max_pooling)(sent_conv)
        return Model(sent_char, sent_mp)

    def build(self):
        a = self.a
        b = self.get_b()

        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        a_embedding = embedding_layer(a)
        b_embedding = embedding_layer(b)

        if self.config.level == 'word_level':
            a_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='a_char_base')
            b_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='b_char_base')
            weights_char = np.load(os.path.join(self.config.embedding_path, 'char_level', self.config.embedding_file))
            char_emb = self.char_embedding(weights_char)
            a_char_embedding = char_emb(a_char)
            a_embedding = concatenate([a_embedding, a_char_embedding])
            b_char_embedding = char_emb(b_char)
            b_embedding = concatenate([b_embedding, b_char_embedding])

        matching1 = FullMatching()
        a_matching1 = matching1([a_embedding, GlobalAvgPool1D()(b_embedding)])
        b_matching1 = matching1([b_embedding, GlobalAvgPool1D()(a_embedding)])

        cos = Dot(axes=-1, normalize=True)([a_embedding, b_embedding])
        wb = Lambda(lambda x: softmax(x, axis=1), output_shape=lambda x: x)(cos)
        wa = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=lambda x: x)(cos))

        a_ = Dot(axes=1)([wa, b_embedding])
        b_ = Dot(axes=1)([wb, a_embedding])

        matching3 = AttentiveMatching()
        a_matching3 = matching3([a_embedding, a_])
        b_matching3 = matching3([b_embedding, b_])

        size = K.int_shape(a_embedding)

        a_max = Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), self.config.max_len), x[1]),
            output_shape=lambda x: size)([wb, a_embedding])
        b_max = Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), self.config.max_len), x[1]),
            output_shape=lambda x: size)([wa, b_embedding])

        matching4 = MaxAttentiveMatching()
        a_matching4 = matching4([a_embedding, b_max])
        b_matching4 = matching4([b_embedding, a_max])

        a_matching = concatenate([a_matching1, a_matching3, a_matching4])
        b_matching = concatenate([b_matching1, b_matching3, b_matching4])

        neg = Lambda(lambda x: -x, output_shape=lambda x: x)
        substract1 = Add()([a_embedding, neg(a_)])
        multiply1 = Multiply()([a_embedding, a_])
        substract2 = Add()([b_embedding, neg(b_)])
        multiply2 = Multiply()([b_embedding, b_])

        substract1_max = Add()([a_embedding, neg(b_max)])
        multiply1_max = Multiply()([a_embedding, b_max])
        substract2_max = Add()([b_embedding, neg(a_max)])
        multiply2_max = Multiply()([b_embedding, a_max])

        a_att = concatenate([a_, b_max, multiply1, substract1, multiply1_max, substract1_max])
        b_att = concatenate([b_, a_max, multiply2, substract2, multiply2_max, substract2_max])

        filter_lengths = [2, 3, 4, 5]
        a_conv_layers = []
        b_conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = AttMatchingConv1D(filters=200, kernel_size=filter_length, padding='same',
                                           activation='relu', strides=1)
            a_c = conv_layer([a_embedding, a_att, a_matching])
            b_c = conv_layer([b_embedding, b_att, b_matching])
            a_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(a_c)
            a_flatten = Flatten()(a_maxpooling)
            a_conv_layers.append(a_flatten)
            b_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(b_c)
            b_flatten = Flatten()(b_maxpooling)
            b_conv_layers.append(b_flatten)
        a_conv = concatenate(inputs=a_conv_layers)
        b_conv = concatenate(inputs=b_conv_layers)

        similarity = self.cosine()
        dropout = Dropout(self.config.dropout)

        simi = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(a_conv),
                                                                     dropout(b_conv)])

        return simi


class PreAttCNN(LanguageModel):

    def build(self):
        a = self.a
        b = self.get_b()

        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        a_embedding = embedding_layer(a)
        b_embedding = embedding_layer(b)

        matching1 = FullMatching()
        a_matching1 = matching1([a_embedding, GlobalAvgPool1D()(b_embedding)])
        b_matching1 = matching1([b_embedding, GlobalAvgPool1D()(a_embedding)])

        cos = Dot(axes=-1, normalize=True)([a_embedding, b_embedding])
        wb = Lambda(lambda x: softmax(x, axis=1), output_shape=lambda x: x)(cos)
        wa = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=lambda x: x)(cos))

        a_ = Dot(axes=1)([wa, b_embedding])
        b_ = Dot(axes=1)([wb, a_embedding])

        matching3 = AttentiveMatching()
        a_matching3 = matching3([a_embedding, a_])
        b_matching3 = matching3([b_embedding, b_])

        size = K.int_shape(a_embedding)

        a_max = Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), self.config.max_len), x[1]),
            output_shape=lambda x: size)([wb, a_embedding])
        b_max = Lambda(
            lambda x: K.batch_dot(K.one_hot(K.argmax(x[0], axis=1), self.config.max_len), x[1]),
            output_shape=lambda x: size)([wa, b_embedding])

        matching4 = MaxAttentiveMatching()
        a_matching4 = matching4([a_embedding, b_max])
        b_matching4 = matching4([b_embedding, a_max])

        a_matching = concatenate([a_matching1, a_matching3, a_matching4])
        b_matching = concatenate([b_matching1, b_matching3, b_matching4])

        neg = Lambda(lambda x: -x, output_shape=lambda x: x)
        substract1 = Add()([a_embedding, neg(a_)])
        multiply1 = Multiply()([a_embedding, a_])
        substract2 = Add()([b_embedding, neg(b_)])
        multiply2 = Multiply()([b_embedding, b_])

        substract1_max = Add()([a_embedding, neg(b_max)])
        multiply1_max = Multiply()([a_embedding, b_max])
        substract2_max = Add()([b_embedding, neg(a_max)])
        multiply2_max = Multiply()([b_embedding, a_max])

        a_att = concatenate([a_, b_max, multiply1, substract1, multiply1_max, substract1_max])
        b_att = concatenate([b_, a_max, multiply2, substract2, multiply2_max, substract2_max])

        a_embedding = concatenate([a_embedding, a_att, a_matching])
        b_embedding = concatenate([b_embedding, b_att, b_matching])

        sd = SpatialDropout1D(0.2)
        a_embedding = sd(a_embedding)
        b_embedding = sd(b_embedding)

        filter_lengths = [2, 3, 4, 5]
        a_conv_layers = []
        b_conv_layers = []
        for filter_length in filter_lengths:
            conv_layer = Conv1D(filters=300, kernel_size=filter_length, padding='same',
                                activation='relu', strides=1)
            a_c = conv_layer(a_embedding)
            b_c = conv_layer(b_embedding)
            a_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(a_c)
            a_flatten = Flatten()(a_maxpooling)
            a_conv_layers.append(a_flatten)
            b_maxpooling = MaxPooling1D(pool_size=self.config.max_len)(b_c)
            b_flatten = Flatten()(b_maxpooling)
            b_conv_layers.append(b_flatten)
        a_conv = concatenate(inputs=a_conv_layers)
        b_conv = concatenate(inputs=b_conv_layers)

        similarity = self.cosine()
        dropout = Dropout(self.config.dropout)

        simi = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(a_conv),
                                                                     dropout(b_conv)])

        return simi


class ESIM(LanguageModel):

    def char_embedding(self, weights):
        sent_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32')

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='char_embedding_layer', trainable=True)
        sent_char_embedding = TimeDistributed(embedding_layer)(sent_char)
        conv_layer = Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1)
        sent_conv = TimeDistributed(conv_layer)(sent_char_embedding)
        max_pooling = GlobalMaxPooling1D()
        sent_mp = TimeDistributed(max_pooling)(sent_conv)
        return Model(sent_char, sent_mp)

    def build(self):
        a = self.a
        b = self.get_b()

        weights = np.load(
            os.path.join(self.config.embedding_path, self.config.level + '_level', self.config.embedding_file))

        embedding_layer = Embedding(input_dim=weights.shape[0],
                                    output_dim=weights.shape[-1],
                                    weights=[weights], name='embedding_layer', trainable=True)
        a_embedding = embedding_layer(a)
        b_embedding = embedding_layer(b)

        if self.config.level == 'word_level':
            a_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='a_char_base')
            b_char = Input(shape=(self.config.max_len, self.config.char_per_word), dtype='int32',
                           name='b_char_base')
            weights_char = np.load(os.path.join(self.config.embedding_path, 'char_level', self.config.embedding_file))
            char_emb = self.char_embedding(weights_char)
            a_char_embedding = char_emb(a_char)
            a_embedding = concatenate([a_embedding, a_char_embedding])
            b_char_embedding = char_emb(b_char)
            b_embedding = concatenate([b_embedding, b_char_embedding])

        bilstm_layer = Bidirectional(CuDNNLSTM(300, return_sequences=True))
        a_lstm = bilstm_layer(a_embedding)
        b_lstm = bilstm_layer(b_embedding)

        attention = Dot(axes=-1)([a_lstm, b_lstm])

        wb = Lambda(lambda x: softmax(x, axis=1), output_shape=lambda x: x)(attention)
        wa = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2), output_shape=lambda x: x)(attention))
        a_ = Dot(axes=1)([wa, b_lstm])
        b_ = Dot(axes=1)([wb, a_lstm])

        neg = Lambda(lambda x: -x, output_shape=lambda x: x)
        substract1 = Add()([a_lstm, neg(a_)])
        mutiply1 = Multiply()([a_lstm, a_])
        substract2 = Add()([b_lstm, neg(b_)])
        mutiply2 = Multiply()([b_lstm, b_])

        m_a = concatenate([a_lstm, a_, substract1, mutiply1], axis=-1)
        m_b = concatenate([b_lstm, b_, substract2, mutiply2], axis=-1)

        compose = Bidirectional(CuDNNLSTM(300, return_sequences=True))
        v_a = compose(m_a)
        v_b = compose(m_b)

        a_maxpool = GlobalMaxPool1D()(v_a)
        b_maxpool = GlobalMaxPool1D()(v_b)
        a_avgpool = GlobalAvgPool1D()(v_a)
        b_avgpool = GlobalAvgPool1D()(v_b)
        a = concatenate([a_avgpool, a_maxpool], axis=-1)
        b = concatenate([b_avgpool, b_maxpool], axis=-1)
        similarity = self.cosine()
        dropout = Dropout(self.config.dropout)

        simi = Lambda(similarity, output_shape=lambda _: (None, 1))([dropout(a),
                                                                     dropout(b)])

        return simi
