# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer


class FullMatching(Layer):
    def __init__(self, perspective_num=10, **kwargs):
        self.perspective_num = perspective_num
        self.kernel = None
        super(FullMatching, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.max_len = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel', shape=(self.perspective_num, self.dim),
                                      initializer='glorot_uniform')
        super(FullMatching, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sent1 = inputs[0]
        sent2 = inputs[1]

        v1 = K.expand_dims(sent1, -2) * self.kernel
        v2 = self.kernel * K.expand_dims(sent2, axis=1)
        v2 = K.expand_dims(v2, 1)
        v1 = K.l2_normalize(v1, axis=-1)
        v2 = K.l2_normalize(v2, axis=-1)
        matching = K.sum(v1 * v2, axis=-1)
        return matching

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-2], self.perspective_num)


class MaxMatching(Layer):
    def __init__(self, perspective_num=10, **kwargs):
        self.perspective_num = perspective_num
        self.kernel = None
        super(MaxMatching, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.max_len = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel', shape=(self.perspective_num, self.dim),
                                      initializer='glorot_uniform')
        super(MaxMatching, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sent1 = inputs[0]
        sent2 = inputs[1]

        v1 = K.expand_dims(sent1, -2) * self.kernel
        v2 = K.expand_dims(sent2, -2) * self.kernel
        v1 = K.l2_normalize(v1, axis=-1)
        v2 = K.l2_normalize(v2, axis=-1)
        matching = K.max(K.sum(K.expand_dims(v1, 2) * K.expand_dims(v2, 1), axis=-1), axis=-2)
        return matching

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-2], self.perspective_num)


class AttentiveMatching(Layer):
    def __init__(self, perspective_num=10, **kwargs):
        self.perspective_num = perspective_num
        self.kernel = None
        super(AttentiveMatching, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.max_len = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel', shape=(self.perspective_num, self.dim),
                                      initializer='glorot_uniform')
        super(AttentiveMatching, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sent1 = inputs[0]
        sent2 = inputs[1]

        v1 = K.expand_dims(sent1, -2) * self.kernel
        v2 = K.expand_dims(sent2, -2) * self.kernel
        v1 = K.l2_normalize(v1, axis=-1)
        v2 = K.l2_normalize(v2, axis=-1)
        matching = K.sum(v1 * v2, axis=-1)
        return matching

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-2], self.perspective_num)


class MaxAttentiveMatching(Layer):
    def __init__(self, perspective_num=10, **kwargs):
        self.perspective_num = perspective_num
        self.kernel = None
        super(MaxAttentiveMatching, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.max_len = input_shape[0][1]
        self.kernel = self.add_weight(name='kernel', shape=(self.perspective_num, self.dim),
                                      initializer='glorot_uniform')
        super(MaxAttentiveMatching, self).build(input_shape)

    def call(self, inputs, **kwargs):
        sent1 = inputs[0]
        sent2 = inputs[1]

        v1 = K.expand_dims(sent1, -2) * self.kernel
        v2 = K.expand_dims(sent2, -2) * self.kernel
        v1 = K.l2_normalize(v1, axis=-1)
        v2 = K.l2_normalize(v2, axis=-1)
        matching = K.sum(v1 * v2, axis=-1)
        return matching

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][-2], self.perspective_num)
