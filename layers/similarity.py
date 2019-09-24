# -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dense, Activation, Multiply, Add, Lambda
from keras.initializers import Constant


class SimAttention(Layer):
    def __init__(self, **kwargs):
        super(SimAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dim = input_shape[0][-1] * 3
        self.W = self.add_weight(name='W_simi', shape=(self.dim, 1), initializer='glorot_uniform')
        super(SimAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        H = inputs[0]
        U = inputs[1]
        num_context_words = K.shape(H)[1]
        num_query_words = K.shape(U)[1]
        H_repeat = K.concatenate([[1, 1], [num_query_words], [1]], 0)
        U_repeat = K.concatenate([[1], [num_context_words], [1, 1]], 0)
        H = K.tile(K.expand_dims(H, axis=2), H_repeat)
        U = K.tile(K.expand_dims(U, axis=1), U_repeat)
        M = H * U
        concatenated = K.concatenate([H, U, M], axis=-1)
        # test = K.dot(concatenated, self.W)
        dot_product = K.squeeze(K.dot(concatenated, self.W), axis=-1)
        return dot_product

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])


class C2QAttention(Layer):

    def __init__(self, **kwargs):
        super(C2QAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(C2QAttention, self).build(input_shape)

    def call(self, inputs):
        s, q = inputs
        s = K.softmax(s, axis=-1)
        return K.batch_dot(s, q)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[1][-1]


class Q2CAttention(Layer):

    def __init__(self, **kwargs):
        super(Q2CAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Q2CAttention, self).build(input_shape)

    def call(self, inputs):
        s, c = inputs
        max_s = K.max(s, axis=-1)
        context_to_query_attention = K.softmax(max_s)
        h = K.batch_dot(context_to_query_attention, c)
        expanded_weighted_sum = K.expand_dims(h, 1)
        num_of_repeatations = K.shape(c)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[1][-1]


class Highway(Layer):

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):
        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        dim = input_shape[-1]
        transform_gate_bias_initializer = Constant(self.transform_gate_bias)
        self.dense_1 = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)
        self.dense_1.build(input_shape)
        self.dense_2 = Dense(units=dim)
        self.dense_2.build(input_shape)
        self.trainable_weights = self.dense_1.trainable_weights + self.dense_2.trainable_weights

        super(Highway, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dim = K.int_shape(x)[-1]
        transform_gate = self.dense_1(x)
        transform_gate = Activation("sigmoid")(transform_gate)
        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        transformed_data = self.dense_2(x)
        transformed_data = Activation(self.activation)(transformed_data)
        transformed_gated = Multiply()([transform_gate, transformed_data])
        identity_gated = Multiply()([carry_gate, x])
        value = Add()([transformed_gated, identity_gated])
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Highway, self).get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config
