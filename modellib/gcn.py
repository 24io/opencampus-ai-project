import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, Model


def create_initial_adj_matrix(num_nodes, window_size=3):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(max(0, i - window_size), min(num_nodes, i + window_size + 1)):
            adj_matrix[i, j] = 1
    return adj_matrix


class GraphConv(layers.Layer):
    def __init__(self, units, activation=None, l2_reg=0.01):
        super().__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name='kernel')
        self.b = self.add_weight(
            shape=(self.units,), initializer='zeros', name='bias')

    def call(self, inputs):
        x, a = inputs
        h = tf.matmul(x, self.w)
        h = tf.matmul(a, h)
        out = h + self.b
        return self.activation(out) if self.activation is not None else out


class GraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, dropout_rate=0.6, activation='relu', l2_reg=0.01):
        super(GraphAttention, self).__init__()
        self.units = units
        self.num_heads = num_heads
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = activations.get(activation)
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.kernels = [self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name=f'kernel_{i}') for i in range(self.num_heads)]
        self.attention_kernels = [self.add_weight(
            shape=(self.units * 2, 1),
            initializer='glorot_uniform',
            regularizer=tf.keras.regularizers.l2(self.l2_reg),
            name=f'att_kernel_{i}') for i in range(self.num_heads)]

    def call(self, inputs):
        x, a = inputs
        outputs = []
        for head in range(self.num_heads):
            attention = self._attention_mechanism(x, a, self.kernels[head], self.attention_kernels[head])
            output = tf.matmul(attention, x)
            outputs.append(output)
        output = tf.concat(outputs, axis=-1)
        output = self.dropout(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def _attention_mechanism(self, x, a, kernel, attention_kernel):
        features = tf.matmul(x, kernel)
        attention_input = tf.concat([
            tf.repeat(features[:, :, tf.newaxis, :], repeats=features.shape[1], axis=2),
            tf.repeat(features[:, tf.newaxis, :, :], repeats=features.shape[1], axis=1)
        ], axis=-1)
        e = tf.squeeze(tf.matmul(attention_input, attention_kernel), axis=-1)
        e = tf.nn.leaky_relu(e)
        mask = tf.cast(a, dtype=tf.bool)
        e = tf.where(mask, e, tf.float32.min)
        attention = tf.nn.softmax(e, axis=-1)
        return attention


def create_gcn_model(num_nodes, num_features, use_attention=True):
    x_input = Input(shape=(num_nodes, num_features))
    a_input = Input(shape=(num_nodes, num_nodes))

    if use_attention:
        # Main branch with attention
        layer1 = GraphAttention(32, num_heads=4)([x_input, a_input])
        layer2 = GraphAttention(16, num_heads=2)([layer1, a_input])

        # Residual branch
        residual = GraphConv(num_features, activation='relu')([x_input, a_input])

        # Combine main and residual branches
        combined = layers.Concatenate()([layer2, residual])

        # Apply dense layers to reduce dimensionality
        dense1 = layers.TimeDistributed(layers.Dense(32, activation='relu'))(combined)
        dense2 = layers.TimeDistributed(layers.Dense(16, activation='relu'))(dense1)
    else:
        # GraphConv branch
        layer1 = GraphConv(32, activation='relu')([x_input, a_input])
        layer2 = GraphConv(16, activation='relu')([layer1, a_input])
        residual = GraphConv(16, activation='relu')([x_input, a_input])
        dense2 = layers.Add()([layer2, residual])

    # Output layer for multi-label binary classification
    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(dense2)
    output = layers.Reshape((num_nodes,))(output)

    model = Model(inputs=[x_input, a_input], outputs=output)
    return model