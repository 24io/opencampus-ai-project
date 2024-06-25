# Import necessary libraries

import tensorflow as tf
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Dense, Dropout


class GCNModel(KerasModel):
    def __init__(self, input_shape):
        super(GCNModel, self).__init__()
        self._input_shape = input_shape

        # GCN layers
        self.gc1 = GCNConv(64, activation='relu')
        self.gc2 = GCNConv(32, activation='relu')

        # Fully connected layers
        self.dense1 = Dense(128, activation='relu')
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(1, activation='sigmoid')  # Changed to 1 for binary classification

    def build(self, input_shape):
        super(GCNModel, self).build(input_shape)

    def call(self, inputs, training=False):
        x, a = inputs

        # Apply GCN layers
        x = self.gc1([x, a])
        x = self.gc2([x, a])

        # Apply fully connected layers
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

    def build_graph(self):
        x = Input(shape=self.input_shape)
        return KerasModel(inputs=[x], outputs=self.call(x))


class GraphDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.adj = self.create_adjacency_matrix(x.shape[2])

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = batch_x.transpose(0, 2, 1, 3).squeeze(axis=-1)
        batch_adj = np.tile(self.adj, (len(batch_x), 1, 1))

        # Convert to TensorFlow tensors
        batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
        batch_adj = tf.convert_to_tensor(batch_adj, dtype=tf.float32)
        batch_y = tf.convert_to_tensor(batch_y, dtype=tf.float32)

        return [batch_x, batch_adj], batch_y

    def create_adjacency_matrix(self, num_nodes):
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            if i > 0:
                adj[i, i - 1] = 1
            adj[i, i] = 1
            if i < num_nodes - 1:
                adj[i, i + 1] = 1
        return adj


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, Input, Model


def reshape_bands(matrixbands):
    return np.squeeze(matrixbands).transpose(0, 2, 1)


def create_initial_adj_matrix(num_nodes, window_size=3):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(max(0, i - window_size), min(num_nodes, i + window_size + 1)):
            adj_matrix[i, j] = 1
    return adj_matrix


class GraphConv(layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            initializer='glorot_uniform',
            name='kernel')
        self.b = self.add_weight(
            shape=(self.units,), initializer='zeros', name='bias')

    def call(self, inputs):
        x, a = inputs
        h = tf.matmul(x, self.w)
        h = tf.matmul(a, h)
        out = h + self.b
        return self.activation(out) if self.activation is not None else out


def create_gcn_model(num_nodes, num_features):
    x_input = Input(shape=(num_nodes, num_features))
    a_input = Input(shape=(num_nodes, num_nodes))

    gc1 = GraphConv(64, activation='relu')([x_input, a_input])
    gc2 = GraphConv(32, activation='relu')([gc1, a_input])
    gc3 = GraphConv(1)([gc2, a_input])

    output = layers.Flatten()(gc3)
    output = layers.Dense(num_nodes, activation='sigmoid')(output)

    model = Model(inputs=[x_input, a_input], outputs=output)
    return model
