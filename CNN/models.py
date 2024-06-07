# Import necessary libraries
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, Activation, BatchNormalization, ZeroPadding2D
from tensorflow.keras.regularizers import l2

## Baseline Model
class Baseline(Model):
    def __init(self, input_shape):
        super(Baseline, self).__init__()
        self.input_shape = Input(shape=input_shape)

        # First bottleneck unit
        self.bn1 = BatchNormalization()
        self.activation1 = Activation('selu')
        self.conv1 = Conv2D(32, kernel_size=(5, 5), padding='same', kernel_regularizer=l2(0.02))
        
        self.bn2 = BatchNormalization()
        self.activation2 = Activation('selu')
        self.conv2 = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.02))
        
        # Corner detection
        self.bn3 = BatchNormalization()
        self.padding = ZeroPadding2D(padding=(0, 3))
        self.conv3 = Conv2D(32, kernel_size=(21, 7), padding='valid', activation='tanh')
        self.conv4 = Conv2D(128, kernel_size=(1, 3), padding='same', activation='tanh')
        
        # Fully-connected predictor
        self.flat = Flatten()
        self.classify = Dense(512, activation='sigmoid')
        self.dropout = Dropout(0.1)
        self.result = Dense(input_shape[1], activation='sigmoid')

    def call(self, inputs):
        # First bottleneck unit
        x = self.bn1(inputs)
        x = self.activation_1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.activation_2(x)
        x = self.conv2(x)
        
        merged = tf.keras.layers.add([inputs, x])
        
        # Corner detection
        x = self.bn3(merged)
        x = self.padding(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Fully-connected predictor
        x = self.flat(x)
        x = self.classify(x)
        x = self.dropout(x)
        x = self.result(x)
        
        return x
