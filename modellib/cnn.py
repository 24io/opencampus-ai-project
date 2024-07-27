from tensorflow.keras import Model as KerasModel
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from modellib.losses import weighted_binary_crossentropy


class Baseline(KerasModel):
    def __init__(self, input_shape):
        super(Baseline, self).__init__()
        self._input_shape = input_shape

        # First bottleneck unit
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('selu')
        self.conv1 = layers.Conv2D(32, kernel_size=(5, 5), padding='same', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.02))

        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation('selu')
        self.conv2 = layers.Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='lecun_normal', kernel_regularizer=l2(0.02))

        # Corner detection
        self.bn3 = layers.BatchNormalization()
        self.padding = layers.ZeroPadding2D(padding=(0, 3))
        self.conv3 = layers.Conv2D(32, kernel_size=(21, 7), padding='valid', activation='tanh', kernel_initializer='glorot_uniform')
        self.conv4 = layers.Conv2D(128, kernel_size=(1, 3), padding='same', activation='tanh', kernel_initializer='glorot_uniform')

        # Fully-connected predictor
        self.flat = layers.Flatten()
        self.classify = layers.Dense(512, activation='sigmoid', kernel_initializer='glorot_uniform')
        self.dropout = layers.Dropout(0.1)
        self.result = layers.Dense(input_shape[1], activation='sigmoid', kernel_initializer='glorot_uniform')


    def build(self, input_shape):
        super(Baseline, self).build(input_shape)

    def call(self, inputs, training=False):
        # First bottleneck unit
        x = self.bn1(inputs, training=training)
        x = self.activation1(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.activation2(x)
        x = self.conv2(x)

        merged = layers.add([inputs, x])

        # Corner detection
        x = self.bn3(merged, training=training)
        x = self.padding(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Fully-connected predictor
        x = self.flat(x)
        x = self.classify(x)
        x = self.dropout(x, training=training)
        x = self.result(x)

        return x

    def build_graph(self):
        x = layers.Input(shape=self.input_shape)
        return KerasModel(inputs=[x], outputs=self.call(x))


# Function to create the model and compile it with the custom loss function
def create_compile_model_custom_loss(input_shape, optimizer, class_weights, metrics=None):
    model = Baseline(input_shape)
    model.build(input_shape=(None,) + input_shape)

    # Compile with custom loss function
    model.compile(
        loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, class_weights),
        optimizer=optimizer,
        metrics=metrics
    )

    return model
