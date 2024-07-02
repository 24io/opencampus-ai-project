import tensorflow as tf


def weighted_binary_crossentropy(y_true, y_pred, class_weights: dict[int, float]):
    """
    Computes the weighted binary crossentropy loss for a multi-label binary classification problem.
    :param y_true: true label tensor with shape (batch_size, num_classes)
    :param y_pred: predicted label tensor with shape (batch_size, num_classes)
    :param class_weights:
    :return: tensor with shape (batch_size,)
    """

    # Ensure y_true and y_pred have the same shape
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip prediction values to avoid log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # Calculate binary crossentropy
    bce = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

    # Apply class weights
    weights = y_true * class_weights[1] + (1. - y_true) * class_weights[0]

    # Calculate the weighted loss
    weighted_bce = weights * bce

    # Return the mean loss over the batch
    return tf.reduce_mean(weighted_bce)
