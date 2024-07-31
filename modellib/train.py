import os

import matplotlib.pyplot as plt
from tensorflow.keras import callbacks as cb


def train_model(model, train_dataset, val_dataset, num_epochs, log_dir):
    """
    Trains the model on the training dataset and validates it on the validation dataset.

    :param model: Any Keras model
    :param train_dataset: A tf.data.Dataset object containing training data
    :param val_dataset: A tf.data.Dataset object containing validation data
    :param num_epochs: An integer specifying the number of epochs to train the model
    :param log_dir: Path to the directory where logs should be written
    :return: Trained model and training/validation loss history

    Note:
    - The model should be compiled before calling this function.
    - If validation loss does not improve for 10 consecutive epochs, training will stop early.
    - Model weights are saved every 10 epochs.
    - TensorBoard logs are written to the log_dir.
    - Best weights are restored after training.
    """
    model_class_name = model.__class__.__name__

    # Early stopping callback
    early_stopping = cb.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    # Log model weights every 10 epochs
    checkpoint = cb.ModelCheckpoint(
        filepath=os.path.join(log_dir, f"{model_class_name}.{{epoch:02d}}-{{val_loss:.4f}}.weights.h5"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True
    )
    tensorboard = cb.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Fit model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[early_stopping, checkpoint, tensorboard]
    )

    # Save final weights
    model.save_weights(f'{model_class_name}_final.weights.h5')

    return model, history.history['loss'], history.history['val_loss']


def plot_losses(train_losses, val_losses):
    """
    Plots the training and validation losses.
    :param train_losses: A list of training losses as returned by the train_model function
    :param val_losses: A list of validation losses as returned by the train_model function
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

