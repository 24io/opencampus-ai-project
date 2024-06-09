import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Define the training function
def train_model(model, train_dataset, val_dataset, loss_fn, optimizer, num_epochs, log_dir):
    """
    Trains the given model using the provided training and validation datasets, loss function, and optimizer.

    Args:
        model (tf.keras.Model): The model to be trained.
        train_dataset (tf.data.Dataset): The dataset for training the model.
        val_dataset (tf.data.Dataset): The dataset for validating the model during training.
        loss_fn (tf.keras.losses.Loss): The loss function used for training.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer used for training.
        num_epochs (int): The number of epochs to train the model.
        log_dir (str): Directory for storing TensorBoard logs.

    Returns:
        tuple: A tuple containing the trained model, a list of training losses, and a list of validation losses.

    The function performs the following steps:
    1. Creates a TensorBoard writer for logging.
    2. Defines early stopping and model checkpoint callbacks.
    3. Initializes lists to track training and validation losses.
    4. Iterates over the number of epochs:
       - Trains the model for one epoch and logs the training loss.
       - Validates the model and logs the validation loss.
       - Prints the training and validation losses for each epoch.
       - Applies early stopping and saves the best model checkpoint.
    5. If early stopping is triggered, stops training.
    6. Saves the final model weights.
    7. Returns the trained model, training losses, and validation losses.
    """
    # Create TensorBoard writer
    writer = tf.summary.create_file_writer(log_dir)

    # Define early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath="best_model_checkpoint.keras", monitor='val_loss', save_best_only=True)

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        epoch_train_loss = train_one_epoch(model, train_dataset, loss_fn, optimizer)
        train_losses.append(epoch_train_loss)

        # Logging training loss
        with writer.as_default():
            tf.summary.scalar('Loss/Train', epoch_train_loss, step=epoch)

        # Validation phase
        epoch_val_loss = validate_one_epoch(model, val_dataset, loss_fn)
        val_losses.append(epoch_val_loss)

        # Logging validation loss
        with writer.as_default():
            tf.summary.scalar('Loss/Validation', epoch_val_loss, step=epoch)

        print(f"Train Loss: {epoch_train_loss:.4f} | Validation Loss: {epoch_val_loss:.4f}")

        # Early stopping and checkpointing
        early_stopping.on_epoch_end(epoch, logs={'val_loss': epoch_val_loss})
        checkpoint.on_epoch_end(epoch, logs={'val_loss': epoch_val_loss})

        if early_stopping.stopped_epoch > 0:
            print("Early stopping triggered")
            break

    model.save_weights("best_model.keras")

    return model, train_losses, val_losses


# Define function to train for one epoch
def train_one_epoch(model, train_dataset, loss_fn, optimizer):
    """
   Trains the model for one epoch using the provided training dataset, loss function, and optimizer.

   Args:
       model (tf.keras.Model): The model to be trained.
       train_dataset (tf.data.Dataset): The dataset for training the model.
       loss_fn (tf.keras.losses.Loss): The loss function used for training.
       optimizer (tf.keras.optimizers.Optimizer): The optimizer used for training.

   Returns:
       float: The average training loss for the epoch.

   The function performs the following steps:
   1. Initializes the epoch's training loss to zero.
   2. Iterates over the training dataset:
      - Performs a forward pass of the model.
      - Calculates the loss between the model outputs and true labels.
      - Computes the gradients of the loss with respect to the model's trainable variables.
      - Applies the gradients to update the model's weights.
      - Accumulates the batch loss to the epoch's total loss.
   3. Averages the total loss over all batches to obtain the epoch's average training loss.
   4. Returns the average training loss for the epoch.
   """
    epoch_train_loss = 0.0

    for features, labels in train_dataset:
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            loss = loss_fn(labels, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_train_loss += loss.numpy() * features.shape[0]

    epoch_train_loss /= len(train_dataset)
    return epoch_train_loss


# Define function to validate for one epoch
def validate_one_epoch(model, val_dataset, loss_fn):
    """
    Validates the model for one epoch using the provided validation dataset and loss function.

    Args:
        model (tf.keras.Model): The model to be validated.
        val_dataset (tf.data.Dataset): The dataset for validating the model.
        loss_fn (tf.keras.losses.Loss): The loss function used for validation.

    Returns:
        float: The average validation loss for the epoch.

    The function performs the following steps:
    1. Initializes the epoch's validation loss to zero.
    2. Iterates over the validation dataset:
       - Performs a forward pass of the model in inference mode.
       - Calculates the loss between the model outputs and true labels.
       - Accumulates the batch loss to the epoch's total loss.
    3. Averages the total loss over all batches to obtain the epoch's average validation loss.
    4. Returns the average validation loss for the epoch.
    """
    epoch_val_loss = 0.0

    for features, labels in val_dataset:
        outputs = model(features, training=False)
        loss = loss_fn(labels, outputs)

        epoch_val_loss += loss.numpy() * features.shape[0]

    epoch_val_loss /= len(val_dataset)
    return epoch_val_loss

