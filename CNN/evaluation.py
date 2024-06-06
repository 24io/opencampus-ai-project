import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(model, test_dataset, loss_fn):
    # Initialize variables to store loss and other metrics
    test_loss = 0.0
    num_samples = 0
    
    # Lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through the test dataset
    for features, labels in test_dataset:
        # Get model predictions
        outputs = model(features, training=False)
        
        # Compute the loss
        loss = loss_fn(labels, outputs)
        
        # Aggregate the loss
        test_loss += loss.numpy() * features.shape[0]
        num_samples += features.shape[0]

        # Convert predictions to class labels
        predicted_labels.extend(tf.argmax(outputs, axis=1).numpy())
        true_labels.extend(tf.argmax(labels, axis=1).numpy())
    
    # Average loss over all samples
    average_test_loss = test_loss / num_samples

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Print the metrics
    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'loss': average_test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }