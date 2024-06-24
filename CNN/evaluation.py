# Imports
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


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

        # Apply threshold to get binary predictions
        predicted_labels.extend((outputs.numpy() > 0.5).astype(int))
        true_labels.extend(labels.numpy().astype(int))

    # Average loss over all samples
    average_test_loss = test_loss / num_samples

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Flatten the arrays for metric calculations
    # true_labels_flat = true_labels.flatten()
    # predicted_labels_flat = predicted_labels.flatten()

    # Evaluation metrics
    print(f"Test Loss: {average_test_loss:.4f}")
    metrics = calculate_metrics(true_labels, predicted_labels)

    # Calculate evaluation metrics
    # accuracy = accuracy_score(true_labels_flat, predicted_labels_flat)
    # precision = precision_score(true_labels_flat, predicted_labels_flat, average='micro')
    # recall = recall_score(true_labels_flat, predicted_labels_flat, average='micro')
    # f1 = f1_score(true_labels_flat, predicted_labels_flat, average='micro')
    # mcm = multilabel_confusion_matrix(true_labels_flat, predicted_labels_flat)
    #
    # # Calculate overall TP, TN, FP, and FN
    # tp = np.sum(mcm[:, 1, 1])
    # fn = np.sum(mcm[:, 1, 0])
    # tn = np.sum(mcm[:, 0, 0])
    # fp = np.sum(mcm[:, 0, 1])
    #
    # # Create aggregated confusion matrix
    # aggregated_mcm = np.array([[tn, fp],
    #                            [fn, tp]])
    #
    # # Print the metrics
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"True Positives: {tp}")
    # print(f"False Negatives: {fn}")
    # print(f"True Negatives: {tn}")
    # print(f"False Positives: {fp}")

    return metrics


def calculate_metrics(true_labels: np.ndarray, predicted_labels: np.ndarray):
    """
    :param true_labels: binary ground truth labels (2D np.ndarray of shape (n_samples, n_classes))
    :param predicted_labels: binary predicted labels (2D np.ndarray of shape (n_samples, n_classes))
    :return: dictionary containing evaluation metrics
    """

    # Flatten the arrays for metric calculations
    true_labels_flat = true_labels.flatten()
    predicted_labels_flat = predicted_labels.flatten()

    # Calc metrics
    element_wise_accuracy = np.mean(predicted_labels_flat == true_labels_flat)
    report = classification_report(
        true_labels_flat,
        predicted_labels_flat,
        labels=[0, 1],
        target_names=['no block', 'block'],
        zero_division=0
    )

    # Get confusion matrix
    cm = confusion_matrix(true_labels_flat, predicted_labels_flat)

    # Calculate true positives, false negatives, true negatives, false positives
    tp = np.sum(cm[1, 1])
    fn = np.sum(cm[1, 0])
    tn = np.sum(cm[0, 0])
    fp = np.sum(cm[0, 1])

    # Print the metrics
    print('Element-wise Accuracy:', element_wise_accuracy)
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positives: {tp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")

    return {
        'element_wise_accuracy': element_wise_accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'true_positives': tp,
        'false_negatives': fn,
        'true_negatives': tn,
        'false_positives': fp
    }
