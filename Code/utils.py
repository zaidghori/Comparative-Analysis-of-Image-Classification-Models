import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.preprocessing import label_binarize


def predict_on_test(model, test_loader, device, class_names):
    """
    Make predictions on test set.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run inference on
        class_names: List of class names
    
    Returns:
        List of (filename, predicted_class) tuples
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for imgs, img_paths in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            for path, pred in zip(img_paths, preds):
                predictions.append((path, class_names[pred.item()]))
    
    return predictions


def compute_metrics(model, loader, device, class_names):
    """
    Compute confusion matrix and classification report.
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device
        class_names: List of class names
    
    Returns:
        confusion_matrix, classification_report, true_labels, predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=class_names, digits=4)
    
    return cm, report, all_labels, all_preds


def compute_map(model, loader, device, n_classes):
    """
    Compute mean Average Precision (mAP).
    
    Args:
        model: Trained model
        loader: DataLoader
        device: Device
        n_classes: Number of classes
    
    Returns:
        per_class_ap, mean_ap
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.hstack(all_labels)
    
    # Binarize labels for one-vs-rest evaluation
    labels_bin = label_binarize(all_labels, classes=np.arange(n_classes))
    
    ap_per_class = []
    for i in range(n_classes):
        try:
            ap = average_precision_score(labels_bin[:, i], all_probs[:, i])
        except ValueError:  # If class not present
            ap = np.nan
        ap_per_class.append(ap)
    
    mean_ap = np.nanmean(ap_per_class)
    
    return ap_per_class, mean_ap


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title="Training History"):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def save_predictions(predictions, filename):
    """Save predictions to file."""
    with open(filename, 'w') as f:
        for img_file, pred_class in predictions:
            f.write(f"{img_file},{pred_class}\n")
    print(f"Predictions saved to {filename}")