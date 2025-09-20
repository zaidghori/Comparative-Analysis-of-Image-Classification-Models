import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

from data_handler import DataHandler
from models import MLPModel, VGG16, PretrainedVGG16
from trainer import Trainer
from hyperparameter_search import (
    search_mlp_hyperparameters,
    search_vgg16_hyperparameters,
    search_pretrained_vgg16_hyperparameters,
    print_search_results
)
from utils import (
    predict_on_test,
    compute_metrics,
    compute_map,
    plot_confusion_matrix,
    plot_training_history,
    save_predictions
)


def train_mlp(data_dir, device, search_hyperparameters=True):
    """Train MLP model."""
    print("\n" + "="*50)
    print("Training MLP Model")
    print("="*50)
    
    if search_hyperparameters:
        # Hyperparameter search
        results = search_mlp_hyperparameters(data_dir, img_size=64, device=device)
        best_config, _ = print_search_results(results, "MLP")
    else:
        # Use default best config
        best_config = {
            'hidden_sizes': [1024, 512],
            'lr': 0.0005,
            'dropout': 0.5,
            'batch_size': 64
        }
    
    # Train with best config
    print("\n=== Training MLP with best configuration ===")
    img_size = 64
    
    # Load data
    data_handler = DataHandler(
        data_dir, 
        img_size=img_size, 
        batch_size=best_config['batch_size']
    )
    train_loader, val_loader = data_handler.load_data()
    
    # Create model
    input_size = 3 * img_size * img_size
    model = MLPModel(
        input_size=input_size,
        hidden_sizes=best_config['hidden_sizes'],
        num_classes=data_handler.num_classes,
        dropout=best_config['dropout']
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train
    trainer = Trainer(
        model, device, train_loader, val_loader,
        criterion, optimizer, scheduler,
        log_dir="runs/mlp_final"
    )
    
    best_val_acc = trainer.fit(num_epochs=20)
    print(f"\nFinal MLP validation accuracy: {best_val_acc:.4f}")
    
    # Get training history
    history = trainer.get_history()
    
    # Evaluate
    cm, report, _, _ = compute_metrics(model, val_loader, device, data_handler.classes)
    ap_per_class, mean_ap = compute_map(model, val_loader, device, data_handler.num_classes)
    
    print("\n=== MLP Classification Report ===")
    print(report)
    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")
    
    # Test predictions
    test_dir = os.path.join(data_dir, "test/unknown")
    if os.path.exists(test_dir):
        test_loader = data_handler.get_test_loader(test_dir)
        predictions = predict_on_test(model, test_loader, device, data_handler.classes)
        save_predictions(predictions, "mlp_predictions.txt")
    
    return model, history, cm, mean_ap


def train_vgg16(data_dir, device, search_hyperparameters=True):
    """Train VGG16 model from scratch."""
    print("\n" + "="*50)
    print("Training VGG16 Model (From Scratch)")
    print("="*50)
    
    if search_hyperparameters:
        # Hyperparameter search
        results = search_vgg16_hyperparameters(data_dir, img_size=128, device=device)
        best_config, _ = print_search_results(results, "VGG16")
    else:
        # Use default best config
        best_config = {
            'lr': 0.0001,
            'dropout': 0.3,
            'batch_size': 64
        }
    
    # Train with best config
    print("\n=== Training VGG16 with best configuration ===")
    img_size = 128
    
    # Load data
    data_handler = DataHandler(
        data_dir, 
        img_size=img_size, 
        batch_size=best_config['batch_size']
    )
    train_loader, val_loader = data_handler.load_data()
    
    # Create model
    model = VGG16(
        num_classes=data_handler.num_classes,
        dropout=best_config['dropout'],
        img_size=img_size
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'], weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train
    trainer = Trainer(
        model, device, train_loader, val_loader,
        criterion, optimizer, scheduler,
        log_dir="runs/vgg16_final"
    )
    
    best_val_acc = trainer.fit(num_epochs=30)
    print(f"\nFinal VGG16 validation accuracy: {best_val_acc:.4f}")
    
    # Get training history
    history = trainer.get_history()
    
    # Evaluate
    cm, report, _, _ = compute_metrics(model, val_loader, device, data_handler.classes)
    ap_per_class, mean_ap = compute_map(model, val_loader, device, data_handler.num_classes)
    
    print("\n=== VGG16 Classification Report ===")
    print(report)
    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")
    
    # Test predictions
    test_dir = os.path.join(data_dir, "test/unknown")
    if os.path.exists(test_dir):
        test_loader = data_handler.get_test_loader(test_dir)
        predictions = predict_on_test(model, test_loader, device, data_handler.classes)
        save_predictions(predictions, "vgg16_predictions.txt")
    
    return model, history, cm, mean_ap


def train_pretrained_vgg16(data_dir, device, search_hyperparameters=True):
    """Train Pretrained VGG16 model."""
    print("\n" + "="*50)
    print("Training Pretrained VGG16 Model")
    print("="*50)
    
    if search_hyperparameters:
        # Hyperparameter search
        results = search_pretrained_vgg16_hyperparameters(data_dir, img_size=224, device=device)
        best_config, _ = print_search_results(results, "Pretrained VGG16")
    else:
        # Use default best config
        best_config = {
            'lr': 5e-05,
            'dropout': 0.3,
            'batch_size': 64
        }
    
    # Train with best config
    print("\n=== Training Pretrained VGG16 with best configuration ===")
    img_size = 224
    
    # Load data
    data_handler = DataHandler(
        data_dir, 
        img_size=img_size, 
        batch_size=best_config['batch_size']
    )
    train_loader, val_loader = data_handler.load_data()
    
    # Create model
    model = PretrainedVGG16(
        num_classes=data_handler.num_classes,
        dropout=best_config['dropout'],
        freeze_features=True
    ).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_config['lr'], weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Train
    trainer = Trainer(
        model, device, train_loader, val_loader,
        criterion, optimizer, scheduler,
        log_dir="runs/pretrained_vgg16_final"
    )
    
    best_val_acc = trainer.fit(num_epochs=20)
    print(f"\nFinal Pretrained VGG16 validation accuracy: {best_val_acc:.4f}")
    
    # Get training history
    history = trainer.get_history()
    
    # Evaluate
    cm, report, _, _ = compute_metrics(model, val_loader, device, data_handler.classes)
    ap_per_class, mean_ap = compute_map(model, val_loader, device, data_handler.num_classes)
    
    print("\n=== Pretrained VGG16 Classification Report ===")
    print(report)
    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")
    
    # Test predictions
    test_dir = os.path.join(data_dir, "test/unknown")
    if os.path.exists(test_dir):
        test_loader = data_handler.get_test_loader(test_dir)
        predictions = predict_on_test(model, test_loader, device, data_handler.classes)
        save_predictions(predictions, "pretrained_vgg16_predictions.txt")
    
    return model, history, cm, mean_ap


def main():
    parser = argparse.ArgumentParser(description='Train image classification models')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model', type=str, choices=['mlp', 'vgg16', 'pretrained_vgg16', 'all'],
                        default='all', help='Which model to train')
    parser.add_argument('--search_hyperparameters', action='store_true',
                        help='Whether to perform hyperparameter search')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train models
    if args.model == 'mlp' or args.model == 'all':
        train_mlp(args.data_dir, device, args.search_hyperparameters)
    
    if args.model == 'vgg16' or args.model == 'all':
        train_vgg16(args.data_dir, device, args.search_hyperparameters)
    
    if args.model == 'pretrained_vgg16' or args.model == 'all':
        train_pretrained_vgg16(args.data_dir, device, args.search_hyperparameters)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()