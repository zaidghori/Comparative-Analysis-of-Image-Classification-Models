import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import time

from data_handler import DataHandler
from Models import MLPModel, VGG16, PretrainedVGG16
from trainer import Trainer


def search_mlp_hyperparameters(data_dir, img_size=64, device='cuda'):
    """
    Hyperparameter search for MLP model.
    
    Args:
        data_dir: Path to dataset
        img_size: Image size
        device: Device to train on
    
    Returns:
        List of (config, best_val_acc) tuples
    """
    # Define search space
    hidden_sizes_options = [[1024, 512], [512, 256]]
    learning_rates = [1e-3, 5e-4]
    dropouts = [0.5]
    batch_sizes = [64]
    
    search_space = list(product(hidden_sizes_options, learning_rates, dropouts, batch_sizes))
    results = []
    
    for hidden_sizes, lr, dropout, batch_size in search_space:
        print(f"\n=== Training MLP with hidden={hidden_sizes}, lr={lr}, "
              f"dropout={dropout}, batch={batch_size} ===")
        
        # Load data
        data_handler = DataHandler(data_dir, img_size=img_size, batch_size=batch_size)
        train_loader, val_loader = data_handler.load_data()
        
        # Create model
        input_size = 3 * img_size * img_size
        model = MLPModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=data_handler.num_classes,
            dropout=dropout
        ).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Train
        trainer = Trainer(
            model, device, train_loader, val_loader,
            criterion, optimizer, scheduler,
            log_dir=f"runs/mlp_search/{hidden_sizes}_{lr}_{dropout}_{batch_size}"
        )
        
        best_val_acc = trainer.fit(num_epochs=10)
        
        config = {
            'hidden_sizes': hidden_sizes,
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size
        }
        results.append((config, best_val_acc))
    
    return results


def search_vgg16_hyperparameters(data_dir, img_size=128, device='cuda'):
    """
    Hyperparameter search for VGG16 model.
    
    Args:
        data_dir: Path to dataset
        img_size: Image size
        device: Device to train on
    
    Returns:
        List of (config, best_val_acc) tuples
    """
    # Define search space
    learning_rates = [1e-4, 5e-5]
    dropouts = [0.5, 0.3]
    batch_sizes = [64]
    
    search_space = list(product(learning_rates, dropouts, batch_sizes))
    results = []
    
    for lr, dropout, batch_size in search_space:
        print(f"\n=== Training VGG16 with lr={lr}, dropout={dropout}, batch={batch_size} ===")
        
        # Load data
        data_handler = DataHandler(data_dir, img_size=img_size, batch_size=batch_size)
        train_loader, val_loader = data_handler.load_data()
        
        # Create model
        model = VGG16(
            num_classes=data_handler.num_classes,
            dropout=dropout,
            img_size=img_size
        ).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Train
        trainer = Trainer(
            model, device, train_loader, val_loader,
            criterion, optimizer, scheduler,
            log_dir=f"runs/vgg16_search/{lr}_{dropout}_{batch_size}"
        )
        
        best_val_acc = trainer.fit(num_epochs=20)
        
        config = {
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size
        }
        results.append((config, best_val_acc))
    
    return results


def search_pretrained_vgg16_hyperparameters(data_dir, img_size=224, device='cuda'):
    """
    Hyperparameter search for Pretrained VGG16 model.
    
    Args:
        data_dir: Path to dataset
        img_size: Image size
        device: Device to train on
    
    Returns:
        List of (config, best_val_acc) tuples
    """
    # Define search space
    learning_rates = [1e-4, 5e-5]
    dropouts = [0.5, 0.3]
    batch_sizes = [64]
    
    search_space = list(product(learning_rates, dropouts, batch_sizes))
    results = []
    
    for lr, dropout, batch_size in search_space:
        print(f"\n=== Training Pretrained VGG16 with lr={lr}, dropout={dropout}, batch={batch_size} ===")
        
        # Load data
        data_handler = DataHandler(data_dir, img_size=img_size, batch_size=batch_size)
        train_loader, val_loader = data_handler.load_data()
        
        # Create model
        model = PretrainedVGG16(
            num_classes=data_handler.num_classes,
            dropout=dropout,
            freeze_features=True
        ).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Train
        trainer = Trainer(
            model, device, train_loader, val_loader,
            criterion, optimizer, scheduler,
            log_dir=f"runs/pretrained_vgg16_search/{lr}_{dropout}_{batch_size}"
        )
        
        best_val_acc = trainer.fit(num_epochs=10)
        
        config = {
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size
        }
        results.append((config, best_val_acc))
    
    return results


def print_search_results(results, model_name):
    """Print hyperparameter search results."""
    print(f"\n==== {model_name} Hyperparameter Search Results ====")
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    for config, acc in sorted_results:
        print(f"Config: {config} -> Val Accuracy: {acc:.4f}")
    
    best_config, best_acc = sorted_results[0]
    print(f"\nBest config: {best_config}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    return best_config, best_acc