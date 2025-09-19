import os
import time
import copy
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Handles training and validation of models."""
    
    def __init__(self, model, device, train_loader, val_loader, criterion,
                 optimizer, scheduler=None, log_dir="runs", model_dir="models"):
        """
        Initialize Trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to run training on
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            log_dir: Directory for TensorBoard logs
            model_dir: Directory to save model checkpoints
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Setup logging
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{timestamp}")
        
        # Setup model directory
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
    
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for imgs, labels in pbar:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar("train/loss", epoch_loss, epoch)
        self.writer.add_scalar("train/acc", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
            for imgs, labels in pbar:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)
                
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Log to TensorBoard
        self.writer.add_scalar("val/loss", epoch_loss, epoch)
        self.writer.add_scalar("val/acc", epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def fit(self, num_epochs, save_best=True):
        """Train the model for specified number of epochs."""
        best_acc = 0.0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_acc = self.train_one_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if save_best and val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                model_name = self.model.__class__.__name__
                save_path = os.path.join(
                    self.model_dir,
                    f"best_{model_name}_epoch{epoch}_acc{val_acc:.3f}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model: {save_path}")
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Close TensorBoard writer
        self.writer.close()
        
        return best_acc
    
    def get_history(self):
        """Return training history."""
        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs
        }