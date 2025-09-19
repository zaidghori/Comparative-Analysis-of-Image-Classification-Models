import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


class DataHandler:
    """Handles data loading and preprocessing for all models."""
    
    def __init__(self, data_dir, img_size=224, batch_size=32, num_workers=2):
        """
        Initialize DataHandler.
        
        Args:
            data_dir: Root directory of the dataset
            img_size: Size to resize images to
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_data(self):
        """Load train, validation, and test datasets."""
        train_path = os.path.join(self.data_dir, "train")
        valid_path = os.path.join(self.data_dir, "valid")
        
        train_dataset = datasets.ImageFolder(train_path, transform=self.train_transform)
        valid_dataset = datasets.ImageFolder(valid_path, transform=self.test_transform)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        )
        
        # Get class names and number of classes
        self.classes = train_dataset.classes
        self.num_classes = len(self.classes)
        
        return train_loader, valid_loader
    
    def get_test_loader(self, test_dir):
        """Create DataLoader for test set."""
        test_dataset = TestDataset(test_dir, transform=self.test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_loader


class TestDataset(Dataset):
    """Custom dataset for test images without labels."""
    
    def __init__(self, folder, transform=None):
        self.files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_path)