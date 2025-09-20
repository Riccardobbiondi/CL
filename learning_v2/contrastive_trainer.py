#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contrastive Learning Trainer per dataset AirSim
Addestra un agente usando contrastive learning su anchor/positives
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import torch._dynamo
from torch.utils.data.dataloader import get_worker_info

# Ottimizzazione: Sopprime gli errori di compilazione (es. Triton su Windows) e torna all'esecuzione standard
torch._dynamo.config.suppress_errors = True

# Configurazione device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AirSimContrastiveDataset(Dataset):
    """
    Dataset per contrastive learning con anchor/positives da AirSim
    (Versione ottimizzata con pre-caching dei percorsi)
    """
    def __init__(self, dataset_path, transform=None, max_samples=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []

        print("Pre-caching dataset paths...")
        # Trova tutte le cartelle anchor_XXXXX
        anchor_dirs = sorted(glob.glob(os.path.join(dataset_path, "anchor_*")))

        if max_samples:
            anchor_dirs = anchor_dirs[:max_samples]

        for anchor_dir in anchor_dirs:
            anchor_path = os.path.join(anchor_dir, "anchor.png")
            positive_paths = glob.glob(os.path.join(anchor_dir, "positive_*.png"))

            if os.path.exists(anchor_path) and positive_paths:
                self.samples.append((anchor_path, positive_paths))

        print(f"Found {len(self.samples)} valid anchor/positive pairs.")

        if len(self.samples) == 0:
            raise ValueError(f"No valid anchor/positive pairs found in {dataset_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, positive_paths = self.samples[idx]

        # Carica anchor
        anchor_img = Image.open(anchor_path).convert('RGB')

        # Scegli un positivo casuale dalla lista pre-caricata
        positive_path = np.random.choice(positive_paths)
        positive_img = Image.open(positive_path).convert('RGB')

        # Applica trasformazioni se specificate
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)

        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'anchor_dir': os.path.basename(os.path.dirname(anchor_path))
        }

class L2Norm(nn.Module):
    """Layer per normalizzazione L2"""
    def __init__(self, dim=1):
        super(L2Norm, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

class ContrastiveEncoder(nn.Module):
    """
    Encoder per contrastive learning - estrae features dalle immagini
    """
    def __init__(self, embedding_dim=128, backbone='resnet18'):
        super(ContrastiveEncoder, self).__init__()
        
        if backbone == 'resnet18':
            import torchvision.models as models
            self.backbone = models.resnet18(pretrained=True)
            # Rimuovi l'ultimo layer di classificazione
            self.backbone.fc = nn.Identity()
            backbone_dim = 512
        elif backbone == 'simple_cnn':
            # CNN semplice per training più veloce
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            backbone_dim = 256
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Projection head per contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            L2Norm(dim=1)  # Normalizza gli embeddings
        )
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection_head(features)
        return embeddings

class ContrastiveLoss(nn.Module):
    """
    InfoNCE Loss per contrastive learning
    """
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, anchor_embeddings, positive_embeddings):
        # Calcola similarità coseno
        similarity_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature
        
        # Labels: ogni anchor è simile al suo positivo corrispondente
        labels = torch.arange(len(anchor_embeddings)).to(similarity_matrix.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss

        # !!! manca temperatura -> qui è fissa
        # raccogliere informazioni su temperature migliori in tempo reale

class ContrastiveTrainer:
    """
    Trainer per contrastive learning
    """
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = ContrastiveLoss()
        
        # Ottimizzazione: GradScaler per Automatic Mixed Precision (AMP) - API aggiornata
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            anchor_imgs = batch['anchor'].to(device)
            positive_imgs = batch['positive'].to(device)
            
            self.optimizer.zero_grad()
            
            # Ottimizzazione: Automatic Mixed Precision (AMP) - API aggiornata
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                # Forward pass
                anchor_embeddings = self.model(anchor_imgs)
                positive_embeddings = self.model(positive_imgs)
                
                # Calculate loss
                loss = self.criterion(anchor_embeddings, positive_embeddings)
            
            # Backward pass con scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                anchor_imgs = batch['anchor'].to(device)
                positive_imgs = batch['positive'].to(device)
                
                # Ottimizzazione: AMP anche in validazione per coerenza - API aggiornata
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    anchor_embeddings = self.model(anchor_imgs)
                    positive_embeddings = self.model(positive_imgs)
                    
                    loss = self.criterion(anchor_embeddings, positive_embeddings)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print progress
            if val_loss is not None:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        print("Training completed!")
    
    def save_checkpoint(self, filename):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filename)
        print(f"Checkpoint saved: {filename}")
    
    def plot_losses(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

def get_transforms(img_size=224):
    """
    Definisce le trasformazioni per le immagini
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def main():
    parser = argparse.ArgumentParser(description='Contrastive Learning for AirSim Dataset')
    parser.add_argument('--backbone', type=str, default='simple_cnn', 
                        choices=['resnet18', 'simple_cnn'], help='Backbone architecture')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to use (for testing)')
    parser.add_argument('--val_split', type=float, default=0.2, 
                        help='Validation split ratio')
    
    args = parser.parse_args()
    
    # --- Inizio Ottimizzazioni Main ---
    
    # Ottimizzazione: Abilita il caricamento dati parallelo in modo sicuro su Windows
    # get_worker_info() restituisce None nel processo principale, permettendoci di impostare
    # num_workers > 0 solo quando non siamo in un processo worker.
    is_worker = get_worker_info() is not None
    
    # Impostazioni per ottimizzazione
    if os.name == 'nt' and not is_worker:
        # Su Windows, nel processo principale, calcoliamo il numero di workers
        NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 2 else 0
    elif os.name == 'nt' and is_worker:
        # Nei processi figli su Windows, num_workers deve essere 0
        NUM_WORKERS = 0
    else:
        # Su altri OS (es. Linux), possiamo usare più workers senza problemi
        NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 1 else 0

    # Abilita TF32 per GPU Ampere (velocizza ulteriormente senza perdita di precisione)
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # --- Fine Ottimizzazioni Main ---

    # Path al dataset
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory
    dataset_path = os.path.join(base_dir, "dataset_final")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"📁 Using dataset: {dataset_path}")
    
    # Preparazione transforms
    train_transform, val_transform = get_transforms()
    
    # Carica dataset completo
    full_dataset = AirSimContrastiveDataset(
        dataset_path, 
        transform=train_transform, 
        max_samples=args.max_samples
    )
    
    # Split train/validation
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Aggiorna transform per validation
    val_dataset.dataset.transform = val_transform
    
    print(f"📊 Dataset split - Train: {train_size}, Val: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True # Ottimizzazione: trasferimenti più veloci a GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True # Ottimizzazione: trasferimenti più veloci a GPU
    )
    
    # Modello
    model = ContrastiveEncoder(
        embedding_dim=args.embedding_dim, 
        backbone=args.backbone
    )
    
    # Ottimizzazione: Compila il modello con torch.compile (per PyTorch 2.0+)
    # NOTA: La compilazione viene saltata su Windows se si usano i worker (num_workers > 0)
    # per evitare problemi di compatibilità con il multiprocessing.
    if NUM_WORKERS == 0 or os.name != 'nt':
        try:
            model = torch.compile(model)
            print("🚀 Model compiled successfully with torch.compile()!")
        except Exception as e:
            print(f"⚠️ Could not compile model with torch.compile(): {e}. Running un-optimized model.")
    else:
        print("⚠️ Skipping torch.compile() on Windows with num_workers > 0 to ensure compatibility.")


    print(f"🧠 Model: {args.backbone}, Embedding dim: {args.embedding_dim}")
    
    # Trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr
    )
    
    # Training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trainer.train(args.epochs)
    
    # Salva modello finale
    final_model_path = f"contrastive_model_final_{timestamp}.pth"
    trainer.save_checkpoint(final_model_path)
    
    # Plot delle loss
    plot_path = f"training_losses_final_{timestamp}.png"
    trainer.plot_losses(plot_path)
    
    print(f"✅ Training completed!")
    print(f"📄 Model saved: {final_model_path}")
    print(f"📊 Loss plot saved: {plot_path}")

if __name__ == "__main__":
    main()
