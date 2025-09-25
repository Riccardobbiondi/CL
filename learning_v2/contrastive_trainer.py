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
import pandas as pd
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import torch._dynamo
from torch.utils.data.dataloader import get_worker_info
from tqdm import tqdm

# Ottimizzazione: Sopprime gli errori di compilazione (es. Triton su Windows) e torna all'esecuzione standard
torch._dynamo.config.suppress_errors = True

# Configurazione device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

class AirSimContrastiveDataset(Dataset):
    """
    Dataset per contrastive learning con hard negative mining.
    Implementa il lazy loading della matrice di similarità per compatibilità con multiprocessing.
    """
    def __init__(self, dataset_path, similarity_matrix_path, transform=None, max_samples=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        self.similarity_matrix_path = similarity_matrix_path
        self.similarity_matrix = None  # Inizializzato a None, verrà caricato da ogni worker

        print("Pre-caching dataset paths...")
        # Questo viene eseguito nel processo principale, quindi tqdm è sicuro qui
        anchor_dirs = sorted(glob.glob(os.path.join(dataset_path, "anchor_*")))

        if max_samples:
            anchor_dirs = anchor_dirs[:max_samples]

        for anchor_dir in tqdm(anchor_dirs, desc="Caching dataset"):
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
        # Lazy loading della matrice di similarità: ogni worker carica la sua copia
        if self.similarity_matrix is None:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            print(f"[Worker {worker_id}] Loading similarity matrix from {self.similarity_matrix_path}...")
            try:
                df_sim = pd.read_csv(self.similarity_matrix_path, header=0, index_col=0)
                self.similarity_matrix = df_sim.to_numpy()
                print(f"[Worker {worker_id}] Similarity matrix loaded successfully.")
            except Exception as e:
                # Stampa un errore più dettagliato in caso di fallimento
                raise IOError(f"[Worker {worker_id}] Failed to load similarity matrix: {e}")

        anchor_path, positive_paths = self.samples[idx]

        # Carica anchor e positivo
        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_path = np.random.choice(positive_paths)
        positive_img = Image.open(positive_path).convert('RGB')

        # --- Hard Negative Mining ---
        anchor_similarities = self.similarity_matrix[idx]
        
        hard_negative_threshold_min = 0.4
        hard_negative_threshold_max = 0.9
        
        candidate_indices = np.where(
            (anchor_similarities > hard_negative_threshold_min) &
            (anchor_similarities < hard_negative_threshold_max)
        )[0]

        if len(candidate_indices) > 0:
            negative_idx = np.random.choice(candidate_indices)
        else:
            possible_indices = list(range(len(self.samples)))
            possible_indices.remove(idx)
            negative_idx = np.random.choice(possible_indices)
        
        negative_similarity = torch.tensor(anchor_similarities[negative_idx], dtype=torch.float32)
        
        negative_anchor_path, _ = self.samples[negative_idx]
        negative_img = Image.open(negative_anchor_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return {
            'anchor': anchor_img,
            'positive': positive_img,
            'negative': negative_img,
            'negative_similarity': negative_similarity
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
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone.fc = nn.Identity()
            backbone_dim = 512
        elif backbone == 'simple_cnn':
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
    InfoNCE Loss che include hard negatives e temperatura dinamica.
    """
    def __init__(self, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.base_temperature = base_temperature

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings, negative_similarity):
        all_negatives = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        
        temperature = self.base_temperature / (negative_similarity.to(anchor_embeddings.device) + 1e-6)
        temperature = temperature.unsqueeze(1)

        sim_pos = F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=1).unsqueeze(1)
        sim_neg = F.cosine_similarity(anchor_embeddings.unsqueeze(1), all_negatives.unsqueeze(0), dim=2)

        sim_pos /= temperature
        sim_neg /= temperature

        logits = torch.cat([sim_pos, sim_neg], dim=1)
        
        labels = torch.zeros(len(anchor_embeddings), dtype=torch.long).to(logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

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
        
        # API Corretta per GradScaler
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in loop:
            anchor_imgs = batch['anchor'].to(device)
            positive_imgs = batch['positive'].to(device)
            negative_imgs = batch['negative'].to(device)
            neg_sim = batch['negative_similarity'].to(device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                anchor_embeddings = self.model(anchor_imgs)
                positive_embeddings = self.model(positive_imgs)
                negative_embeddings = self.model(negative_imgs)
                
                loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings, neg_sim)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        loop = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in loop:
                anchor_imgs = batch['anchor'].to(device)
                positive_imgs = batch['positive'].to(device)
                negative_imgs = batch['negative'].to(device)
                neg_sim = batch['negative_similarity'].to(device)
                
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    anchor_embeddings = self.model(anchor_imgs)
                    positive_embeddings = self.model(positive_imgs)
                    negative_embeddings = self.model(negative_imgs)
                    
                    loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings, neg_sim)
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs):
        print(f"Starting training for {num_epochs} epochs...")
        
        main_loop = tqdm(range(num_epochs), desc="Epochs")
        for epoch in main_loop:
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            log_msg = f"Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}"
            main_loop.set_postfix_str(log_msg)
            
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
    parser.add_argument('--backbone', type=str, default='simple_cnn', choices=['resnet18', 'simple_cnn'])
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--similarity_matrix', type=str, default="../agent/similarity_matrix.csv", help="Path to the similarity matrix file.")
    
    args = parser.parse_args()
    
    is_worker = get_worker_info() is not None
    
    if os.name == 'nt' and not is_worker:
        NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 2 else 0
    elif os.name == 'nt' and is_worker:
        NUM_WORKERS = 0
    else:
        NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 1 else 0

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, "dataset_final")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"📁 Using dataset: {dataset_path}")
    
    train_transform, val_transform = get_transforms()
    
    full_dataset = AirSimContrastiveDataset(
        dataset_path,
        similarity_matrix_path=args.similarity_matrix,
        transform=train_transform, 
        max_samples=args.max_samples
    )
    
    dataset_size = len(full_dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = val_transform
    
    print(f"📊 Dataset split - Train: {train_size}, Val: {val_size}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    model = ContrastiveEncoder(
        embedding_dim=args.embedding_dim, 
        backbone=args.backbone
    )
    
    if NUM_WORKERS == 0 or os.name != 'nt':
        try:
            model = torch.compile(model)
            print("🚀 Model compiled successfully with torch.compile()!")
        except Exception as e:
            print(f"⚠️ Could not compile model with torch.compile(): {e}. Running un-optimized model.")
    else:
        print("⚠️ Skipping torch.compile() on Windows with num_workers > 0 to ensure compatibility.")

    print(f"🧠 Model: {args.backbone}, Embedding dim: {args.embedding_dim}")
    
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    trainer.train(args.epochs)
    
    final_model_path = f"contrastive_model_final_{timestamp}.pth"
    trainer.save_checkpoint(final_model_path)
    
    plot_path = f"training_losses_final_{timestamp}.png"
    trainer.plot_losses(plot_path)
    
    print(f"✅ Training completed!")
    print(f"📄 Model saved: {final_model_path}")
    print(f"📊 Loss plot saved: {plot_path}")

if __name__ == "__main__":
    main()
