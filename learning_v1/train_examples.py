#!/usr/bin/env python3
"""
Script di esempio per training contrastive learning
"""

# Esempi di comandi per diversi scenari:

# 1. Training veloce per test (dataset v4, CNN semplice, pochi samples)
# python contrastive_trainer.py --dataset_version v4 --backbone simple_cnn --epochs 10 --max_samples 100

# 2. Training completo con ResNet18 (più lento ma migliore qualità)
# python contrastive_trainer.py --dataset_version v4 --backbone resnet18 --epochs 50 --batch_size 16

# 3. Training su dataset v3 con parametri personalizzati  
# python contrastive_trainer.py --dataset_version v3 --embedding_dim 256 --lr 5e-4 --epochs 100

# 4. Training con validation split maggiore
# python contrastive_trainer.py --dataset_version v4 --val_split 0.3

print("=== AirSim Contrastive Learning Trainer ===")
print()
print("Esempi di utilizzo:")
print()
print("1. Training veloce per test:")
print("   python contrastive_trainer.py --dataset_version v4 --backbone simple_cnn --epochs 10 --max_samples 100")
print()
print("2. Training completo con ResNet18:")
print("   python contrastive_trainer.py --dataset_version v4 --backbone resnet18 --epochs 50")
print()
print("3. Training personalizzato:")
print("   python contrastive_trainer.py --dataset_version v3 --embedding_dim 256 --lr 5e-4")
print()
print("4. Lista parametri disponibili:")
print("   python contrastive_trainer.py --help")
print()
print("Nota: Installa prima le dipendenze con:")
print("   pip install -r requirements.txt")
print()

# Versioni disponibili
import os
import glob

base_dir = os.path.dirname(os.path.dirname(__file__))
datasets = glob.glob(os.path.join(base_dir, "dataset_*"))

if datasets:
    print("Dataset disponibili:")
    for dataset_path in sorted(datasets):
        dataset_name = os.path.basename(dataset_path)
        anchor_count = len(glob.glob(os.path.join(dataset_path, "anchor_*")))
        print(f"   - {dataset_name}: {anchor_count} samples")
else:
    print("⚠️  Nessun dataset trovato. Esegui prima generate.py per creare un dataset.")
