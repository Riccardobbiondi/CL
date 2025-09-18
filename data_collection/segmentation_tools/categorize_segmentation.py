#!/usr/bin/env python3
"""
Script interattivo per categorizzare manualmente i grigi della segmentation mask
Mostra visualmente le aree e permette di assegnare categorie
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import messagebox, simpledialog

class SegmentationCategorizer:
    def __init__(self, image_path="segmentation_debug.png"):
        self.image_path = image_path
        self.seg_array = None
        self.unique_values = None
        self.categories = {
            'sky': [],
            'trees': [],
            'buildings': [],
            'ground': [],
            'unknown': []
        }
        
    def load_image(self):
        """Carica e analizza l'immagine di segmentazione"""
        if not os.path.exists(self.image_path):
            print(f"❌ File {self.image_path} non trovato")
            return False
            
        try:
            seg_img = Image.open(self.image_path)
            print(f"📁 Caricato {self.image_path}")
            print(f"📐 Dimensioni: {seg_img.size}")
            print(f"🎨 Modalità: {seg_img.mode}")
            
            # Se RGB, prendi solo canale R
            if seg_img.mode == 'RGB':
                self.seg_array = np.array(seg_img)[:, :, 0]
                print("📍 Usando canale R (segmentation IDs)")
            else:
                self.seg_array = np.array(seg_img.convert('L'))
                print("📍 Convertito in grayscale")
                
            self.unique_values = np.unique(self.seg_array)
            print(f"🔍 Valori grigi trovati: {list(self.unique_values)}")
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento: {e}")
            return False
    
    def show_value_preview(self, gray_value):
        """Mostra un'anteprima di dove appare un valore grigio specifico"""
        if self.seg_array is None:
            return None
            
        # Crea maschera per il valore specifico
        mask = (self.seg_array == gray_value)
        
        # Crea immagine colorata per visualizzazione
        preview = np.zeros((self.seg_array.shape[0], self.seg_array.shape[1], 3), dtype=np.uint8)
        
        # Sfondo grigio scuro
        preview[:, :] = [50, 50, 50]
        
        # Evidenzia in giallo le aree con questo valore
        preview[mask] = [255, 255, 0]  # Giallo brillante
        
        return Image.fromarray(preview)
    
    def categorize_interactive_console(self):
        """Categorizzazione via console con anteprime"""
        print(f"\n🎯 CATEGORIZZAZIONE INTERATTIVA")
        print(f"📊 Trovati {len(self.unique_values)} valori grigi unici")
        print(f"📋 Categorie disponibili: sky, trees, buildings, ground, unknown")
        print(f"💡 Digita 'skip' per saltare, 'quit' per uscire")
        
        for i, gray_val in enumerate(self.unique_values):
            # Statistiche del valore
            pixel_count = np.sum(self.seg_array == gray_val)
            percentage = (pixel_count / self.seg_array.size) * 100
            
            print(f"\n📸 Valore {i+1}/{len(self.unique_values)}: Grigio {gray_val}")
            print(f"   📊 {pixel_count} pixel ({percentage:.1f}%)")
            
            # Salva anteprima
            preview = self.show_value_preview(gray_val)
            preview_file = f"preview_gray_{gray_val}.png"
            preview.save(preview_file)
            print(f"   👁️ Anteprima salvata: {preview_file}")
            
            # Chiedi categoria
            while True:
                category = input(f"   🏷️ Categoria per grigio {gray_val} [sky/trees/buildings/ground/unknown/skip/quit]: ").lower().strip()
                
                if category == 'quit':
                    return False
                elif category == 'skip':
                    break
                elif category in self.categories:
                    self.categories[category].append(gray_val)
                    print(f"   ✅ Grigio {gray_val} → {category}")
                    break
                else:
                    print(f"   ❌ Categoria non valida. Usa: {list(self.categories.keys())}")
        
        return True
    
    def auto_suggest_categories(self):
        """Suggerisce categorizzazioni automatiche basate su euristica"""
        print(f"\n🤖 SUGGERIMENTI AUTOMATICI:")
        
        # Analizza distribuzione e posizione
        suggestions = {}
        
        for gray_val in self.unique_values:
            mask = (self.seg_array == gray_val)
            pixel_count = np.sum(mask)
            percentage = (pixel_count / self.seg_array.size) * 100
            
            # Analizza posizione verticale media
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0:
                avg_y = np.mean(y_coords) / self.seg_array.shape[0]  # Normalizzato 0-1
                
                # Euristica semplice
                if avg_y < 0.3 and percentage > 5:  # Parte alta + area significativa
                    suggestions[gray_val] = 'sky'
                elif 0.3 <= avg_y <= 0.8 and percentage > 2:  # Parte media
                    if gray_val < 100:
                        suggestions[gray_val] = 'trees'
                    else:
                        suggestions[gray_val] = 'buildings'
                elif avg_y > 0.8:  # Parte bassa
                    suggestions[gray_val] = 'ground'
                else:
                    suggestions[gray_val] = 'unknown'
                    
                print(f"   Grigio {gray_val:3d}: {suggestions[gray_val]:>9} (y={avg_y:.2f}, {percentage:.1f}%)")
        
        # Chiedi se accettare suggerimenti
        accept = input(f"\n❓ Accettare questi suggerimenti? [y/n]: ").lower().strip()
        if accept == 'y':
            for gray_val, category in suggestions.items():
                self.categories[category].append(gray_val)
            print("✅ Suggerimenti accettati")
            return True
            
        return False
    
    def save_categories(self, filename="segmentation_categories.json"):
        """Salva le categorie in un file JSON"""
        try:
            # Rimuovi liste vuote
            clean_categories = {k: v for k, v in self.categories.items() if v}
            
            # Aggiungi categoria combinata obstacles
            if clean_categories.get('trees') or clean_categories.get('buildings'):
                obstacles = []
                obstacles.extend(clean_categories.get('trees', []))
                obstacles.extend(clean_categories.get('buildings', []))
                clean_categories['obstacles'] = obstacles
            
            with open(filename, 'w') as f:
                json.dump(clean_categories, f, indent=2)
                
            print(f"💾 Categorie salvate in {filename}")
            
            # Genera anche codice Python
            py_filename = "segmentation_config.py"
            with open(py_filename, 'w') as f:
                f.write("# Configurazione categorie segmentazione generata automaticamente\\n")
                f.write("SEGMENTATION_CATEGORIES = {\\n")
                for category, values in clean_categories.items():
                    f.write(f"    '{category}': {values},\\n")
                f.write("}\\n")
                
            print(f"🐍 Configurazione Python salvata in {py_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def print_summary(self):
        """Stampa riassunto delle categorizzazioni"""
        print(f"\\n📋 RIASSUNTO CATEGORIE:")
        total_categorized = 0
        
        for category, values in self.categories.items():
            if values:
                print(f"   {category:>10}: {values}")
                total_categorized += len(values)
        
        uncategorized = len(self.unique_values) - total_categorized
        print(f"\\n📊 Categorizzati: {total_categorized}/{len(self.unique_values)}")
        if uncategorized > 0:
            print(f"⚠️ Non categorizzati: {uncategorized}")
    
    def load_existing_categories(self, filename="segmentation_categories.json"):
        """Carica categorie esistenti se disponibili"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    loaded_categories = json.load(f)
                
                # Aggiorna le categorie
                for category, values in loaded_categories.items():
                    if category in self.categories:
                        self.categories[category] = values
                        
                print(f"📂 Categorie caricate da {filename}")
                return True
                
            except Exception as e:
                print(f"⚠️ Errore caricamento categorie esistenti: {e}")
                
        return False

def main():
    """Funzione principale"""
    print("🎨 CATEGORIZZATORE SEGMENTATION MASK")
    print("="*50)
    
    # Inizializza categorizzatore
    categorizer = SegmentationCategorizer()
    
    # Carica immagine
    if not categorizer.load_image():
        return
    
    # Carica categorie esistenti se disponibili
    categorizer.load_existing_categories()
    
    # Menu principale
    while True:
        print(f"\\n🔧 MENU PRINCIPALE:")
        print("1. Categorizzazione automatica (suggerimenti)")
        print("2. Categorizzazione manuale (console)")
        print("3. Mostra riassunto categorie")
        print("4. Salva categorie")
        print("5. Esci")
        
        choice = input("Scelta [1-5]: ").strip()
        
        if choice == '1':
            categorizer.auto_suggest_categories()
        elif choice == '2':
            if not categorizer.categorize_interactive_console():
                break
        elif choice == '3':
            categorizer.print_summary()
        elif choice == '4':
            categorizer.save_categories()
        elif choice == '5':
            break
        else:
            print("❌ Scelta non valida")
    
    # Salvataggio finale
    categorizer.print_summary()
    
    if any(categorizer.categories.values()):
        save = input("\\n💾 Salvare le categorie prima di uscire? [y/n]: ").lower().strip()
        if save == 'y':
            categorizer.save_categories()
    
    print("👋 Arrivederci!")

if __name__ == "__main__":
    main()
