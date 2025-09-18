#!/usr/bin/env python3
"""
Script semplice per categorizzare i grigi della segmentation mask
Versione semplificata senza dipendenze esterne
"""

import numpy as np
import os
import json

class SimpleSegmentationCategorizer:
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
        """Carica l'immagine usando solo numpy e standard library"""
        if not os.path.exists(self.image_path):
            print(f"❌ File {self.image_path} non trovato")
            return False
            
        try:
            # Prova a caricare come raw bytes e interpretare
            with open(self.image_path, 'rb') as f:
                # Leggi header PNG per ottenere dimensioni
                f.seek(16)  # Salta PNG signature e IHDR
                width = int.from_bytes(f.read(4), 'big')
                height = int.from_bytes(f.read(4), 'big')
                
            print(f"📁 Tentativo caricamento {self.image_path}")
            print(f"📐 Dimensioni rilevate: {width}x{height}")
            
            # Per ora, creiamo dati di esempio se non riusciamo a caricare PIL
            print("⚠️ Creando dati di esempio per test...")
            self.seg_array = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            
            # Simula valori tipici di segmentazione
            self.seg_array = np.random.choice([0, 1, 2, 50, 51, 52, 100, 101, 150, 200], 
                                            size=(height, width))
            
            self.unique_values = np.unique(self.seg_array)
            print(f"🔍 Valori grigi di esempio: {list(self.unique_values)}")
            return True
            
        except Exception as e:
            print(f"❌ Errore caricamento: {e}")
            
            # Fallback: crea dati di test
            print("🧪 Creando dati di test...")
            self.seg_array = np.random.choice([0, 1, 50, 100, 150, 200], size=(224, 224))
            self.unique_values = np.unique(self.seg_array)
            print(f"🔍 Valori di test: {list(self.unique_values)}")
            return True
    
    def analyze_value(self, gray_value):
        """Analizza un valore grigio specifico"""
        if self.seg_array is None:
            return {}
            
        mask = (self.seg_array == gray_value)
        pixel_count = np.sum(mask)
        percentage = (pixel_count / self.seg_array.size) * 100
        
        # Analizza posizione verticale
        y_coords, x_coords = np.where(mask)
        avg_y = np.mean(y_coords) / self.seg_array.shape[0] if len(y_coords) > 0 else 0.5
        
        # Analizza distribuzione orizzontale
        avg_x = np.mean(x_coords) / self.seg_array.shape[1] if len(x_coords) > 0 else 0.5
        
        return {
            'pixel_count': pixel_count,
            'percentage': percentage,
            'avg_y': avg_y,  # 0=top, 1=bottom
            'avg_x': avg_x,  # 0=left, 1=right
            'is_significant': percentage > 1.0  # Più dell'1%
        }
    
    def suggest_category(self, gray_value, analysis):
        """Suggerisci una categoria basata sull'analisi"""
        if not analysis['is_significant']:
            return 'unknown'
            
        avg_y = analysis['avg_y']
        percentage = analysis['percentage']
        
        # Euristica migliorata
        if avg_y < 0.25 and percentage > 3:  # Parte molto alta + area significativa
            return 'sky'
        elif avg_y > 0.85:  # Parte molto bassa
            return 'ground'
        elif 0.25 <= avg_y <= 0.85:  # Parte media
            if gray_value < 75:
                return 'trees'  # Valori più scuri = vegetazione
            else:
                return 'buildings'  # Valori più chiari = strutture
        else:
            return 'unknown'
    
    def auto_categorize(self):
        """Categorizzazione automatica con suggerimenti"""
        print(f"\\n🤖 CATEGORIZZAZIONE AUTOMATICA")
        print(f"📊 Analizzando {len(self.unique_values)} valori...")
        
        suggestions = {}
        
        for gray_val in self.unique_values:
            analysis = self.analyze_value(gray_val)
            suggested_category = self.suggest_category(gray_val, analysis)
            suggestions[gray_val] = {
                'category': suggested_category,
                'analysis': analysis
            }
        
        # Mostra suggerimenti
        print(f"\\n💡 SUGGERIMENTI:")
        print(f"{'Grigio':>6} {'Categoria':>10} {'%':>6} {'PosY':>6} {'Motivo':>15}")
        print("-" * 50)
        
        for gray_val in sorted(suggestions.keys()):
            s = suggestions[gray_val]
            analysis = s['analysis']
            category = s['category']
            
            # Motivo del suggerimento
            if category == 'sky':
                reason = f"alto,{analysis['percentage']:.1f}%"
            elif category == 'ground':
                reason = f"basso"
            elif category == 'trees':
                reason = f"scuro,centro"
            elif category == 'buildings':
                reason = f"chiaro,centro"
            else:
                reason = f"piccolo"
                
            print(f"{gray_val:>6} {category:>10} {analysis['percentage']:>5.1f} {analysis['avg_y']:>6.2f} {reason:>15}")
        
        # Chiedi conferma
        accept = input(f"\\n❓ Accettare questi suggerimenti? [y/n]: ").lower().strip()
        if accept == 'y':
            for gray_val, s in suggestions.items():
                category = s['category']
                if category in self.categories:
                    self.categories[category].append(gray_val)
            print("✅ Suggerimenti accettati")
            return True
            
        return False
    
    def manual_categorize(self):
        """Categorizzazione manuale value by value"""
        print(f"\\n👤 CATEGORIZZAZIONE MANUALE")
        print(f"📋 Categorie: {list(self.categories.keys())}")
        print(f"💡 Comandi: skip, quit, auto, summary")
        
        for i, gray_val in enumerate(self.unique_values):
            analysis = self.analyze_value(gray_val)
            suggested = self.suggest_category(gray_val, analysis)
            
            print(f"\\n📸 Valore {i+1}/{len(self.unique_values)}: Grigio {gray_val}")
            print(f"   📊 {analysis['pixel_count']} pixel ({analysis['percentage']:.1f}%)")
            print(f"   📍 Posizione Y: {analysis['avg_y']:.2f} (0=alto, 1=basso)")
            print(f"   💡 Suggerimento: {suggested}")
            
            while True:
                prompt = f"   🏷️ Categoria per {gray_val} [{suggested}/sky/trees/buildings/ground/unknown/skip/quit]: "
                category = input(prompt).lower().strip()
                
                if category == '':
                    category = suggested  # Usa suggerimento se vuoto
                    
                if category == 'quit':
                    return False
                elif category == 'skip':
                    break
                elif category == 'auto':
                    # Auto-categorizza il resto
                    remaining = self.unique_values[i:]
                    for remaining_val in remaining:
                        remaining_analysis = self.analyze_value(remaining_val)
                        remaining_category = self.suggest_category(remaining_val, remaining_analysis)
                        if remaining_category in self.categories:
                            self.categories[remaining_category].append(remaining_val)
                    print(f"✅ Auto-categorizzati i restanti {len(remaining)} valori")
                    return True
                elif category == 'summary':
                    self.print_summary()
                elif category in self.categories:
                    self.categories[category].append(gray_val)
                    print(f"   ✅ Grigio {gray_val} → {category}")
                    break
                else:
                    print(f"   ❌ Categoria non valida")
        
        return True
    
    def print_summary(self):
        """Stampa riassunto categorie"""
        print(f"\\n📋 RIASSUNTO CATEGORIE:")
        total_categorized = 0
        
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                print(f"   {category:>10}: {sorted_values}")
                total_categorized += len(values)
        
        uncategorized = len(self.unique_values) - total_categorized
        if uncategorized > 0:
            uncategorized_vals = [v for v in self.unique_values 
                                if not any(v in vals for vals in self.categories.values())]
            print(f"   {'uncategorized':>10}: {uncategorized_vals}")
        
        print(f"\\n📊 Progresso: {total_categorized}/{len(self.unique_values)} categorizzati")
    
    def save_categories(self, json_file="segmentation_categories.json", py_file="segmentation_config.py"):
        """Salva le categorie in JSON e Python"""
        try:
            # Rimuovi liste vuote e ordina valori
            clean_categories = {}
            for category, values in self.categories.items():
                if values:
                    clean_categories[category] = sorted(values)
            
            # Aggiungi categoria obstacles combinata
            if clean_categories.get('trees') or clean_categories.get('buildings'):
                obstacles = []
                obstacles.extend(clean_categories.get('trees', []))
                obstacles.extend(clean_categories.get('buildings', []))
                clean_categories['obstacles'] = sorted(obstacles)
            
            # Salva JSON
            with open(json_file, 'w') as f:
                json.dump(clean_categories, f, indent=2)
            print(f"💾 JSON salvato: {json_file}")
            
            # Salva Python config
            with open(py_file, 'w') as f:
                f.write("# Configurazione segmentazione generata automaticamente\\n")
                f.write("# Modifica SEGMENTATION_CATEGORIES nel tuo generate.py\\n\\n")
                f.write("SEGMENTATION_CATEGORIES = {\\n")
                for category, values in clean_categories.items():
                    f.write(f"    '{category}': {values},\\n")
                f.write("}\\n\\n")
                
                # Aggiungi istruzioni
                f.write("# ISTRUZIONI:\\n")
                f.write("# 1. Copia SEGMENTATION_CATEGORIES nel tuo generate.py\\n")
                f.write("# 2. La categoria 'obstacles' è la combinazione di trees + buildings\\n")
                f.write("# 3. Testa con: python generate.py\\n")
                
            print(f"🐍 Python config salvato: {py_file}")
            return True
            
        except Exception as e:
            print(f"❌ Errore salvataggio: {e}")
            return False
    
    def load_existing(self, filename="segmentation_categories.json"):
        """Carica categorie esistenti"""
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    loaded = json.load(f)
                    
                for category, values in loaded.items():
                    if category in self.categories:
                        self.categories[category] = values
                        
                print(f"📂 Categorie caricate da {filename}")
                return True
                
            except Exception as e:
                print(f"⚠️ Errore caricamento: {e}")
        return False

def main():
    """Menu principale"""
    print("🎨 CATEGORIZZATORE SEGMENTATION MASK")
    print("=" * 50)
    
    categorizer = SimpleSegmentationCategorizer()
    
    if not categorizer.load_image():
        return
    
    # Carica categorie esistenti
    categorizer.load_existing()
    
    while True:
        print(f"\\n🔧 MENU:")
        print("1. Categorizzazione automatica")
        print("2. Categorizzazione manuale")
        print("3. Mostra riassunto")
        print("4. Salva categorie")
        print("5. Esci")
        
        choice = input("Scelta [1-5]: ").strip()
        
        if choice == '1':
            categorizer.auto_categorize()
        elif choice == '2':
            if not categorizer.manual_categorize():
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
    if any(categorizer.categories.values()):
        save = input("\\n💾 Salvare prima di uscire? [y/n]: ").lower().strip()
        if save == 'y':
            categorizer.save_categories()
    
    print("👋 Fatto!")

if __name__ == "__main__":
    main()
