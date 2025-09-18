#!/usr/bin/env python3
"""
Script per categorizzare i grigi della segmentation mask con VISUALIZZAZIONE COLORI
Mostra il colore grigio nel terminale per aiutare la categorizzazione
"""

import os
import json
import struct

class ColorfulCategorizer:
    def __init__(self, image_path="segmentation_debug.png"):
        self.image_path = image_path
        self.categories = {
            'sky': [],
            'trees': [],
            'buildings': [],
            'ground': [],
            'unknown': []
        }
        
        # Valori di esempio (modifica con i tuoi veri valori)
        self.gray_values = [0, 1, 2, 5, 10, 15, 25, 30, 45, 50, 60, 80, 100, 120, 150, 180, 200, 255]
        
    def show_gray_color(self, gray_value):
        """Mostra il colore grigio usando escape codes ANSI"""
        # Converte valore grigio (0-255) in escape code per background
        # ANSI 256-color: 232-255 sono i grigi (24 livelli)
        ansi_gray = 232 + int((gray_value / 255) * 23)
        
        # Determina colore testo (bianco o nero) per contrasto
        text_color = "30" if gray_value > 127 else "37"  # nero se grigio chiaro, bianco se scuro
        
        # Crea blocco colorato
        color_block = f"\\033[48;5;{ansi_gray}m\\033[{text_color}m  {gray_value:3d}  \\033[0m"
        
        return color_block
    
    def show_color_palette(self):
        """Mostra una palette di tutti i grigi da categorizzare"""
        print("\\n🎨 PALETTE COLORI DA CATEGORIZZARE:")
        print("=" * 60)
        
        # Mostra in righe di 8 colori
        for i in range(0, len(self.gray_values), 8):
            row = self.gray_values[i:i+8]
            
            # Riga con i colori
            color_row = " ".join(self.show_gray_color(val) for val in row)
            print(f"   {color_row}")
            
            # Riga con i numeri sotto (per riferimento)
            number_row = " ".join(f"  {val:3d}  " for val in row)
            print(f"   {number_row}")
            print()
    
    def categorize_with_colors(self):
        """Categorizzazione interattiva con visualizzazione colori"""
        print("\\n👤 CATEGORIZZAZIONE VISUALE")
        print("🏷️ Categorie: sky, trees, buildings, ground, unknown")
        print("💡 Comandi: 'palette' (mostra tutti), 'summary' (riassunto), 'auto' (auto-categorizza restanti), 'done' (finisci)")
        
        for i, gray_val in enumerate(self.gray_values):
            print(f"\\n📸 Valore {i+1}/{len(self.gray_values)}:")
            
            # Mostra il colore in grande
            big_color_block = self.show_gray_color(gray_val)
            print(f"   🎨 Colore: {big_color_block} {big_color_block} {big_color_block}")
            
            # Mostra nel contesto con colori vicini
            context_start = max(0, i-3)
            context_end = min(len(self.gray_values), i+4)
            context_values = self.gray_values[context_start:context_end]
            
            print("   📍 Contesto:")
            context_row = " ".join(
                f"[{self.show_gray_color(val)}]" if val == gray_val 
                else self.show_gray_color(val)
                for val in context_values
            )
            print(f"      {context_row}")
            
            # Suggerimento automatico
            suggestion = self.suggest_category(gray_val)
            
            while True:
                response = input(f"   🏷️ Categoria per grigio {gray_val} [{suggestion}]: ").strip().lower()
                
                if response == '':
                    response = suggestion
                
                if response == 'palette':
                    self.show_color_palette()
                elif response == 'summary':
                    self.print_summary()
                elif response == 'auto':
                    # Auto-categorizza i restanti
                    remaining = self.gray_values[i:]
                    for val in remaining:
                        auto_cat = self.suggest_category(val)
                        self.categories[auto_cat].append(val)
                    print(f"✅ Auto-categorizzati {len(remaining)} valori restanti")
                    return True
                elif response == 'done':
                    return True
                elif response in self.categories:
                    self.categories[response].append(gray_val)
                    print(f"   ✅ {gray_val} → {response}")
                    break
                else:
                    print(f"   ❌ Comando non valido: {response}")
        
        return True
    
    def suggest_category(self, gray_value):
        """Suggerisci categoria basata su valore grigio"""
        if gray_value <= 10:
            return 'sky'
        elif 11 <= gray_value <= 50:
            return 'trees'
        elif 51 <= gray_value <= 120:
            return 'buildings'
        elif 121 <= gray_value <= 220:
            return 'ground'
        else:
            return 'unknown'
    
    def print_summary(self):
        """Stampa riassunto con colori"""
        print(f"\\n📋 RIASSUNTO CATEGORIE:")
        print("=" * 50)
        
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                
                # Mostra nome categoria
                print(f"\\n🏷️ {category.upper()}:")
                
                # Mostra colori in questa categoria
                color_row = " ".join(self.show_gray_color(val) for val in sorted_values)
                print(f"   {color_row}")
                
                # Mostra valori numerici
                print(f"   Valori: {sorted_values}")
        
        # Mostra non categorizzati
        categorized = set()
        for values in self.categories.values():
            categorized.update(values)
        
        uncategorized = [val for val in self.gray_values if val not in categorized]
        if uncategorized:
            print(f"\\n⚠️ NON CATEGORIZZATI:")
            color_row = " ".join(self.show_gray_color(val) for val in uncategorized)
            print(f"   {color_row}")
            print(f"   Valori: {uncategorized}")
    
    def save_config(self):
        """Salva configurazione finale"""
        # Pulizia
        clean = {k: sorted(v) for k, v in self.categories.items() if v}
        
        # Aggiungi obstacles
        if clean.get('trees') or clean.get('buildings'):
            obstacles = []
            obstacles.extend(clean.get('trees', []))
            obstacles.extend(clean.get('buildings', []))
            clean['obstacles'] = sorted(obstacles)
        
        # Salva JSON
        with open('segmentation_visual_config.json', 'w') as f:
            json.dump(clean, f, indent=2)
        
        # Salva Python
        with open('segmentation_visual_config.py', 'w') as f:
            f.write("# CONFIGURAZIONE SEGMENTAZIONE (Categorizzazione Visuale)\\n")
            f.write("# Sostituisci in generate.py\\n\\n")
            f.write("SEGMENTATION_CATEGORIES = {\\n")
            for category, values in clean.items():
                f.write(f"    '{category}': {values},\\n")
            f.write("}\\n")
        
        print(f"\\n💾 CONFIGURAZIONE SALVATA:")
        print(f"   📄 segmentation_visual_config.json")
        print(f"   🐍 segmentation_visual_config.py")
        
        return clean
    
    def load_real_values_from_file(self):
        """Carica valori reali dal file segmentation_debug.png se disponibile"""
        if not os.path.exists(self.image_path):
            print(f"⚠️ {self.image_path} non trovato, usando valori di esempio")
            return False
        
        try:
            # Leggi file info
            size = os.path.getsize(self.image_path)
            print(f"📁 Trovato {self.image_path} ({size} bytes)")
            
            # Per ora usiamo valori di esempio, ma puoi aggiornare con i veri valori
            print("💡 Usando valori di esempio. Aggiorna self.gray_values con i tuoi veri valori!")
            
            # Se hai i veri valori, aggiornali qui:
            # self.gray_values = [i veri valori dalla tua immagine]
            
            return True
            
        except Exception as e:
            print(f"❌ Errore lettura file: {e}")
            return False

def test_colors():
    """Testa la visualizzazione colori"""
    print("🧪 TEST VISUALIZZAZIONE COLORI:")
    print("=" * 40)
    
    test_values = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255]
    
    categorizer = ColorfulCategorizer()
    
    for val in test_values:
        color_block = categorizer.show_gray_color(val)
        print(f"   Grigio {val:3d}: {color_block}")
    
    print("\\n💡 Se vedi i colori correttamente, il terminale supporta ANSI colors!")

def main():
    """Menu principale"""
    print("🎨 CATEGORIZZATORE VISUALE SEGMENTATION MASK")
    print("=" * 55)
    
    # Test colori
    test_colors()
    
    # Inizializza
    categorizer = ColorfulCategorizer()
    categorizer.load_real_values_from_file()
    
    # Menu
    while True:
        print(f"\\n🔧 MENU:")
        print("1. Mostra palette completa")
        print("2. Categorizzazione visuale interattiva")
        print("3. Riassunto categorie")
        print("4. Salva configurazione")
        print("5. Esci")
        
        choice = input("Scelta [1-5]: ").strip()
        
        if choice == '1':
            categorizer.show_color_palette()
        elif choice == '2':
            categorizer.categorize_with_colors()
        elif choice == '3':
            categorizer.print_summary()
        elif choice == '4':
            categorizer.save_config()
        elif choice == '5':
            break
        else:
            print("❌ Scelta non valida")
    
    # Salvataggio finale
    if any(categorizer.categories.values()):
        save = input("\\n💾 Salvare prima di uscire? [y/n]: ").lower().strip()
        if save == 'y':
            config = categorizer.save_config()
            
            print(f"\\n✅ COMPLETATO!")
            print(f"🔧 Copia SEGMENTATION_CATEGORIES da segmentation_visual_config.py")
            print(f"🔧 Incollalo in generate.py per usare la nuova configurazione")
    
    print("👋 Arrivederci!")

if __name__ == "__main__":
    main()
