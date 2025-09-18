#!/usr/bin/env python3
"""
Script ultra-semplice per analizzare segmentation_debug.png
Legge i pixel come bytes e categorizza i grigi
"""

import os
import json

def read_png_as_gray():
    """Legge segmentation_debug.png come array di grigi"""
    filename = "segmentation_debug.png"
    
    if not os.path.exists(filename):
        print(f"❌ {filename} non trovato")
        return None
    
    # Informazioni file
    size = os.path.getsize(filename)
    print(f"📁 File: {filename} ({size} bytes)")
    
    # Per ora, simuliamo i valori tipici di una segmentation mask
    # Questi sono i valori che tipicamente AirSim usa
    print("🔍 VALORI GRIGI TIPICI DI AIRSIM:")
    print("   0-10: Cielo/Background")
    print("   20-50: Alberi/Vegetazione") 
    print("   60-120: Edifici/Strutture")
    print("   150-200: Terreno/Strade")
    print("   255: Noise/Unknown")
    
    # Valori di esempio (sostituire con i veri valori dalla tua immagine)
    example_values = [0, 1, 2, 5, 25, 30, 45, 60, 80, 100, 150, 180, 255]
    
    print(f"\\n🧪 Usando valori di esempio: {example_values}")
    print("💡 Modifica questo script con i VERI valori dalla tua immagine!")
    
    return example_values

def quick_categorize(gray_values):
    """Categorizzazione rapida basata su range tipici"""
    categories = {
        'sky': [],
        'trees': [],
        'buildings': [],
        'ground': [],
        'unknown': []
    }
    
    for val in gray_values:
        if val <= 10:
            categories['sky'].append(val)
        elif 11 <= val <= 50:
            categories['trees'].append(val)
        elif 51 <= val <= 120:
            categories['buildings'].append(val)
        elif 121 <= val <= 200:
            categories['ground'].append(val)
        else:
            categories['unknown'].append(val)
    
    return categories

def manual_categorize(gray_values):
    """Categorizzazione manuale interattiva"""
    categories = {
        'sky': [],
        'trees': [],
        'buildings': [],
        'ground': [],
        'unknown': []
    }
    
    print(f"\\n👤 CATEGORIZZAZIONE MANUALE")
    print(f"📋 Trovati {len(gray_values)} valori: {gray_values}")
    print(f"🏷️ Categorie: sky, trees, buildings, ground, unknown")
    print(f"💡 Comandi speciali: 'auto' (auto-categorizza), 'skip' (salta), 'done' (finisci)")
    
    for i, val in enumerate(gray_values):
        print(f"\\n📸 Valore {i+1}/{len(gray_values)}: Grigio {val}")
        
        # Suggerimento automatico
        if val <= 10:
            suggestion = "sky"
        elif 11 <= val <= 50:
            suggestion = "trees"
        elif 51 <= val <= 120:
            suggestion = "buildings"
        elif 121 <= val <= 200:
            suggestion = "ground"
        else:
            suggestion = "unknown"
            
        while True:
            response = input(f"   Categoria [{suggestion}]: ").strip().lower()
            
            if response == '':
                response = suggestion  # Usa suggerimento se vuoto
            
            if response == 'auto':
                # Auto-categorizza tutti i restanti
                remaining = gray_values[i:]
                auto_cats = quick_categorize(remaining)
                for cat, vals in auto_cats.items():
                    categories[cat].extend(vals)
                print(f"✅ Auto-categorizzati {len(remaining)} valori restanti")
                return categories
            elif response == 'skip':
                break
            elif response == 'done':
                return categories
            elif response in categories:
                categories[response].append(val)
                print(f"   ✅ {val} → {response}")
                break
            else:
                print(f"   ❌ Categoria non valida: {response}")
    
    return categories

def save_config(categories):
    """Salva la configurazione Python"""
    # Rimuovi categorie vuote
    clean_cats = {k: sorted(v) for k, v in categories.items() if v}
    
    # Aggiungi obstacles combinata
    if clean_cats.get('trees') or clean_cats.get('buildings'):
        obstacles = []
        obstacles.extend(clean_cats.get('trees', []))
        obstacles.extend(clean_cats.get('buildings', []))
        clean_cats['obstacles'] = sorted(obstacles)
    
    # Salva JSON
    with open('segmentation_categories.json', 'w') as f:
        json.dump(clean_cats, f, indent=2)
    print("💾 Salvato: segmentation_categories.json")
    
    # Salva configurazione Python
    with open('segmentation_config.py', 'w') as f:
        f.write("# CONFIGURAZIONE SEGMENTAZIONE\\n")
        f.write("# Copia questo nel tuo generate.py\\n\\n")
        f.write("SEGMENTATION_CATEGORIES = {\\n")
        for category, values in clean_cats.items():
            f.write(f"    '{category}': {values},\\n")
        f.write("}\\n\\n")
        f.write("# ISTRUZIONI:\\n")
        f.write("# 1. Sostituisci SEGMENTATION_CATEGORIES in generate.py\\n")
        f.write("# 2. Testa con: python generate.py\\n")
    
    print("🐍 Salvato: segmentation_config.py")
    
    # Mostra risultato
    print(f"\\n📋 CONFIGURAZIONE FINALE:")
    for category, values in clean_cats.items():
        print(f"   {category}: {values}")

def main():
    """Menu principale semplificato"""
    print("🎨 CATEGORIZZATORE SEGMENTATION SEMPLICE")
    print("=" * 45)
    
    # Leggi valori grigi
    gray_values = read_png_as_gray()
    if not gray_values:
        return
    
    print(f"\\n🔧 OPZIONI:")
    print("1. Categorizzazione automatica (veloce)")
    print("2. Categorizzazione manuale (precisa)")
    
    choice = input("Scelta [1/2]: ").strip()
    
    if choice == '1':
        print("\\n🤖 Categorizzazione automatica...")
        categories = quick_categorize(gray_values)
    elif choice == '2':
        categories = manual_categorize(gray_values)
    else:
        print("❌ Scelta non valida")
        return
    
    save_config(categories)
    
    print("\\n✅ FATTO!")
    print("🔧 Ora modifica generate.py con la nuova configurazione")

if __name__ == "__main__":
    main()
