#!/usr/bin/env python3
"""
Script per leggere VERI valori pixel da segmentation_debug.png
Funziona anche senza PIL usando solo lettura raw
"""

import os
import struct
import zlib
import json

def read_png_pixels(filename):
    """Legge pixel da PNG senza PIL"""
    try:
        with open(filename, 'rb') as f:
            # Verifica PNG signature
            signature = f.read(8)
            if signature != b'\\x89PNG\\r\\n\\x1a\\n':
                print("❌ Non è un file PNG valido")
                return None
            
            # Leggi IHDR chunk
            f.read(4)  # chunk length
            chunk_type = f.read(4)
            if chunk_type != b'IHDR':
                print("❌ IHDR chunk non trovato")
                return None
                
            width = struct.unpack('>I', f.read(4))[0]
            height = struct.unpack('>I', f.read(4))[0]
            bit_depth = struct.unpack('B', f.read(1))[0]
            color_type = struct.unpack('B', f.read(1))[0]
            
            print(f"📐 PNG info: {width}x{height}, bit_depth={bit_depth}, color_type={color_type}")
            
            # Salta il resto dell'IHDR
            f.read(7)
            
            # Cerca IDAT chunks e leggi i dati
            idat_data = b''
            while True:
                try:
                    chunk_length = struct.unpack('>I', f.read(4))[0]
                    chunk_type = f.read(4)
                    
                    if chunk_type == b'IDAT':
                        idat_data += f.read(chunk_length)
                        f.read(4)  # CRC
                    elif chunk_type == b'IEND':
                        break
                    else:
                        f.read(chunk_length + 4)  # Skip chunk + CRC
                except:
                    break
            
            if not idat_data:
                print("❌ Nessun dato IDAT trovato")
                return None
            
            # Decomprimi i dati
            try:
                raw_data = zlib.decompress(idat_data)
                print(f"✅ Dati decompressi: {len(raw_data)} bytes")
            except:
                print("❌ Errore decompressione")
                return None
            
            # Estrai pixel (semplificato per grayscale)
            pixels = []
            bytes_per_pixel = 1 if color_type == 0 else 3  # 0=grayscale, 2=RGB
            bytes_per_row = width * bytes_per_pixel + 1  # +1 per filter byte
            
            for y in range(height):
                row_start = y * bytes_per_row + 1  # +1 per saltare filter byte
                for x in range(width):
                    if color_type == 0:  # Grayscale
                        pixel_idx = row_start + x
                        if pixel_idx < len(raw_data):
                            pixels.append(raw_data[pixel_idx])
                    elif color_type == 2:  # RGB - prendi solo canale R
                        pixel_idx = row_start + x * 3
                        if pixel_idx < len(raw_data):
                            pixels.append(raw_data[pixel_idx])
            
            return pixels
            
    except Exception as e:
        print(f"❌ Errore lettura PNG: {e}")
        return None

def analyze_pixels(pixels):
    """Analizza i pixel per trovare valori unici"""
    if not pixels:
        return []
    
    # Trova valori unici
    unique_values = sorted(list(set(pixels)))
    total_pixels = len(pixels)
    
    print(f"\\n🔍 ANALISI PIXEL:")
    print(f"📊 Pixel totali: {total_pixels}")
    print(f"🎯 Valori unici: {len(unique_values)}")
    print(f"📋 Range: {min(unique_values)} - {max(unique_values)}")
    print(f"📝 Tutti i valori: {unique_values}")
    
    # Conta frequenze
    print(f"\\n📈 DISTRIBUZIONE:")
    value_counts = {}
    for val in unique_values:
        count = pixels.count(val)
        percentage = (count / total_pixels) * 100
        value_counts[val] = {'count': count, 'percentage': percentage}
        
    # Mostra top 15
    sorted_by_freq = sorted(value_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"{'Grigio':>6} {'Pixel':>8} {'%':>6}")
    print("-" * 22)
    for val, stats in sorted_by_freq[:15]:
        print(f"{val:>6} {stats['count']:>8} {stats['percentage']:>5.1f}")
    
    return unique_values

def smart_categorize(unique_values):
    """Categorizzazione intelligente basata su valori AirSim tipici"""
    categories = {
        'sky': [],
        'trees': [],
        'buildings': [],
        'ground': [],
        'unknown': []
    }
    
    print(f"\\n🤖 CATEGORIZZAZIONE AUTOMATICA:")
    
    for val in unique_values:
        # Logica migliorata per AirSim
        if val == 0:
            categories['sky'].append(val)
            reason = "valore 0 = background/cielo"
        elif 1 <= val <= 15:
            categories['sky'].append(val)
            reason = "valori bassi = cielo"
        elif 16 <= val <= 60:
            categories['trees'].append(val)
            reason = "valori medi-bassi = vegetazione"
        elif 61 <= val <= 130:
            categories['buildings'].append(val)
            reason = "valori medi = strutture"
        elif 131 <= val <= 220:
            categories['ground'].append(val)
            reason = "valori alti = terreno"
        else:
            categories['unknown'].append(val)
            reason = "valori estremi = rumore"
            
        print(f"   Grigio {val:3d} → {reason}")
    
    return categories

def interactive_categorize(unique_values):
    """Categorizzazione interattiva"""
    categories = {
        'sky': [],
        'trees': [],
        'buildings': [],
        'ground': [],
        'unknown': []
    }
    
    print(f"\\n👤 CATEGORIZZAZIONE INTERATTIVA:")
    print(f"📝 Valori da categorizzare: {unique_values}")
    print(f"🏷️ Categorie: sky, trees, buildings, ground, unknown")
    print(f"💡 Comandi: 'auto' (finisci automaticamente), 'quit' (esci)")
    
    for i, val in enumerate(unique_values):
        # Suggerimento automatico
        if val == 0 or val <= 15:
            suggestion = "sky"
        elif 16 <= val <= 60:
            suggestion = "trees"
        elif 61 <= val <= 130:
            suggestion = "buildings"
        elif 131 <= val <= 220:
            suggestion = "ground"
        else:
            suggestion = "unknown"
        
        print(f"\\n📸 Valore {i+1}/{len(unique_values)}: Grigio {val}")
        
        while True:
            response = input(f"   Categoria [{suggestion}]: ").strip().lower()
            
            if response == '':
                response = suggestion
            
            if response == 'auto':
                # Auto-categorizza i restanti
                remaining = unique_values[i:]
                auto_cats = smart_categorize(remaining)
                for cat, vals in auto_cats.items():
                    categories[cat].extend(vals)
                print(f"✅ Auto-categorizzati {len(remaining)} valori restanti")
                return categories
            elif response == 'quit':
                return categories
            elif response in categories:
                categories[response].append(val)
                print(f"   ✅ {val} → {response}")
                break
            else:
                print(f"   ❌ Categoria non valida: {response}")
    
    return categories

def save_final_config(categories):
    """Salva configurazione finale"""
    # Pulizia e ordinamento
    clean = {k: sorted(v) for k, v in categories.items() if v}
    
    # Aggiungi obstacles
    if clean.get('trees') or clean.get('buildings'):
        obstacles = []
        obstacles.extend(clean.get('trees', []))
        obstacles.extend(clean.get('buildings', []))
        clean['obstacles'] = sorted(obstacles)
    
    # JSON
    with open('segmentation_categories_final.json', 'w') as f:
        json.dump(clean, f, indent=2)
    
    # Python config
    with open('segmentation_config_final.py', 'w') as f:
        f.write("# ===== CONFIGURAZIONE SEGMENTAZIONE FINALE =====\\n")
        f.write("# Generata automaticamente dai pixel reali\\n")
        f.write("# Sostituisci SEGMENTATION_CATEGORIES in generate.py\\n\\n")
        f.write("SEGMENTATION_CATEGORIES = {\\n")
        for category, values in clean.items():
            f.write(f"    '{category}': {values},\\n")
        f.write("}\\n\\n")
        f.write("# ISTRUZIONI USO:\\n")
        f.write("# 1. Copia tutto SEGMENTATION_CATEGORIES\\n")
        f.write("# 2. Incolla in generate.py (sovrascrivi quello esistente)\\n")
        f.write("# 3. Esegui: python generate.py\\n")
        f.write("# 4. Controlla i file debug_masks/ per verificare\\n")
    
    print(f"\\n💾 SALVATO:")
    print(f"   📄 segmentation_categories_final.json")
    print(f"   🐍 segmentation_config_final.py")
    
    print(f"\\n📋 CONFIGURAZIONE FINALE:")
    for category, values in clean.items():
        print(f"   {category:>10}: {values}")
        
    return clean

def main():
    """Funzione principale"""
    print("🎨 LETTORE PIXEL SEGMENTATION_DEBUG.PNG")
    print("=" * 50)
    
    filename = "segmentation_debug.png"
    
    if not os.path.exists(filename):
        print(f"❌ {filename} non trovato nella directory corrente")
        print("💡 Assicurati di essere in data_collection/ e di aver generato la debug mask")
        return
    
    # Leggi pixel reali
    print("📖 Lettura pixel reali...")
    pixels = read_png_pixels(filename)
    
    if not pixels:
        print("❌ Impossibile leggere i pixel")
        return
    
    # Analizza
    unique_values = analyze_pixels(pixels)
    
    if not unique_values:
        print("❌ Nessun valore trovato")
        return
    
    # Menu categorizzazione
    print(f"\\n🔧 OPZIONI CATEGORIZZAZIONE:")
    print("1. Automatica (veloce, basata su euristica)")
    print("2. Interattiva (precisa, controllo manuale)")
    
    choice = input("Scelta [1/2]: ").strip()
    
    if choice == '1':
        categories = smart_categorize(unique_values)
    elif choice == '2':
        categories = interactive_categorize(unique_values)
    else:
        print("❌ Scelta non valida")
        return
    
    # Salva risultato
    final_config = save_final_config(categories)
    
    print(f"\\n✅ COMPLETATO!")
    print(f"🔧 Ora copia SEGMENTATION_CATEGORIES da segmentation_config_final.py")
    print(f"🔧 Incollalo in generate.py e testa con: python generate.py")

if __name__ == "__main__":
    main()
