#!/usr/bin/env python3
"""
Script per analizzare la segmentation_debug.png e scoprire i valori grigi
"""

import numpy as np
from PIL import Image
import os

def analyze_segmentation_debug():
    """Analizza segmentation_debug.png per categorizzare i grigi"""
    
    debug_file = "segmentation_debug.png"
    
    if not os.path.exists(debug_file):
        print(f"❌ File {debug_file} non trovato")
        print("💡 Genera prima una segmentation mask eseguendo generate.py")
        return
        
    try:
        # Carica immagine segmentazione
        seg_img = Image.open(debug_file)
        print(f"📁 Analizzando {debug_file}")
        print(f"📐 Dimensioni: {seg_img.size}")
        print(f"🎨 Modalità: {seg_img.mode}")
        
        # Se RGB, prendi solo canale R (che contiene gli ID)
        if seg_img.mode == 'RGB':
            seg_array = np.array(seg_img)[:, :, 0]  # Solo canale rosso
            print("📍 Usando canale R (contiene segmentation IDs)")
        else:
            seg_array = np.array(seg_img.convert('L'))
            print("📍 Convertito in grayscale")
            
        # Analisi valori
        unique_values = np.unique(seg_array)
        print(f"\n🔍 ANALISI SEGMENTAZIONE:")
        print(f"📊 Valori grigi trovati: {len(unique_values)}")
        print(f"🎯 Range: {unique_values.min()} - {unique_values.max()}")
        print(f"📋 Tutti i valori: {list(unique_values)}")
        
        # Distribuzione pixel
        print(f"\n📈 DISTRIBUZIONE PIXEL:")
        value_counts = []
        for val in unique_values:
            count = np.sum(seg_array == val)
            percentage = (count / seg_array.size) * 100
            value_counts.append((val, count, percentage))
            
        # Ordina per frequenza
        value_counts.sort(key=lambda x: x[1], reverse=True)
        
        for val, count, percentage in value_counts[:15]:  # Top 15
            print(f"  Grigio {val:3d}: {count:7d} pixel ({percentage:5.1f}%)")
            
        # Suggerimenti categorizzazione
        print(f"\n💡 SUGGERIMENTI CATEGORIZZAZIONE:")
        
        # Categorizza automaticamente
        low_vals = [v for v in unique_values if v < 50]
        mid_low_vals = [v for v in unique_values if 50 <= v < 100]
        mid_vals = [v for v in unique_values if 100 <= v < 150]
        high_vals = [v for v in unique_values if v >= 150]
        
        if low_vals:
            print(f"🌌 Cielo (valori bassi 0-49): {low_vals}")
        if mid_low_vals:
            print(f"🌳 Alberi (valori medi-bassi 50-99): {mid_low_vals}")
        if mid_vals:
            print(f"🏢 Edifici (valori medi 100-149): {mid_vals}")
        if high_vals:
            print(f"🌍 Terreno (valori alti 150+): {high_vals}")
            
        # Genera configurazione
        obstacles = low_vals + mid_low_vals + mid_vals  # Tutto tranne cielo e terreno alto
        
        print(f"\n🔧 CONFIGURAZIONE SUGGERITA:")
        print("SEGMENTATION_CATEGORIES = {")
        if low_vals:
            print(f"    'sky': {low_vals},")
        if mid_low_vals:
            print(f"    'trees': {mid_low_vals},")
        if mid_vals:
            print(f"    'buildings': {mid_vals},")
        if high_vals:
            print(f"    'ground': {high_vals},")
        if obstacles:
            print(f"    'obstacles': {obstacles}  # Combinazione alberi + edifici")
        print("}")
        
        # Salva report
        with open("segmentation_analysis.txt", "w") as f:
            f.write(f"Analisi segmentation_debug.png\\n")
            f.write(f"Valori: {list(unique_values)}\\n")
            f.write(f"Distribuzione:\\n")
            for val, count, percentage in value_counts:
                f.write(f"  {val}: {count} pixel ({percentage:.1f}%)\\n")
                
        print(f"\\n💾 Report salvato in segmentation_analysis.txt")
        
    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    analyze_segmentation_debug()
