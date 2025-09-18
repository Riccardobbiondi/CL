#!/usr/bin/env python3
"""
ESEMPIO DI USO: Come categorizzare i grigi della segmentation mask
Script di esempio che mostra il workflow completo
"""

import os
import sys

def main():
    print("🎨 ESEMPIO USO STRUMENTI CATEGORIZZAZIONE")
    print("=" * 55)
    
    # Check se siamo nella directory corretta
    if not os.path.exists("../generate.py"):
        print("❌ Esegui questo script da data_collection/segmentation_tools/")
        return
    
    # Check se esiste segmentation_debug.png
    debug_file = "../segmentation_debug.png"
    if os.path.exists(debug_file):
        size = os.path.getsize(debug_file)
        print(f"✅ Trovato {debug_file} ({size} bytes)")
    else:
        print(f"⚠️ {debug_file} non trovato")
        print("💡 Prima genera una segmentation mask con:")
        print("   cd ..")
        print("   python generate.py")
        print()
    
    print("🚀 WORKFLOW COMPLETO:")
    print()
    
    print("1️⃣ GENERA SEGMENTATION MASK (se non hai):")
    print("   cd ..")
    print("   python generate.py")
    print("   # Questo crea segmentation_debug.png")
    print()
    
    print("2️⃣ CATEGORIZZA I GRIGI (RACCOMANDATO):")
    print("   python ascii_categorize.py")
    print("   # Segui il menu interattivo")
    print("   # Scegli opzione 2 per vedere la palette")
    print("   # Scegli opzione 3 per categorizzare")
    print()
    
    print("3️⃣ ALTERNATIVE (se ascii_categorize.py non funziona):")
    print("   python quick_categorize.py      # Veloce con valori di esempio")
    print("   python categorize_simple.py     # Semplice senza dipendenze")
    print("   python visual_categorize.py     # Con colori (se terminale supporta)")
    print()
    
    print("4️⃣ AGGIORNA GENERATE.PY:")
    print("   # Dopo categorizzazione, vengono generati:")
    print("   #   segmentation_*_config.py")
    print("   #   segmentation_categories.json")
    print("   # Copia SEGMENTATION_CATEGORIES dal file .py")
    print("   # Incollalo in ../generate.py (linea ~18)")
    print()
    
    print("5️⃣ TESTA LA CONFIGURAZIONE:")
    print("   cd ..")
    print("   python generate.py")
    print("   # Controlla cartella debug_masks/ per verificare")
    print()
    
    print("📋 ESEMPIO CONFIGURAZIONE:")
    print("SEGMENTATION_CATEGORIES = {")
    print("    'sky': [0, 1, 2, 5, 10],")
    print("    'trees': [25, 30, 45, 50],")
    print("    'buildings': [60, 80, 100, 120],")
    print("    'ground': [150, 180, 200],")
    print("    'obstacles': [25, 30, 45, 50, 60, 80, 100, 120],")
    print("}")
    print()
    
    print("💡 SUGGERIMENTI:")
    print("• Inizia sempre con ascii_categorize.py (più compatibile)")
    print("• La categoria 'obstacles' = trees + buildings")
    print("• Valori bassi (0-50) = cielo/sfondo")
    print("• Valori medi (50-120) = strutture")
    print("• Valori alti (120+) = terreno")
    print("• Testa sempre la configurazione finale")
    print()
    
    # Suggerimento per prossimo step
    if os.path.exists(debug_file):
        print("✅ PRONTO! Puoi iniziare con:")
        print("   python ascii_categorize.py")
    else:
        print("🔧 PRIMO STEP: genera segmentation_debug.png")
        print("   cd ..")
        print("   python generate.py")
    
    print()
    print("📖 Leggi README.md per dettagli completi su ogni strumento")

if __name__ == "__main__":
    main()
