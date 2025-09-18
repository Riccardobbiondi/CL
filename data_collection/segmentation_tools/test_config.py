#!/usr/bin/env python3
"""Test caricamento configurazione"""

import json
import os

def load_pixel_config():
    """Carica configurazione dai file pixel click JSON"""
    # Lista di possibili file di configurazione (in ordine di priorità)
    config_dir = os.path.join("segmentation_tools")
    possible_files = [
        "segmentation_config.json",  # File principale
        "segmentation_pixel_config.json"  # File alternativo
    ]
    
    # Aggiungi anche file con timestamp
    if os.path.exists(config_dir):
        timestamped_files = [f for f in os.listdir(config_dir) 
                           if f.startswith("segmentation_pixel_config_") and f.endswith(".json")]
        # Ordina per timestamp (più recente prima) 
        timestamped_files.sort(reverse=True)
        possible_files.extend([os.path.join(config_dir, f) for f in timestamped_files])
    
    # Prova ogni file finché non ne trova uno valido
    for relative_path in possible_files:
        if not relative_path.startswith(config_dir):
            config_file = os.path.join(config_dir, relative_path)
        else:
            config_file = relative_path
            
        if os.path.exists(config_file):
            print(f"🔧 Tentativo caricamento configurazione da: {config_file}")
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Estrai le categorie dal JSON
                if 'categories' in data:
                    config = data['categories']
                    print(f"✅ Configurazione JSON caricata: {list(config.keys())}")
                    
                    # Stampa info immagine se disponibile
                    if 'image_info' in data:
                        info = data['image_info']
                        print(f"   Immagine analizzata: {info.get('width', '?')}x{info.get('height', '?')}")
                        print(f"   Valori unici: {info.get('unique_values', '?')}")
                    
                    # Stampa statistiche categorie
                    total = data.get('total_categorized', sum(len(values) for values in config.values()))
                    print(f"   Totale categorizzati: {total}")
                    
                    for cat, values in config.items():
                        print(f"   {cat}: {len(values)} valori ({values[:3]}{'...' if len(values) > 3 else ''})")
                    
                    return config
                else:
                    print(f"⚠️ Campo 'categories' non trovato in {config_file}")
                    
            except Exception as e:
                print(f"❌ Errore caricamento {config_file}: {e}")
                continue
    
    print(f"ℹ️ Nessun file di configurazione trovato in {config_dir}")
    return None

if __name__ == "__main__":
    config = load_pixel_config()
    if config:
        print(f"\n🎯 CONFIGURAZIONE FINALE:")
        for categoria, valori in config.items():
            print(f"  {categoria}: {valori}")
    else:
        print("❌ Nessuna configurazione caricata")
