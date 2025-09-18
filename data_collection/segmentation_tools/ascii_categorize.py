#!/usr/bin/env python3
"""
Script per categorizzare grigi con visualizzazione ASCII (compatibile tutti i terminali)
Usa caratteri ASCII per mostrare le tonalità di grigio
"""

import os
import json

class ASCIICategorizer:
    def __init__(self, image_path="segmentation_debug.png"):
        self.image_path = image_path
        self.categories = {
            'sky': [],
            'trees': [],
            'buildings': [],
            'ground': [],
            'unknown': []
        }
        
        # Valori di esempio - AGGIORNA QUESTI con i tuoi veri valori!
        self.gray_values = [0, 1, 2, 5, 10, 15, 25, 30, 45, 50, 60, 80, 100, 120, 150, 180, 200, 255]
        
        # Caratteri ASCII per rappresentare tonalità (dal più scuro al più chiaro)
        self.ascii_chars = " .:-=+*#%@"
        
    def gray_to_ascii(self, gray_value):
        """Converte valore grigio (0-255) in carattere ASCII rappresentativo"""
        # Normalizza 0-255 -> 0-9 (indici dell'array ascii_chars)
        index = min(9, int((gray_value / 255) * 9))
        char = self.ascii_chars[index]
        
        # Crea blocco visuale con il carattere
        block = char * 6  # Ripeti il carattere 6 volte per visibilità
        
        return f"[{block}] {gray_value:3d}"
    
    def show_ascii_palette(self):
        """Mostra palette completa con caratteri ASCII"""
        print("\\n🎨 PALETTE TONALITÀ ASCII:")
        print("=" * 70)
        print("Legenda: " + " → ".join(f"'{c}'" for c in self.ascii_chars))
        print("(da nero a bianco)")
        print()
        
        # Mostra in righe di 4 per leggibilità
        for i in range(0, len(self.gray_values), 4):
            row = self.gray_values[i:i+4]
            
            for val in row:
                ascii_repr = self.gray_to_ascii(val)
                print(f"   {ascii_repr}")
            
            print()  # Riga vuota tra gruppi
    
    def categorize_interactive(self):
        """Categorizzazione interattiva con visualizzazione ASCII"""
        print("\\n👤 CATEGORIZZAZIONE INTERATTIVA")
        print("🏷️ Categorie: sky, trees, buildings, ground, unknown")
        print("💡 Comandi speciali:")
        print("   'palette' - mostra tutti i colori")
        print("   'summary' - mostra riassunto categorie")
        print("   'auto'    - auto-categorizza i restanti")
        print("   'done'    - termina categorizzazione")
        
        for i, gray_val in enumerate(self.gray_values):
            print(f"\\n" + "="*50)
            print(f"📸 Valore {i+1}/{len(self.gray_values)}")
            
            # Mostra la tonalità corrente in grande
            current_ascii = self.gray_to_ascii(gray_val)
            print(f"\\n🎨 TONALITÀ CORRENTE:")
            print(f"   {current_ascii}")
            print(f"   {current_ascii}")
            print(f"   {current_ascii}")
            
            # Mostra contesto (valori vicini)
            print(f"\\n📍 CONTESTO (valori vicini):")
            context_start = max(0, i-2)
            context_end = min(len(self.gray_values), i+3)
            
            for j in range(context_start, context_end):
                context_val = self.gray_values[j]
                context_ascii = self.gray_to_ascii(context_val)
                
                if j == i:
                    print(f" → {context_ascii} ← QUESTO")
                else:
                    print(f"   {context_ascii}")
            
            # Suggerimento automatico
            suggestion = self.suggest_category(gray_val)
            print(f"\\n💡 Suggerimento automatico: {suggestion}")
            
            # Input utente
            while True:
                prompt = f"\\n🏷️ Categoria per grigio {gray_val} [{suggestion}]: "
                response = input(prompt).strip().lower()
                
                if response == '':
                    response = suggestion
                
                if response == 'palette':
                    self.show_ascii_palette()
                elif response == 'summary':
                    self.print_summary()
                elif response == 'auto':
                    # Auto-categorizza i restanti
                    remaining = self.gray_values[i:]
                    for val in remaining:
                        auto_cat = self.suggest_category(val)
                        self.categories[auto_cat].append(val)
                    print(f"\\n✅ Auto-categorizzati {len(remaining)} valori restanti")
                    return True
                elif response == 'done':
                    return True
                elif response in self.categories:
                    self.categories[response].append(gray_val)
                    print(f"\\n✅ Grigio {gray_val} assegnato a: {response}")
                    break
                else:
                    print(f"\\n❌ Comando non riconosciuto: '{response}'")
                    print(f"   Categorie valide: {list(self.categories.keys())}")
                    print(f"   Comandi: palette, summary, auto, done")
        
        return True
    
    def suggest_category(self, gray_value):
        """Suggerisci categoria basata su valore grigio e logica AirSim"""
        if gray_value <= 10:
            return 'sky'        # Valori molto bassi = cielo/background
        elif 11 <= gray_value <= 50:
            return 'trees'      # Valori medi-bassi = vegetazione
        elif 51 <= gray_value <= 120:
            return 'buildings'  # Valori medi = strutture artificiali
        elif 121 <= gray_value <= 220:
            return 'ground'     # Valori alti = terreno/strade
        else:
            return 'unknown'    # Valori estremi = rumore/altro
    
    def print_summary(self):
        """Stampa riassunto con visualizzazione ASCII"""
        print(f"\\n📋 RIASSUNTO CATEGORIE:")
        print("=" * 60)
        
        total_categorized = 0
        
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                total_categorized += len(values)
                
                print(f"\\n🏷️ {category.upper()} ({len(values)} valori):")
                print(f"   Valori numerici: {sorted_values}")
                print(f"   Visualizzazione:")
                
                for val in sorted_values:
                    ascii_repr = self.gray_to_ascii(val)
                    print(f"     {ascii_repr}")
        
        # Mostra non categorizzati
        categorized_set = set()
        for values in self.categories.values():
            categorized_set.update(values)
        
        uncategorized = [val for val in self.gray_values if val not in categorized_set]
        if uncategorized:
            print(f"\\n⚠️ NON CATEGORIZZATI ({len(uncategorized)} valori):")
            print(f"   Valori: {uncategorized}")
            print(f"   Visualizzazione:")
            for val in uncategorized:
                ascii_repr = self.gray_to_ascii(val)
                print(f"     {ascii_repr}")
        
        print(f"\\n📊 Progresso: {total_categorized}/{len(self.gray_values)} categorizzati")
    
    def save_config(self):
        """Salva configurazione finale"""
        # Pulizia e ordinamento
        clean = {k: sorted(v) for k, v in self.categories.items() if v}
        
        # Aggiungi categoria obstacles (combinazione alberi + edifici)
        if clean.get('trees') or clean.get('buildings'):
            obstacles = []
            obstacles.extend(clean.get('trees', []))
            obstacles.extend(clean.get('buildings', []))
            clean['obstacles'] = sorted(obstacles)
        
        # Salva JSON
        json_file = 'segmentation_ascii_config.json'
        with open(json_file, 'w') as f:
            json.dump(clean, f, indent=2)
        
        # Salva configurazione Python
        py_file = 'segmentation_ascii_config.py'
        with open(py_file, 'w') as f:
            f.write("# ===== CONFIGURAZIONE SEGMENTAZIONE FINALE =====\\n")
            f.write("# Generata tramite categorizzazione ASCII visuale\\n")
            f.write("# Sostituisci SEGMENTATION_CATEGORIES in generate.py\\n\\n")
            f.write("SEGMENTATION_CATEGORIES = {\\n")
            for category, values in clean.items():
                f.write(f"    '{category}': {values},\\n")
            f.write("}\\n\\n")
            f.write("# ISTRUZIONI:\\n")
            f.write("# 1. Copia tutto il dizionario SEGMENTATION_CATEGORIES\\n")
            f.write("# 2. Incollalo in generate.py (linea ~18)\\n")
            f.write("# 3. Esegui: python generate.py\\n")
            f.write("# 4. Controlla cartella debug_masks/ per verificare risultati\\n")
        
        print(f"\\n💾 CONFIGURAZIONE SALVATA:")
        print(f"   📄 {json_file}")
        print(f"   🐍 {py_file}")
        
        # Mostra configurazione finale
        print(f"\\n📋 CONFIGURAZIONE FINALE:")
        for category, values in clean.items():
            print(f"   {category:>12}: {values}")
        
        print(f"\\n🔧 PROSSIMI PASSI:")
        print(f"   1. Apri {py_file}")
        print(f"   2. Copia SEGMENTATION_CATEGORIES")
        print(f"   3. Incollalo in generate.py")
        print(f"   4. Testa con: python generate.py")
        
        return clean
    
    def update_real_values(self):
        """Permette all'utente di aggiornare i valori grigi reali"""
        print(f"\\n🔧 AGGIORNA VALORI REALI")
        print(f"📋 Valori correnti: {self.gray_values}")
        print(f"\\n💡 Se hai i VERI valori dalla tua segmentation_debug.png,")
        print(f"   inseriscili ora separati da virgole (es: 0,1,25,50,100,255)")
        print(f"   Altrimenti premi ENTER per usare quelli di esempio")
        
        user_input = input("\\nValori grigi reali: ").strip()
        
        if user_input:
            try:
                new_values = [int(x.strip()) for x in user_input.split(',')]
                new_values = sorted(list(set(new_values)))  # Rimuovi duplicati e ordina
                
                print(f"✅ Aggiornati a: {new_values}")
                self.gray_values = new_values
                return True
                
            except ValueError:
                print(f"❌ Formato non valido. Usando valori di esempio.")
                return False
        else:
            print(f"📝 Usando valori di esempio")
            return False

def main():
    """Menu principale"""
    print("🎨 CATEGORIZZATORE ASCII SEGMENTATION MASK")
    print("=" * 60)
    print("✨ Versione compatibile con tutti i terminali!")
    
    # Inizializza
    categorizer = ASCIICategorizer()
    
    # Check file
    if os.path.exists(categorizer.image_path):
        size = os.path.getsize(categorizer.image_path)
        print(f"📁 Trovato {categorizer.image_path} ({size} bytes)")
    else:
        print(f"⚠️ {categorizer.image_path} non trovato")
    
    # Menu principale
    while True:
        print(f"\\n🔧 MENU PRINCIPALE:")
        print("1. Aggiorna valori grigi reali")
        print("2. Mostra palette ASCII completa")
        print("3. Categorizzazione interattiva")
        print("4. Mostra riassunto categorie")
        print("5. Salva configurazione")
        print("6. Esci")
        
        choice = input("\\nScelta [1-6]: ").strip()
        
        if choice == '1':
            categorizer.update_real_values()
        elif choice == '2':
            categorizer.show_ascii_palette()
        elif choice == '3':
            categorizer.categorize_interactive()
        elif choice == '4':
            categorizer.print_summary()
        elif choice == '5':
            categorizer.save_config()
        elif choice == '6':
            break
        else:
            print("❌ Scelta non valida")
    
    # Salvataggio finale
    if any(categorizer.categories.values()):
        save = input("\\n💾 Salvare la configurazione prima di uscire? [y/n]: ").lower().strip()
        if save == 'y':
            categorizer.save_config()
    
    print("\\n👋 Categorizzazione completata!")
    print("🔧 Ricorda di aggiornare generate.py con la nuova configurazione!")

if __name__ == "__main__":
    main()
