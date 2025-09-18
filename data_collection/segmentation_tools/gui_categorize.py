#!/usr/bin/env python3
"""
Interfaccia grafica semplice per categorizzare i grigi della segmentation mask
Usa tkinter (incluso in Python) - nessuna dipendenza extra
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
from tkinter import font

class SegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 Categorizzatore Grigi Segmentation")
        self.root.geometry("800x600")
        
        # Valori grigi da categorizzare (aggiorna con i tuoi veri valori)
        self.gray_values = [0, 1, 2, 5, 10, 15, 25, 30, 45, 50, 60, 80, 100, 120, 150, 180, 200, 255]
        
        # Categorie
        self.categories = {
            'sky': [],
            'trees': [],
            'buildings': [],
            'ground': [],
            'unknown': []
        }
        
        # Indice valore corrente
        self.current_index = 0
        
        self.setup_ui()
        self.update_display()
        
    def setup_ui(self):
        """Crea l'interfaccia utente"""
        # Titolo
        title_font = font.Font(family="Arial", size=16, weight="bold")
        title_label = tk.Label(self.root, text="🎨 Categorizzatore Segmentation Mask", 
                              font=title_font, fg="blue")
        title_label.pack(pady=10)
        
        # Frame principale
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Frame sinistra - Visualizzazione grigio corrente
        left_frame = tk.LabelFrame(main_frame, text="📸 Grigio Corrente", font=("Arial", 12, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas per mostrare il grigio
        self.canvas = tk.Canvas(left_frame, width=300, height=200, relief=tk.SUNKEN, bd=2)
        self.canvas.pack(pady=20)
        
        # Informazioni valore corrente
        self.info_frame = tk.Frame(left_frame)
        self.info_frame.pack(pady=10)
        
        self.value_label = tk.Label(self.info_frame, text="Grigio: 0", 
                                   font=("Arial", 14, "bold"))
        self.value_label.pack()
        
        self.progress_label = tk.Label(self.info_frame, text="1 / 18", 
                                      font=("Arial", 10))
        self.progress_label.pack()
        
        # Frame destra - Controlli
        right_frame = tk.LabelFrame(main_frame, text="🏷️ Categorizzazione", font=("Arial", 12, "bold"))
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Bottoni categorie
        categories_frame = tk.Frame(right_frame)
        categories_frame.pack(pady=20)
        
        # Crea bottoni per ogni categoria con colori
        category_colors = {
            'sky': '#87CEEB',      # SkyBlue
            'trees': '#228B22',    # ForestGreen
            'buildings': '#708090', # SlateGray
            'ground': '#8B4513',   # SaddleBrown
            'unknown': '#696969'   # DimGray
        }
        
        self.category_buttons = {}
        for i, (category, color) in enumerate(category_colors.items()):
            btn = tk.Button(categories_frame, text=f"{category.upper()}", 
                           bg=color, fg='white', font=("Arial", 11, "bold"),
                           width=12, height=2,
                           command=lambda c=category: self.categorize(c))
            btn.pack(pady=5)
            self.category_buttons[category] = btn
        
        # Controlli navigazione
        nav_frame = tk.Frame(right_frame)
        nav_frame.pack(pady=20)
        
        tk.Button(nav_frame, text="⬅️ Precedente", width=12,
                 command=self.previous_value).pack(pady=5)
        
        tk.Button(nav_frame, text="➡️ Successivo", width=12,
                 command=self.next_value).pack(pady=5)
        
        tk.Button(nav_frame, text="⏭️ Auto Resto", width=12,
                 command=self.auto_categorize_remaining).pack(pady=5)
        
        # Controlli file
        file_frame = tk.Frame(right_frame)
        file_frame.pack(pady=20)
        
        tk.Button(file_frame, text="📂 Carica Valori", width=12,
                 command=self.load_values).pack(pady=5)
        
        tk.Button(file_frame, text="📋 Riassunto", width=12,
                 command=self.show_summary).pack(pady=5)
        
        tk.Button(file_frame, text="💾 Salva Config", width=12,
                 command=self.save_config).pack(pady=5)
        
        # Area riassunto in basso
        summary_frame = tk.LabelFrame(self.root, text="📊 Riassunto Categorie")
        summary_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def rgb_from_gray(self, gray_value):
        """Converte valore grigio in RGB"""
        return (gray_value, gray_value, gray_value)
    
    def update_display(self):
        """Aggiorna la visualizzazione del grigio corrente"""
        if self.current_index >= len(self.gray_values):
            self.show_completion()
            return
            
        gray_val = self.gray_values[self.current_index]
        
        # Aggiorna canvas
        self.canvas.delete("all")
        
        # Colore grigio
        rgb = self.rgb_from_gray(gray_val)
        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        # Rettangolo grande con il grigio
        self.canvas.create_rectangle(50, 50, 250, 150, fill=hex_color, outline="black", width=2)
        
        # Testo con valore (colore contrastante)
        text_color = "white" if gray_val < 128 else "black"
        self.canvas.create_text(150, 100, text=str(gray_val), 
                               font=("Arial", 24, "bold"), fill=text_color)
        
        # Aggiorna etichette
        self.value_label.config(text=f"Grigio: {gray_val}")
        self.progress_label.config(text=f"{self.current_index + 1} / {len(self.gray_values)}")
        
        # Evidenzia categoria corrente se già assegnata
        self.update_category_buttons(gray_val)
        self.update_summary_display()
        
    def update_category_buttons(self, gray_val):
        """Evidenzia il bottone della categoria corrente"""
        # Reset tutti i bottoni
        for btn in self.category_buttons.values():
            btn.config(relief=tk.RAISED)
        
        # Trova categoria corrente
        current_category = None
        for category, values in self.categories.items():
            if gray_val in values:
                current_category = category
                break
                
        if current_category:
            self.category_buttons[current_category].config(relief=tk.SUNKEN)
    
    def categorize(self, category):
        """Assegna valore corrente a una categoria"""
        if self.current_index >= len(self.gray_values):
            return
            
        gray_val = self.gray_values[self.current_index]
        
        # Rimuovi da altre categorie
        for cat_values in self.categories.values():
            if gray_val in cat_values:
                cat_values.remove(gray_val)
        
        # Aggiungi alla categoria selezionata
        self.categories[category].append(gray_val)
        
        # Aggiorna visualizzazione
        self.update_category_buttons(gray_val)
        self.update_summary_display()
        
        # Auto-avanza al prossimo
        self.root.after(500, self.next_value)  # Ritardo di 500ms
    
    def next_value(self):
        """Vai al valore successivo"""
        if self.current_index < len(self.gray_values) - 1:
            self.current_index += 1
            self.update_display()
    
    def previous_value(self):
        """Vai al valore precedente"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def auto_categorize_remaining(self):
        """Auto-categorizza i valori rimanenti"""
        remaining = []
        for i in range(self.current_index, len(self.gray_values)):
            gray_val = self.gray_values[i]
            
            # Check se già categorizzato
            is_categorized = any(gray_val in values for values in self.categories.values())
            if not is_categorized:
                remaining.append(gray_val)
        
        if not remaining:
            messagebox.showinfo("Info", "Tutti i valori sono già categorizzati!")
            return
        
        response = messagebox.askyesno("Auto-categorizza", 
                                      f"Auto-categorizzare {len(remaining)} valori rimanenti?")
        if response:
            for gray_val in remaining:
                category = self.suggest_category(gray_val)
                self.categories[category].append(gray_val)
            
            self.update_display()
            messagebox.showinfo("Completato", f"Auto-categorizzati {len(remaining)} valori!")
    
    def suggest_category(self, gray_value):
        """Suggerisci categoria automatica"""
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
    
    def update_summary_display(self):
        """Aggiorna l'area riassunto"""
        self.summary_text.delete(1.0, tk.END)
        
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                self.summary_text.insert(tk.END, f"{category.upper()}: {sorted_values}\\n")
        
        # Mostra non categorizzati
        categorized = set()
        for values in self.categories.values():
            categorized.update(values)
        
        uncategorized = [v for v in self.gray_values if v not in categorized]
        if uncategorized:
            self.summary_text.insert(tk.END, f"NON CATEGORIZZATI: {uncategorized}\\n")
    
    def show_summary(self):
        """Mostra finestra riassunto"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title("📋 Riassunto Completo")
        summary_window.geometry("500x400")
        
        text_widget = tk.Text(summary_window, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(summary_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Contenuto
        text_widget.insert(tk.END, "📋 RIASSUNTO CATEGORIZZAZIONE\\n")
        text_widget.insert(tk.END, "=" * 40 + "\\n\\n")
        
        total_categorized = 0
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                total_categorized += len(values)
                text_widget.insert(tk.END, f"🏷️ {category.upper()}: {sorted_values}\\n\\n")
        
        text_widget.insert(tk.END, f"📊 Progresso: {total_categorized}/{len(self.gray_values)}\\n")
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def load_values(self):
        """Carica valori personalizzati"""
        dialog = tk.Toplevel(self.root)
        dialog.title("📂 Carica Valori Grigi")
        dialog.geometry("400x200")
        
        tk.Label(dialog, text="Inserisci valori grigi separati da virgole:", 
                font=("Arial", 10)).pack(pady=10)
        
        entry = tk.Text(dialog, height=3, width=50)
        entry.pack(pady=10)
        entry.insert(tk.END, ",".join(map(str, self.gray_values)))
        
        def apply_values():
            try:
                text = entry.get(1.0, tk.END).strip()
                new_values = [int(x.strip()) for x in text.split(',') if x.strip()]
                new_values = sorted(list(set(new_values)))  # Rimuovi duplicati e ordina
                
                self.gray_values = new_values
                self.current_index = 0
                
                # Reset categorie
                for category in self.categories:
                    self.categories[category] = []
                
                self.update_display()
                dialog.destroy()
                messagebox.showinfo("Successo", f"Caricati {len(new_values)} valori!")
                
            except ValueError:
                messagebox.showerror("Errore", "Formato non valido! Usa: 0,1,25,50,100")
        
        tk.Button(dialog, text="✅ Applica", command=apply_values).pack(pady=10)
        tk.Button(dialog, text="❌ Annulla", command=dialog.destroy).pack()
    
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
        json_file = 'segmentation_gui_config.json'
        with open(json_file, 'w') as f:
            json.dump(clean, f, indent=2)
        
        # Salva Python
        py_file = 'segmentation_gui_config.py'
        with open(py_file, 'w') as f:
            f.write("# CONFIGURAZIONE SEGMENTAZIONE (da GUI)\\n")
            f.write("# Sostituisci in generate.py\\n\\n")
            f.write("SEGMENTATION_CATEGORIES = {\\n")
            for category, values in clean.items():
                f.write(f"    '{category}': {values},\\n")
            f.write("}\\n")
        
        messagebox.showinfo("Salvato", f"Configurazione salvata:\\n• {json_file}\\n• {py_file}")
    
    def show_completion(self):
        """Mostra messaggio completamento"""
        self.canvas.delete("all")
        self.canvas.create_text(150, 100, text="✅ COMPLETATO!", 
                               font=("Arial", 20, "bold"), fill="green")
        
        response = messagebox.askyesno("Completato", 
                                      "Categorizzazione completata!\\nVuoi salvare la configurazione?")
        if response:
            self.save_config()

def main():
    """Avvia l'interfaccia grafica"""
    root = tk.Tk()
    app = SegmentationGUI(root)
    
    # Istruzioni iniziali
    messagebox.showinfo("🎨 Categorizzatore Grigi", 
                       "Benvenuto!\\n\\n" +
                       "• Vedrai ogni grigio nel canvas\\n" +
                       "• Clicca una categoria per assegnarlo\\n" +
                       "• Usa i bottoni per navigare\\n" +
                       "• 'Auto Resto' categorizza automaticamente\\n\\n" +
                       "Iniziamo! 🚀")
    
    root.mainloop()

if __name__ == "__main__":
    main()
