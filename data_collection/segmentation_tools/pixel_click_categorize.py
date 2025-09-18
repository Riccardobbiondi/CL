#!/usr/bin/env python3
"""
Interfaccia grafica per categorizzare grigi cliccando direttamente sui pixel
Carica segmentation_debug.png e permette di cliccare sui pixel per categorizzarli
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import json
from PIL import Image, ImageTk
import numpy as np

class PixelClickCategorizer:
    def __init__(self, root):
        self.root = root
        self.root.title("🖱️ Categorizzatore Pixel Click")
        self.root.geometry("1400x900")  # Finestra più grande
        
        # Dati immagine
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.pixel_array = None
        self.scale_factor = 1.0
        
        # Categorie e colori
        self.categories = {
            'sky': [],
            'trees': [],
            'buildings': [],
            'ground': [],
            'unknown': []
        }
        
        # Colori per overlay delle categorie
        self.category_colors = {
            'sky': (135, 206, 235),      # SkyBlue
            'trees': (34, 139, 34),      # ForestGreen
            'buildings': (128, 128, 128), # Gray
            'ground': (139, 69, 19),     # SaddleBrown
            'unknown': (255, 0, 255)     # Magenta
        }
        
        self.selected_category = 'sky'
        self.clicked_pixels = {}  # {gray_value: category}
        self.show_categorized = False  # Flag per visualizzare solo categorizzati
        self.categorized_overlay = None  # Immagine overlay
        
        # Carica configurazione esistente
        self.load_existing_config()
        
        self.setup_ui()
        self.load_default_image()
        
    def load_existing_config(self):
        """Carica configurazione esistente dal JSON se presente"""
        json_file = 'segmentation_config.json'
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    config = json.load(f)
                
                # Carica categorie esistenti
                existing_categories = config.get('categories', {})
                for category, values in existing_categories.items():
                    if category in self.categories:
                        self.categories[category] = values.copy()
                        # Popola anche clicked_pixels per retrocompatibilità
                        for value in values:
                            self.clicked_pixels[value] = category
                
                print(f"📂 Configurazione caricata da {json_file}")
                total_loaded = sum(len(values) for values in self.categories.values())
                print(f"✅ Caricati {total_loaded} valori categorizzati")
                
                # Mostra dettagli caricati
                for category, values in self.categories.items():
                    if values:
                        print(f"   {category.upper()}: {len(values)} valori")
                        
            except Exception as e:
                print(f"⚠️ Errore caricamento configurazione: {e}")
                messagebox.showwarning("Attenzione", 
                                     f"Errore caricamento configurazione:\n{e}\n\nPartendo da configurazione vuota.")
        else:
            print(f"📝 File {json_file} non trovato - partendo da configurazione vuota")
    
    def get_pixel_category(self, gray_value):
        """Restituisce la categoria di un valore grigio se già categorizzato"""
        return self.clicked_pixels.get(int(gray_value))
    
    def setup_ui(self):
        """Crea l'interfaccia utente"""
        # Titolo
        title_label = tk.Label(self.root, text="🖱️ Clicca sui Pixel per Categorizzare", 
                              font=("Arial", 16, "bold"), fg="blue")
        title_label.pack(pady=5)
        
        # Frame principale
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Frame sinistra - Immagine
        left_frame = tk.LabelFrame(main_frame, text="📸 Segmentation Image", font=("Arial", 10, "bold"))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Canvas per immagine con scrollbar - MOLTO PIÙ GRANDE
        canvas_frame = tk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white', cursor='crosshair', 
                               width=900, height=600)  # Canvas molto più grande
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_pixel_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Frame destra - Controlli
        right_frame = tk.LabelFrame(main_frame, text="🏷️ Controlli", font=("Arial", 10, "bold"))
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.configure(width=250)
        
        # Selezione categoria
        category_frame = tk.LabelFrame(right_frame, text="Categoria Attiva")
        category_frame.pack(fill=tk.X, pady=5)
        
        self.category_var = tk.StringVar(value='sky')
        for category in self.categories.keys():
            count = len(self.categories[category])
            label_text = f"{category.upper()} ({count})" if count > 0 else category.upper()
            
            rb = tk.Radiobutton(category_frame, text=label_text, 
                               variable=self.category_var, value=category,
                               command=self.update_selected_category)
            rb.pack(anchor=tk.W, padx=5)
        
        # Info pixel corrente
        info_frame = tk.LabelFrame(right_frame, text="📍 Pixel Info")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.pixel_info = tk.Label(info_frame, text="Clicca su un pixel", 
                                  font=("Courier", 9), justify=tk.LEFT, wraplength=200)
        self.pixel_info.pack(padx=5, pady=5)
        
        # Controlli immagine
        img_controls = tk.LabelFrame(right_frame, text="🖼️ Immagine")
        img_controls.pack(fill=tk.X, pady=5)
        
        tk.Button(img_controls, text="📂 Carica Immagine", width=20,
                 command=self.load_image).pack(pady=2)
        
        # Pulsante per mostrare solo categorizzati
        self.show_cat_var = tk.BooleanVar()
        self.show_cat_button = tk.Checkbutton(img_controls, text="🔴 Mostra Solo Categorizzati", 
                                             variable=self.show_cat_var, width=20,
                                             command=self.toggle_categorized_view)
        self.show_cat_button.pack(pady=2)
        
        tk.Button(img_controls, text="🔄 Reset Zoom", width=20,
                 command=self.reset_zoom).pack(pady=2)
        
        tk.Button(img_controls, text="🧹 Pulisci Selezioni", width=20,
                 command=self.clear_selections).pack(pady=2)
        
        # Azioni
        actions_frame = tk.LabelFrame(right_frame, text="💾 Azioni")
        actions_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(actions_frame, text="🔄 Ricarica Config", width=20,
                 command=self.reload_config).pack(pady=2)
        
        tk.Button(actions_frame, text="📋 Mostra Riassunto", width=20,
                 command=self.show_summary).pack(pady=2)
        
        tk.Button(actions_frame, text="💾 Salva Configurazione", width=20,
                 command=self.save_config).pack(pady=2)
        
        tk.Button(actions_frame, text="📤 Esporta Immagine", width=20,
                 command=self.export_categorized_image).pack(pady=2)
        
        # Status bar
        total_loaded = sum(len(values) for values in self.categories.values())
        initial_status = f"Pronto - {total_loaded} valori già categorizzati caricati"
        self.status_var = tk.StringVar(value=initial_status)
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Riassunto categorie in basso
        summary_frame = tk.LabelFrame(self.root, text="📊 Categorie")
        summary_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=4, wrap=tk.WORD, font=("Courier", 8))
        summary_scroll = tk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Aggiorna riassunto iniziale
        self.update_summary()
        
    def reload_config(self):
        """Ricarica configurazione dal file JSON"""
        response = messagebox.askyesno("Ricarica Configurazione", 
                                     "Ricaricare la configurazione dal file JSON?\n"
                                     "Le modifiche non salvate saranno perse!")
        if response:
            # Reset categorie
            for category in self.categories:
                self.categories[category] = []
            self.clicked_pixels = {}
            
            # Ricarica dal file
            self.load_existing_config()
            
            # Aggiorna UI
            self.update_category_labels()
            self.update_summary()
            self.update_display()
            
            total_loaded = sum(len(values) for values in self.categories.values())
            self.status_var.set(f"✅ Ricaricati {total_loaded} valori dal JSON")
    
    def update_category_labels(self):
        """Aggiorna le etichette delle categorie con i conteggi"""
        # Trova tutti i radiobutton e aggiorna le etichette
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.LabelFrame) and "Controlli" in child.cget("text"):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, tk.LabelFrame) and "Categoria Attiva" in grandchild.cget("text"):
                                # Ricostruisci i radiobutton
                                for rb in grandchild.winfo_children():
                                    rb.destroy()
                                
                                for category in self.categories.keys():
                                    count = len(self.categories[category])
                                    label_text = f"{category.upper()} ({count})" if count > 0 else category.upper()
                                    
                                    rb = tk.Radiobutton(grandchild, text=label_text, 
                                                       variable=self.category_var, value=category,
                                                       command=self.update_selected_category)
                                    rb.pack(anchor=tk.W, padx=5)
                                break
                        break
                break
        
    def load_default_image(self):
        """Carica segmentation_debug.png di default"""
        default_path = "../segmentation_debug.png"
        if os.path.exists(default_path):
            self.load_image_from_path(default_path)
        else:
            self.status_var.set("segmentation_debug.png non trovato - usa 'Carica Immagine'")
    
    def load_image(self):
        """Carica immagine da file"""
        file_path = filedialog.askopenfilename(
            title="Seleziona Segmentation Image",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            initialdir=".."
        )
        
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_image_from_path(self, file_path):
        """Carica immagine dal percorso specificato"""
        try:
            # Carica immagine
            self.original_image = Image.open(file_path)
            
            # Se RGB, prendi solo canale R (per segmentation mask)
            if self.original_image.mode == 'RGB':
                r, g, b = self.original_image.split()
                self.original_image = r  # Solo canale rosso
            elif self.original_image.mode != 'L':
                self.original_image = self.original_image.convert('L')
            
            # Converti in array numpy
            self.pixel_array = np.array(self.original_image)
            
            # Scala l'immagine per renderla PIÙ GRANDE (no thumbnail)
            # Scala di almeno 2x se l'immagine è piccola
            min_display_size = 800
            if (self.original_image.width < min_display_size or 
                self.original_image.height < min_display_size):
                
                scale_factor = max(2.0, min_display_size / max(self.original_image.width, self.original_image.height))
                new_size = (int(self.original_image.width * scale_factor), 
                           int(self.original_image.height * scale_factor))
                
                self.display_image = self.original_image.resize(new_size, Image.Resampling.NEAREST)
                self.scale_factor = scale_factor
            else:
                self.display_image = self.original_image.copy()
                self.scale_factor = 1.0
            
            # Crea overlay per categorizzati
            self.update_display()
            
            # Analizza valori unici
            unique_values = np.unique(self.pixel_array)
            total_categorized = sum(len(values) for values in self.categories.values())
            
            self.status_var.set(f"Caricato: {os.path.basename(file_path)} "
                              f"({self.original_image.width}x{self.original_image.height}) "
                              f"→ display ({self.display_image.width}x{self.display_image.height}) "
                              f"- {len(unique_values)} grigi unici - {total_categorized} già categorizzati")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile caricare immagine:\n{e}")
            
    def update_display(self):
        """Aggiorna la visualizzazione dell'immagine con eventuali overlay"""
        if self.display_image is None:
            return
            
        # Copia dell'immagine base
        display_img = self.display_image.copy()
        
        # Se show_categorized è attivo, crea overlay colorato per pixel categorizzati
        if self.show_categorized and self.clicked_pixels:
            # Converti in RGB per overlay colorato
            if display_img.mode != 'RGB':
                display_img = display_img.convert('RGB')
            
            # Crea array per overlay
            img_array = np.array(display_img)
            
            # Evidenzia tutti i pixel categorizzati nell'immagine originale
            for y in range(self.pixel_array.shape[0]):
                for x in range(self.pixel_array.shape[1]):
                    gray_value = self.pixel_array[y, x]
                    category = self.get_pixel_category(gray_value)
                    
                    if category:
                        # Converti coordinate originali in coordinate display
                        display_x = int(x * self.scale_factor)
                        display_y = int(y * self.scale_factor)
                        
                        # Colore della categoria
                        color = self.category_colors[category]
                        
                        # Applica colore (con area per visibilità)
                        for dx in range(max(1, int(self.scale_factor))):
                            for dy in range(max(1, int(self.scale_factor))):
                                px = display_x + dx
                                py = display_y + dy
                                
                                if (0 <= px < img_array.shape[1] and 
                                    0 <= py < img_array.shape[0]):
                                    img_array[py, px] = color
            
            display_img = Image.fromarray(img_array)
        
        # Converti per tkinter
        self.photo = ImageTk.PhotoImage(display_img)
        
        # Mostra su canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def toggle_categorized_view(self):
        """Toggle per mostrare/nascondere i pixel categorizzati"""
        self.show_categorized = not self.show_categorized
        self.update_display()
        
        status = "attivo" if self.show_categorized else "disattivo"
        print(f"Visualizzazione categorizzati: {status}")
    
    def on_pixel_click(self, event):
        """Gestisce click su pixel"""
        if self.pixel_array is None:
            return
        
        # Converti coordinate canvas in coordinate immagine
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Scala alle coordinate originali
        orig_x = int(canvas_x / self.scale_factor)
        orig_y = int(canvas_y / self.scale_factor)
        
        # Check bounds
        if (0 <= orig_x < self.pixel_array.shape[1] and 
            0 <= orig_y < self.pixel_array.shape[0]):
            
            # Ottieni valore grigio
            gray_value = self.pixel_array[orig_y, orig_x]
            
            # Verifica se già categorizzato
            existing_category = self.get_pixel_category(gray_value)
            new_category = self.category_var.get()
            
            if existing_category:
                if existing_category == new_category:
                    # Stesso colore, stessa categoria - non fare nulla
                    self.status_var.set(f"Grigio {gray_value} già in {existing_category}")
                    return
                else:
                    # Rimuovi da categoria precedente
                    if int(gray_value) in self.categories[existing_category]:
                        self.categories[existing_category].remove(int(gray_value))
                    
                    print(f"🔄 Grigio {gray_value}: {existing_category} → {new_category}")
            else:
                print(f"➕ Grigio {gray_value}: nuovo → {new_category}")
            
            # Categorizza nella nuova categoria
            self.clicked_pixels[int(gray_value)] = new_category
            
            # Aggiungi alla categoria corrente se non presente
            if int(gray_value) not in self.categories[new_category]:
                self.categories[new_category].append(int(gray_value))
            
            # Aggiorna info con stato precedente
            status_text = f"✅ CATEGORIZZATO" if not existing_category else f"🔄 CAMBIATO"
            prev_text = f"\nPrecedente: {existing_category}" if existing_category else ""
            
            self.pixel_info.config(text=f"Pixel: ({orig_x}, {orig_y})\n"
                                       f"Grigio: {gray_value}\n"
                                       f"Categoria: {new_category}\n"
                                       f"Stato: {status_text}{prev_text}")
            
            # Visualizza feedback
            self.show_click_feedback(canvas_x, canvas_y, new_category)
            
            # Aggiorna conteggi e riassunto
            self.update_category_labels()
            self.update_summary()
            
            # Aggiorna display se overlay attivo
            if self.show_categorized:
                self.update_display()
            
            change_type = "aggiornato" if existing_category else "nuovo"
            self.status_var.set(f"Grigio {gray_value} → {new_category} ({change_type})")
    
    def on_mouse_move(self, event):
        """Mostra info pixel sotto il mouse"""
        if self.pixel_array is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        orig_x = int(canvas_x / self.scale_factor)
        orig_y = int(canvas_y / self.scale_factor)
        
        if (0 <= orig_x < self.pixel_array.shape[1] and 
            0 <= orig_y < self.pixel_array.shape[0]):
            
            gray_value = self.pixel_array[orig_y, orig_x]
            
            # Check se già categorizzato
            existing_category = self.get_pixel_category(gray_value)
            
            if existing_category:
                status_text = f"📌 GIÀ CATEGORIZZATO\nCategoria: {existing_category.upper()}"
                color_info = f"\nColore: {self.category_colors[existing_category]}"
            else:
                status_text = "⚪ NON CATEGORIZZATO"
                color_info = ""
            
            self.pixel_info.config(text=f"Mouse: ({orig_x}, {orig_y})\n"
                                       f"Grigio: {gray_value}\n"
                                       f"{status_text}{color_info}")
    
    def show_click_feedback(self, x, y, category):
        """Mostra feedback visuale del click"""
        color = "#{:02x}{:02x}{:02x}".format(*self.category_colors[category])
        
        # Cerchio colorato temporaneo
        circle = self.canvas.create_oval(x-8, y-8, x+8, y+8, 
                                        fill=color, outline="white", width=3)
        
        # Rimuovi dopo 1.5 secondi
        self.root.after(1500, lambda: self.canvas.delete(circle))
    
    def update_selected_category(self):
        """Aggiorna categoria selezionata"""
        self.selected_category = self.category_var.get()
        color = "#{:02x}{:02x}{:02x}".format(*self.category_colors[self.selected_category])
        count = len(self.categories[self.selected_category])
        self.status_var.set(f"Categoria attiva: {self.selected_category.upper()} {color} ({count} valori)")
    
    def update_summary(self):
        """Aggiorna riassunto categorie"""
        self.summary_text.delete(1.0, tk.END)
        
        total = 0
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                total += len(values)
                self.summary_text.insert(tk.END, f"{category.upper()}: {sorted_values}\n")
        
        self.summary_text.insert(tk.END, f"\nTotale categorizzati: {total}")
    
    def clear_selections(self):
        """Pulisce tutte le selezioni"""
        response = messagebox.askyesno("Conferma", "Cancellare tutte le categorizzazioni?\n"
                                                  "Questo cancellerà anche i dati caricati dal JSON!")
        if response:
            self.clicked_pixels = {}
            for category in self.categories:
                self.categories[category] = []
            self.update_category_labels()
            self.update_summary()
            self.update_display()
            self.status_var.set("Selezioni cancellate")
    
    def reset_zoom(self):
        """Reset zoom immagine"""
        if self.original_image:
            # Ricarica l'immagine mantenendo le categorizzazioni
            file_path = getattr(self.original_image, 'filename', '../segmentation_debug.png')
            self.load_image_from_path(file_path)
    
    def show_summary(self):
        """Mostra finestra riassunto dettagliato"""
        summary_window = tk.Toplevel(self.root)
        summary_window.title("📋 Riassunto Dettagliato")
        summary_window.geometry("600x400")
        
        text_widget = tk.Text(summary_window, wrap=tk.WORD, font=("Courier", 10))
        scrollbar = tk.Scrollbar(summary_window, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Contenuto dettagliato
        text_widget.insert(tk.END, "📋 RIASSUNTO CATEGORIZZAZIONE PIXEL\n")
        text_widget.insert(tk.END, "=" * 50 + "\n\n")
        
        total_categorized = 0
        for category, values in self.categories.items():
            if values:
                sorted_values = sorted(values)
                total_categorized += len(values)
                color = self.category_colors[category]
                
                text_widget.insert(tk.END, f"🏷️ {category.upper()} (RGB: {color}):\n")
                text_widget.insert(tk.END, f"   Valori: {sorted_values}\n")
                text_widget.insert(tk.END, f"   Conta: {len(values)}\n\n")
        
        if self.pixel_array is not None:
            unique_total = len(np.unique(self.pixel_array))
            text_widget.insert(tk.END, f"📊 STATISTICHE:\n")
            text_widget.insert(tk.END, f"   Categorizzati: {total_categorized}\n")
            text_widget.insert(tk.END, f"   Grigi unici totali: {unique_total}\n")
            text_widget.insert(tk.END, f"   Progresso: {total_categorized}/{unique_total} ({total_categorized/unique_total*100:.1f}%)\n")
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def save_config(self):
        """Salva configurazione aggiornata"""
        if not any(self.categories.values()):
            messagebox.showwarning("Attenzione", "Nessuna categorizzazione da salvare!")
            return
        
        # Pulizia categorie correnti
        clean = {k: sorted(v) for k, v in self.categories.items() if v}
        
        # Aggiungi obstacles se ci sono trees o buildings
        if clean.get('trees') or clean.get('buildings'):
            obstacles = []
            obstacles.extend(clean.get('trees', []))
            obstacles.extend(clean.get('buildings', []))
            clean['obstacles'] = sorted(list(set(obstacles)))  # Rimuovi duplicati
        
        # Crea timestamp per questa sessione
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepara dati aggiornati
        updated_data = {
            'last_updated': timestamp,
            'image_info': {
                'width': self.original_image.width if self.original_image else 0,
                'height': self.original_image.height if self.original_image else 0,
                'unique_values': len(np.unique(self.pixel_array)) if self.pixel_array is not None else 0
            },
            'categories': clean,
            'total_categorized': sum(len(values) for values in clean.values())
        }
        
        # Salva JSON aggiornato
        json_file = 'segmentation_config.json'
        with open(json_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        # Salva anche file Python aggiornato
        py_file = 'segmentation_pixel_config.py'
        with open(py_file, 'w') as f:
            f.write(f"# CONFIGURAZIONE SEGMENTAZIONE (da Pixel Click)\n")
            f.write(f"# Ultimo aggiornamento: {timestamp}\n")
            f.write(f"# Sostituisci SEGMENTATION_CATEGORIES in generate.py\n\n")
            f.write("SEGMENTATION_CATEGORIES = {\n")
            for category, values in clean.items():
                f.write(f"    '{category}': {values},\n")
            f.write("}\n\n")
            f.write("# ISTRUZIONI:\n")
            f.write("# 1. Copia tutto SEGMENTATION_CATEGORIES\n")
            f.write("# 2. Incolla in ../generate.py (linea ~18)\n")
            f.write("# 3. Testa: cd .. && python generate.py\n")
        
        total_final = updated_data['total_categorized']
        
        messagebox.showinfo("✅ Configurazione Salvata", 
                           f"Configurazione salvata con successo!\n\n"
                           f"📊 STATISTICHE:\n"
                           f"• Totale valori: {total_final}\n\n"
                           f"💾 FILE SALVATI:\n"
                           f"• {json_file} (principale)\n"
                           f"• {py_file} (per generate.py)!")
        
        self.status_var.set(f"✅ Configurazione salvata - {total_final} valori")
    
    def export_categorized_image(self):
        """Esporta immagine con overlay categorie"""
        if self.original_image is None:
            messagebox.showwarning("Attenzione", "Nessuna immagine caricata!")
            return
        
        # Crea immagine RGB per overlay
        rgb_image = Image.new('RGB', self.original_image.size)
        
        # Copia immagine originale in RGB
        for y in range(self.original_image.height):
            for x in range(self.original_image.width):
                gray_val = self.pixel_array[y, x]
                
                # Check se categorizzato
                category = self.get_pixel_category(gray_val)
                if category:
                    # Usa colore categoria
                    color = self.category_colors[category]
                else:
                    # Usa grigio originale
                    color = (gray_val, gray_val, gray_val)
                
                rgb_image.putpixel((x, y), color)
        
        # Salva
        output_file = 'categorized_overlay.png'
        rgb_image.save(output_file)
        
        categorized_count = len(self.clicked_pixels)
        messagebox.showinfo("Esportato", f"Immagine categorizzata salvata: {output_file}\n"
                                        f"Pixel categorizzati evidenziati: {categorized_count}")

def main():
    """Avvia l'interfaccia pixel click"""
    try:
        root = tk.Tk()
        app = PixelClickCategorizer(root)
        
        # Messaggio iniziale
        total_loaded = sum(len(values) for values in app.categories.values())
        start_msg = f"🖱️ Categorizzatore Pixel Click\n\n"
        
        if total_loaded > 0:
            start_msg += f"✅ Caricati {total_loaded} valori dalla configurazione esistente!\n\n"
        
        start_msg += ("ISTRUZIONI:\n\n" +
                     "1. Seleziona una categoria (radio button)\n" +
                     "2. Clicca sui pixel nell'immagine\n" +
                     "3. Vedi se il pixel è già categorizzato\n" +
                     "4. I nuovi valori vengono aggiunti/aggiornati\n" +
                     "5. Salva configurazione quando finito\n\n" +
                     "💡 I pixel già categorizzati mostrano la categoria esistente!")
        
        messagebox.showinfo("🖱️ Categorizzatore Pixel Click", start_msg)
        
        root.mainloop()
        
    except ImportError as e:
        if "PIL" in str(e):
            messagebox.showerror("Errore", "PIL/Pillow non installato!\nInstalla con: pip install Pillow")
        else:
            messagebox.showerror("Errore", f"Dipendenza mancante: {e}")
    except Exception as e:
        messagebox.showerror("Errore", f"Errore avvio: {e}")

if __name__ == "__main__":
    main()
