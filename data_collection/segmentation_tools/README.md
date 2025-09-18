# Strumenti di Categorizzazione Segmentazione

Questa cartella contiene vari script per analizzare e categorizzare i valori grigi delle segmentation mask di AirSim.

## File disponibili:

### 🎨 **ascii_categorize.py** ⭐ (RACCOMANDATO)
Script interattivo con visualizzazione ASCII delle tonalità di grigio.
- ✅ Compatibile con tutti i terminali
- 🎨 Mostra le tonalità usando caratteri ASCII
- 👤 Categorizzazione interattiva guidata
- 💾 Genera configurazione finale per generate.py

**Uso:** `python ascii_categorize.py`

### 📊 **analyze_segmentation.py**
Analisi automatica della segmentation_debug.png
- 📈 Statistiche sui valori grigi
- 🔍 Distribuzione pixel
- 💡 Suggerimenti categorizzazione

**Uso:** `python analyze_segmentation.py`

### 🖥️ **visual_categorize.py**
Categorizzazione con colori ANSI (richiede terminale compatibile)
- 🌈 Visualizzazione colori (se supportati dal terminale)
- 👤 Categorizzazione interattiva
- 📱 Potrebbe non funzionare su tutti i terminali

**Uso:** `python visual_categorize.py`

### 🔧 **read_real_pixels.py**
Lettura diretta dei pixel dal file PNG
- 📖 Legge i veri valori pixel dalla segmentation_debug.png
- 🔍 Analisi dettagliata dei valori reali
- ⚙️ Per utenti avanzati

**Uso:** `python read_real_pixels.py`

### ⚡ **quick_categorize.py**
Categorizzazione rapida con valori di esempio
- 🚀 Veloce e semplice
- 🧪 Usa valori tipici di AirSim
- 📝 Buono per test iniziali

**Uso:** `python quick_categorize.py`

### 🏗️ **categorize_simple.py**
Categorizzatore semplificato senza dipendenze esterne
- 📦 Nessuna dipendenza matplotlib/PIL
- 🎯 Funzionalità essenziali
- 🔧 Backup se altri script non funzionano

**Uso:** `python categorize_simple.py`

### 🔬 **categorize_segmentation.py**
Categorizzatore completo con GUI (richiede matplotlib/tkinter)
- 🖼️ Interfaccia grafica (se disponibile)
- 🔍 Anteprima visuale completa
- 📊 Funzionalità avanzate

**Uso:** `python categorize_segmentation.py`

## 🚀 Workflow Raccomandato:

1. **Genera una segmentation mask:**
   ```bash
   cd ..
   python generate.py  # Genera segmentation_debug.png
   ```

2. **Categorizza i grigi:**
   ```bash
   cd segmentation_tools
   python ascii_categorize.py  # Script raccomandato
   ```

3. **Aggiorna generate.py:**
   - Copia `SEGMENTATION_CATEGORIES` dal file generato
   - Incollalo in `../generate.py` (linea ~18)

4. **Testa la configurazione:**
   ```bash
   cd ..
   python generate.py  # Testa con nuova configurazione
   ```

## 📂 File generati:

Gli script possono generare questi file:
- `segmentation_categories.json` - Configurazione in formato JSON
- `segmentation_*_config.py` - Configurazione Python da copiare
- `preview_gray_*.png` - Anteprime delle tonalità (se generate)
- `segmentation_analysis.txt` - Report di analisi

## 🆘 Risoluzione problemi:

- **"PIL not found":** Usa `ascii_categorize.py` o `categorize_simple.py`
- **"matplotlib not found":** Usa `ascii_categorize.py` (raccomandato)
- **Terminale non mostra colori:** Usa `ascii_categorize.py`
- **File PNG non leggibile:** Usa `quick_categorize.py` con valori di esempio

## 💡 Suggerimenti:

- Inizia sempre con `ascii_categorize.py` - è il più compatibile
- Se hai i veri valori dalla tua immagine, aggiornali negli script
- La categoria 'obstacles' viene creata automaticamente (trees + buildings)
- Testa sempre la configurazione finale con `generate.py`
