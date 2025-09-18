#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script per la generazione del dataset usando AirSim
Genera cartelle anchor_XXXXX con immagine anchor e 6 positivi
"""

import airsim
import os
import sys
import time
import random
import numpy as np
import json
import struct
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO

# Parametri configurazione
IMG_SIZE = (224, 224)
DATASET_VERSION = "v4"  # Cambia questo per creare una nuova versione
N_SAMPLES = 2500  # Numero di anchor da generare
CAPTURE_INTERVAL = 2  # Secondi tra una cattura e l'altra
MIN_ALTITUDE = -8  # Altitudine minima del drone
MAX_ALTITUDE = 0  # Altitudine massima del drone

# Configurazione segmentazione - VALORI GRIGI DA CATEGORIZZARE
# Carica automaticamente dal file pixel click se disponibile
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
    
    print(f"ℹ️ Nessun file di configurazione pixel click trovato in {config_dir}")
    print("   Usando configurazione di default")
    
    # Configurazione di default se non trovata
    return {
        'sky': [0, 1, 2, 3, 4, 5],  # Grigi molto scuri = cielo
        'trees': [40, 41, 42, 43, 44, 45, 50, 51, 52],  # Grigi medi-scuri = alberi
        'buildings': [80, 81, 82, 83, 84, 85, 90, 91, 92],  # Grigi medi = edifici
        'ground': [120, 121, 122, 123, 124, 125],  # Grigi medi-chiari = terreno
        'obstacles': [40, 41, 42, 43, 44, 45, 50, 51, 52, 80, 81, 82, 83, 84, 85, 90, 91, 92]  # Combinazione alberi + edifici
    }

# Carica configurazione dinamicamente
SEGMENTATION_CATEGORIES = load_pixel_config()

# Path configurazione
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Cartella principale del progetto
DATASET_DIR = os.path.join(BASE_DIR, f"dataset_{DATASET_VERSION}")
BACKGROUNDS_DIR = os.path.join(BASE_DIR, "backgrounds")

class DatasetGenerator:
    def __init__(self):
        """Inizializza il generatore del dataset"""
        self.client = None
        self.setup_directories()
        
    def setup_directories(self):
        """Crea la directory del dataset se non exists"""
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"📁 Directory dataset: {DATASET_DIR}")
        
    def connect_airsim(self):
        """Connette al simulatore AirSim"""
        try:
            print("🔌 Connessione ad AirSim...")
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            print("✅ Connesso ad AirSim")
            return True
        except Exception as e:
            print(f"❌ Errore connessione AirSim: {e}")
            return False
    
    def takeoff_and_setup(self):
        """Fa decollare il drone e lo posiziona"""
        print("🚁 Decollo del drone...")
        self.client.takeoffAsync().join()
        time.sleep(2)
        
        # Posiziona il drone ad un'altitudine casuale nel range
        target_alt = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        self.client.moveToZAsync(target_alt, 2).join()
        print(f"✅ Drone posizionato ad altitudine: {target_alt:.2f}m")
    
    def analyze_segmentation_values(self, seg_mask):
        """Analizza i valori grigi della segmentation mask in tempo reale"""
        try:
            if seg_mask is None:
                return
                
            unique_values = np.unique(seg_mask)
            print(f"  📊 Valori grigi nella mask: {list(unique_values)}")
            
            # Conta distribuzione
            for val in unique_values[:8]:  # Mostra solo i primi 8
                count = np.sum(seg_mask == val)
                percentage = (count / seg_mask.size) * 100
                print(f"  Grigio {val:3d}: {percentage:5.1f}% ({count} pixel)")
                
            # Suggerisci categorizzazione automatica
            print(f"  💡 Suggerimento: aggiungi ai SEGMENTATION_CATEGORIES:")
            if len(unique_values) > 0:
                low_vals = unique_values[unique_values < 50]
                mid_vals = unique_values[(unique_values >= 50) & (unique_values < 150)]
                high_vals = unique_values[unique_values >= 150]
                
                if len(low_vals) > 0:
                    print(f"    'sky': {list(low_vals)} (valori bassi)")
                if len(mid_vals) > 0:
                    print(f"    'obstacles': {list(mid_vals)} (valori medi)")
                if len(high_vals) > 0:
                    print(f"    'ground': {list(high_vals)} (valori alti)")
                    
        except Exception as e:
            print(f"  ❌ Errore analisi: {e}")

    def analyze_segmentation_mask(self, mask_path="segmentation_debug.png"):
        """Analizza una maschera di segmentazione esistente per scoprire i valori grigi"""
        if not os.path.exists(mask_path):
            print(f"⚠️ File {mask_path} non trovato")
            return None
            
        try:
            # Carica l'immagine di segmentazione
            seg_img = Image.open(mask_path).convert('L')  # Converte in grayscale
            seg_array = np.array(seg_img)
            
            # Trova tutti i valori unici
            unique_values = np.unique(seg_array)
            
            print(f"🔍 Analisi segmentation mask {mask_path}:")
            print(f"📊 Valori grigi trovati: {len(unique_values)}")
            print(f"🎯 Range: {unique_values.min()} - {unique_values.max()}")
            print(f"📋 Valori: {list(unique_values)}")
            
            # Conta pixel per valore
            value_counts = {}
            for val in unique_values:
                count = np.sum(seg_array == val)
                percentage = (count / seg_array.size) * 100
                value_counts[val] = {'count': count, 'percentage': percentage}
                
            # Mostra statistiche ordinate per frequenza
            print(f"\n📈 Distribuzione pixel (ordinata per frequenza):")
            sorted_values = sorted(value_counts.items(), key=lambda x: x[1]['count'], reverse=True)
            for val, stats in sorted_values[:10]:  # Top 10
                print(f"  Grigio {val:3d}: {stats['count']:6d} pixel ({stats['percentage']:5.1f}%)")
                
            return {
                'unique_values': unique_values,
                'value_counts': value_counts,
                'shape': seg_array.shape
            }
            
        except Exception as e:
            print(f"❌ Errore analisi segmentation mask: {e}")
            return None

    def capture_with_segmentation(self):
        """Cattura sia l'immagine normale che la segmentation mask"""
        try:
            # PRIMA: Prova con immagini compresse (PNG/JPEG) che hanno header validi
            print("[DEBUG] Tentativo con immagini compresse...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),  # compressed=True
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)
            ])
            
            if len(responses) == 2 and all(len(r.image_data_uint8) > 0 for r in responses):
                print("[DEBUG] Risposte compresse ricevute")
                
                # Prova a processare le immagini compresse
                try:
                    # Scene image
                    scene_bytes = responses[0].image_data_uint8
                    print(f"[DEBUG] Scene compressed bytes: {len(scene_bytes)}")
                    print(f"[DEBUG] Scene header: {scene_bytes[:10] if len(scene_bytes) >= 10 else scene_bytes}")
                    
                    scene_img = Image.open(BytesIO(scene_bytes)).convert("RGB")
                    scene_img = scene_img.resize(IMG_SIZE, Image.LANCZOS)
                    print(f"[DEBUG] Scene compressa caricata: {scene_img.size}")
                    
                    # Segmentation image  
                    seg_bytes = responses[1].image_data_uint8
                    if len(seg_bytes) > 0:
                        print(f"[DEBUG] Segmentation compressed bytes: {len(seg_bytes)}")
                        seg_img = Image.open(BytesIO(seg_bytes)).convert("RGB")
                        seg_img = seg_img.resize(IMG_SIZE, Image.LANCZOS)
                        seg_array = np.array(seg_img)[:, :, 0]  # Solo canale R
                        print(f"[DEBUG] Segmentation compressa caricata, unique values: {len(np.unique(seg_array))}")
                        return scene_img, seg_array
                    else:
                        print("[DEBUG] Segmentation vuota, solo scene")
                        return scene_img, None
                        
                except Exception as comp_error:
                    print(f"[DEBUG] Errore con immagini compresse: {comp_error}")
                    # Continua con fallback raw
            
            # FALLBACK: Se compressed non funziona, prova raw con dimensioni fisse di AirSim
            print("[DEBUG] Fallback a immagini raw...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),  # raw
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
            ])
            
            if len(responses) != 2:
                print(f"❌ Numero di risposte inaspettato: {len(responses)}")
                return None, None
                
            # Processa immagine scene raw
            scene_response = responses[0]
            scene_bytes = scene_response.image_data_uint8
            if len(scene_bytes) == 0:
                print("❌ Immagine scene vuota")
                return None, None
            
            print(f"[DEBUG] Scene raw bytes: {len(scene_bytes)}")
            
            # AirSim restituisce tipicamente immagini in formato BGRA o RGB
            # Prova prima le dimensioni standard di AirSim
            scene_img = self.decode_airsim_image(scene_bytes, "scene")
            if scene_img is None:
                return None, None
            
            # Processa segmentation mask raw
            seg_response = responses[1]
            seg_bytes = seg_response.image_data_uint8
            seg_array = None
            
            if len(seg_bytes) > 0:
                print(f"[DEBUG] Segmentation raw bytes: {len(seg_bytes)}")
                seg_img = self.decode_airsim_image(seg_bytes, "segmentation")
                if seg_img is not None:
                    seg_array = np.array(seg_img)[:, :, 0]  # Solo canale R
                    print(f"[DEBUG] Segmentation decodificata, unique values: {len(np.unique(seg_array))}")
            
            return scene_img, seg_array
            
        except Exception as e:
            print(f"❌ Errore generale cattura: {e}")
            return None, None
    
    def decode_airsim_image(self, image_bytes, image_type):
        """Decodifica bytes di immagine da AirSim"""
        try:
            bytes_len = len(image_bytes)
            print(f"[DEBUG] Decodifica {image_type}: {bytes_len} bytes")
            
            # AirSim standard dimensions che funzionano spesso
            standard_sizes = [
                (256, 256, 4),  # BGRA
                (256, 256, 3),  # BGR/RGB
                (144, 256, 3),  # Aspect ratio diverso
                (192, 192, 3),  # Quadrato più piccolo
                (128, 128, 3),  # Ancora più piccolo
                (320, 240, 3),  # VGA
                (480, 270, 3),  # 16:9 piccolo
            ]
            
            for height, width, channels in standard_sizes:
                expected_size = height * width * channels
                if bytes_len == expected_size:
                    print(f"[DEBUG] Tentativo {image_type}: {height}x{width}x{channels}")
                    
                    # Decodifica array
                    img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, channels))
                    
                    # AirSim spesso usa BGR invece di RGB
                    if channels >= 3:
                        # Prova prima BGR -> RGB
                        img_array_rgb = img_array[:, :, [2, 1, 0]] if channels >= 3 else img_array
                        img = Image.fromarray(img_array_rgb[:, :, :3], 'RGB')
                    else:
                        img = Image.fromarray(img_array, 'L')
                    
                    # Ridimensiona alla target size
                    img = img.resize(IMG_SIZE, Image.LANCZOS)
                    print(f"[DEBUG] {image_type} decodificato: {height}x{width} -> {IMG_SIZE}")
                    return img
            
            # Se nessuna dimensione standard funziona, prova calcolo automatico
            print(f"[DEBUG] Calcolo automatico dimensioni per {bytes_len} bytes...")
            
            # Prova 3 canali (RGB)
            if bytes_len % 3 == 0:
                pixels = bytes_len // 3
                side = int(pixels ** 0.5)
                if side * side * 3 == bytes_len and side > 32:  # Dimensione ragionevole
                    print(f"[DEBUG] Tentativo automatico: {side}x{side}x3")
                    img_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((side, side, 3))
                    # Prova BGR -> RGB
                    img_array_rgb = img_array[:, :, [2, 1, 0]]
                    img = Image.fromarray(img_array_rgb, 'RGB')
                    img = img.resize(IMG_SIZE, Image.LANCZOS)
                    return img
            
            print(f"❌ Impossibile decodificare {image_type} con {bytes_len} bytes")
            return None
            
        except Exception as e:
            print(f"❌ Errore decodifica {image_type}: {e}")
            return None

    def extract_obstacles_from_segmentation(self, scene_img, seg_mask):
        """Estrae ostacoli usando la segmentation mask di AirSim"""
        if seg_mask is None:
            print("⚠️ Segmentation mask non disponibile, uso fallback")
            return self.extract_obstacles_fallback(scene_img)
            
        try:
            # Crea maschera per ostacoli basata sui valori grigi configurati
            obstacle_mask = np.zeros(seg_mask.shape, dtype=bool)
            
            # Aggiungi pixel che appartengono alle categorie "obstacles"
            for gray_value in SEGMENTATION_CATEGORIES['obstacles']:
                obstacle_mask |= (seg_mask == gray_value)
            
            # Converti in maschera alpha (0-255)
            alpha_mask = (obstacle_mask * 255).astype(np.uint8)
            
            # Aggiungi un po' di blur per bordi più naturali
            from PIL import ImageFilter
            alpha_pil = Image.fromarray(alpha_mask, 'L')
            alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=1))
            alpha_mask = np.array(alpha_pil)
            
            # Crea RGBA
            scene_array = np.array(scene_img)
            rgba_array = np.dstack([scene_array, alpha_mask])
            
            # Debug: salva le maschere per controllo
            if random.random() < 0.1:  # Salva 10% delle maschere per debug
                debug_dir = "debug_masks"
                os.makedirs(debug_dir, exist_ok=True)
                
                timestamp = int(time.time())
                
                # Salva segmentation originale
                seg_pil = Image.fromarray(seg_mask, 'L')
                seg_pil.save(f"{debug_dir}/seg_original_{timestamp}.png")
                
                # Salva maschera ostacoli
                obstacle_pil = Image.fromarray(alpha_mask, 'L')
                obstacle_pil.save(f"{debug_dir}/obstacles_{timestamp}.png")
                
                print(f"💾 Debug masks salvate: seg_original_{timestamp}.png, obstacles_{timestamp}.png")
            
            return Image.fromarray(rgba_array, 'RGBA')
            
        except Exception as e:
            print(f"❌ Errore estrazione ostacoli da segmentazione: {e}")
            return self.extract_obstacles_fallback(scene_img)
        """Cattura un'immagine dalla camera del drone"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
            ])
            
            if responses and len(responses) > 0:
                img_bytes = responses[0].image_data_uint8
                if len(img_bytes) == 0:
                    return None
                    
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                return img.resize(IMG_SIZE, Image.LANCZOS)
        except Exception as e:
            print(f"❌ Errore cattura immagine: {e}")
        
        return None
    
    def get_segmentation_mask(self):
        """Cattura la maschera di segmentazione da AirSim"""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)
            ])
            
            if responses and len(responses) > 0:
                img_bytes = responses[0].image_data_uint8
                if len(img_bytes) == 0:
                    print("[DEBUG] Nessun dato nella maschera di segmentazione")
                    return None
                
                # Converti in immagine PIL e ridimensiona
                mask = Image.open(BytesIO(img_bytes)).convert("L")
                mask = mask.resize(IMG_SIZE, Image.NEAREST)
                
                return mask
        except Exception as e:
            print(f"❌ Errore cattura maschera segmentazione: {e}")
        
        return None
    
    def extract_obstacles(self, anchor_img, segmentation_mask=None):
        """
        Estrae ostacoli usando prima la segmentation mask di AirSim,
        poi fallback su analisi colori se non disponibile
        """
        # Se abbiamo la segmentation mask, usala
        if segmentation_mask is not None:
            return self.extract_obstacles_from_segmentation(anchor_img, segmentation_mask)
        
        # Altrimenti usa il metodo fallback basato sui colori
        return self.extract_obstacles_fallback(anchor_img)

    def extract_obstacles_fallback(self, anchor_img):
        """
        FALLBACK: crea maschera basata sui COLORI quando segmentation non disponibile
        Identifica il cielo per colore (blu/grigio chiaro) e lo sostituisce
        """
        try:
            # Converti l'immagine in array numpy per analisi colori
            img_array = np.array(anchor_img)
            h, w, c = img_array.shape
            
            # IDENTIFICA IL CIELO PER COLORE - PIÙ PRECISO
            # Il cielo dovrebbe essere nella parte SUPERIORE dell'immagine
            # e avere colori omogenei
            
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Maschera cielo: combina diverse condizioni + posizione
            sky_mask = np.zeros((h, w), dtype=bool)
            
            # Condizione 1: Blu dominante (cielo blu) - PIÙ RIGOROSO
            blue_sky = (b > r + 30) & (b > g + 30) & (b > 120) & (b < 250)  # Evita bianco puro
            
            # Condizione 2: Grigio chiaro uniforme (cielo nuvoloso) - PIÙ RIGOROSO
            gray_diff = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
            gray_sky = (gray_diff < 20) & (r > 160) & (r < 240) & (g > 160) & (g < 240) & (b > 160) & (b < 240)
            
            # Condizione 3: Bianco puro solo se nella parte superiore
            white_sky = (r > 220) & (g > 220) & (b > 220)
            
            # FILTRO POSIZIONALE: Il cielo è principalmente nella metà superiore
            upper_half_mask = np.zeros((h, w), dtype=bool)
            upper_half_mask[:h//2, :] = True  # Solo metà superiore
            
            # Applica filtro posizionale alle condizioni
            blue_sky = blue_sky & upper_half_mask
            gray_sky = gray_sky & upper_half_mask  
            white_sky = white_sky & upper_half_mask
            
            # Combina tutte le condizioni
            sky_mask = blue_sky | gray_sky | white_sky
            
            # FILTRO MORFOLOGICO: Il cielo dovrebbe essere in regioni continue
            # Rimuovi piccole regioni isolate che non possono essere cielo
            from scipy import ndimage
            try:
                # Rimuovi regioni piccole (meno di 100 pixel)
                labeled_array, num_features = ndimage.label(sky_mask)
                for i in range(1, num_features + 1):
                    region_mask = (labeled_array == i)
                    if np.sum(region_mask) < 100:  # Regione troppo piccola
                        sky_mask[region_mask] = False
            except ImportError:
                # Se scipy non disponibile, usa filtro semplice
                pass
            
            # OSTACOLI = tutto ciò che NON è cielo (edifici, alberi)
            obstacle_mask = ~sky_mask
            
            # Converti in RGBA
            anchor_rgba = anchor_img.convert("RGBA")
            anchor_array = np.array(anchor_rgba)
            
            # Applica maschera: 255 per ostacoli (edifici), 0 per cielo
            alpha_channel = np.where(obstacle_mask, 255, 0).astype(np.uint8)
            anchor_array[:, :, 3] = alpha_channel
            
            # Crea immagine risultante  
            obstacles_img = Image.fromarray(anchor_array, 'RGBA')
            
            # Statistiche
            sky_pixels = np.sum(sky_mask)
            obstacle_pixels = np.sum(obstacle_mask)
            total_pixels = h * w
            
            sky_ratio = sky_pixels / total_pixels
            obstacle_ratio = obstacle_pixels / total_pixels
            
            print(f"[DEBUG] COLORE - Cielo da sostituire: {sky_ratio:.1%}, Ostacoli mantenuti: {obstacle_ratio:.1%}")
            
            # Se identifichiamo poco cielo, rilassiamo i criteri MA manteniamo filtro posizionale
            if sky_ratio < 0.1:  # Meno del 10% di cielo
                print("[DEBUG] Poco cielo identificato, rilasso criteri...")
                
                # Criteri più permissivi MA ancora con filtro posizionale
                blue_sky_relaxed = (b > r + 10) & (b > g + 10) & (b > 100)
                gray_diff_relaxed = np.maximum(np.maximum(np.abs(r - g), np.abs(r - b)), np.abs(g - b))
                gray_sky_relaxed = (gray_diff_relaxed < 30) & (r > 140) & (g > 140) & (b > 140)
                white_sky_relaxed = (r > 200) & (g > 200) & (b > 200)
                
                # MANTIENI il filtro posizionale anche nei criteri rilassati
                # Espandi a 3/4 superiori invece che solo metà
                upper_portion_mask = np.zeros((h, w), dtype=bool)
                upper_portion_mask[:3*h//4, :] = True  # 3/4 superiori
                
                blue_sky_relaxed = blue_sky_relaxed & upper_portion_mask
                gray_sky_relaxed = gray_sky_relaxed & upper_portion_mask
                white_sky_relaxed = white_sky_relaxed & upper_portion_mask
                
                sky_mask = blue_sky_relaxed | gray_sky_relaxed | white_sky_relaxed
                obstacle_mask = ~sky_mask
                
                # Riapplica
                alpha_channel = np.where(obstacle_mask, 255, 0).astype(np.uint8)
                anchor_array[:, :, 3] = alpha_channel
                obstacles_img = Image.fromarray(anchor_array, 'RGBA')
                
                sky_ratio = np.sum(sky_mask) / total_pixels
                obstacle_ratio = np.sum(obstacle_mask) / total_pixels
                print(f"[DEBUG] COLORE RILASSATO - Cielo da sostituire: {sky_ratio:.1%}, Ostacoli mantenuti: {obstacle_ratio:.1%}")
            
            return obstacles_img
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            return None
    
    def generate_positives(self, anchor_img, segmentation_mask=None, num_positives=6):
        """
        Genera 6 positivi con logica specifica:
        1-2: Rimuove sky, rimpiazza con black/white
        3-4: Solo trees+obstacles, sfondo indoor
        5-6: Buildings+trees+obstacles, sfondo b_X.png
        """
        if segmentation_mask is None:
            print("⚠️ Nessuna maschera di segmentazione, uso anchor originale")
            return [anchor_img.copy() for _ in range(num_positives)]
        
        # Ottieni background files
        bg_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        ind_files = [f for f in bg_files if f.startswith('ind_')]
        b_files = [f for f in bg_files if f.startswith('b_')]
        
        positives = []
        
        try:
            # Converti segmentation mask in array numpy
            seg_array = np.array(segmentation_mask)
            
            # POSITIVI 1-2: Rimuove sky, rimpiazza con black/white
            for i, bg_color in enumerate(['black', 'white']):
                print(f"[DEBUG] Generando positivo {i+1}: Rimosso sky, sfondo {bg_color}")
                
                # Crea maschera: tutto tranne sky
                mask = np.ones_like(seg_array, dtype=bool)
                if 'sky' in SEGMENTATION_CATEGORIES:
                    sky_values = SEGMENTATION_CATEGORIES['sky']
                    print(f"[DEBUG] Valori sky da configurazione: {sky_values}")
                    
                    # Rimuovi pixel che corrispondono ai valori sky
                    for sky_val in sky_values:
                        sky_pixels = (seg_array == sky_val)
                        mask &= ~sky_pixels  # Rimuovi sky dalla maschera
                        print(f"[DEBUG] Sky value {sky_val}: {np.sum(sky_pixels)} pixels")
                    
                    print(f"[DEBUG] Maschera finale (non-sky): {np.sum(mask)} pixel conservati su {mask.size}")
                else:
                    print("⚠️ Categoria 'sky' non trovata in configurazione")
                
                # Crea immagine con solo elementi non-sky
                result_img = self.apply_selective_mask(anchor_img, mask, bg_color)
                positives.append(result_img)
            
            # POSITIVI 3-4: Solo trees+obstacles, sfondo indoor
            for i in range(2):
                print(f"[DEBUG] Generando positivo {i+3}: Solo trees+obstacles, sfondo indoor")
                
                # Crea maschera: solo trees e obstacles
                mask = np.zeros_like(seg_array, dtype=bool)
                for category in ['trees', 'obstacles']:
                    if category in SEGMENTATION_CATEGORIES:
                        cat_values = SEGMENTATION_CATEGORIES[category]
                        print(f"[DEBUG] Valori {category} da configurazione: {cat_values}")
                        
                        for val in cat_values:
                            cat_pixels = (seg_array == val)
                            mask |= cat_pixels
                            print(f"[DEBUG] {category} value {val}: {np.sum(cat_pixels)} pixels")
                    else:
                        print(f"⚠️ Categoria '{category}' non trovata in configurazione")
                
                print(f"[DEBUG] Maschera trees+obstacles: {np.sum(mask)} pixel conservati")
                
                # Seleziona sfondo indoor
                if ind_files:
                    bg_file = random.choice(ind_files)
                    result_img = self.apply_selective_mask(anchor_img, mask, bg_file, apply_shadow=True)
                else:
                    print("⚠️ Nessun file indoor trovato, uso nero")
                    result_img = self.apply_selective_mask(anchor_img, mask, 'black')
                
                positives.append(result_img)
            
            # POSITIVI 5-6: Buildings+trees+obstacles, sfondo b_X.png
            for i in range(2):
                print(f"[DEBUG] Generando positivo {i+5}: Buildings+trees+obstacles, sfondo b_X")
                
                # Crea maschera: buildings, trees e obstacles
                mask = np.zeros_like(seg_array, dtype=bool)
                for category in ['buildings', 'trees', 'obstacles']:
                    if category in SEGMENTATION_CATEGORIES:
                        cat_values = SEGMENTATION_CATEGORIES[category]
                        print(f"[DEBUG] Valori {category} da configurazione: {cat_values}")
                        
                        for val in cat_values:
                            cat_pixels = (seg_array == val)
                            mask |= cat_pixels
                            print(f"[DEBUG] {category} value {val}: {np.sum(cat_pixels)} pixels")
                    else:
                        print(f"⚠️ Categoria '{category}' non trovata in configurazione")
                
                print(f"[DEBUG] Maschera buildings+trees+obstacles: {np.sum(mask)} pixel conservati")
                
                # Seleziona sfondo b_X
                if b_files:
                    bg_file = random.choice(b_files)
                    result_img = self.apply_selective_mask(anchor_img, mask, bg_file)
                else:
                    print("⚠️ Nessun file b_X trovato, uso nero")
                    result_img = self.apply_selective_mask(anchor_img, mask, 'black')
                
                positives.append(result_img)
                
        except Exception as e:
            print(f"❌ Errore nella generazione positivi: {e}")
            # Fallback: restituisci copie dell'anchor
            return [anchor_img.copy() for _ in range(num_positives)]
        
        return positives
    
    def apply_selective_mask(self, anchor_img, mask, background, apply_shadow=False):
        """
        Applica una maschera selettiva mantenendo solo i pixel specificati
        e sostituendo il resto con il background
        """
        try:
            # Converti anchor in array
            anchor_array = np.array(anchor_img)
            print(f"[DEBUG] Anchor shape: {anchor_array.shape}")
            print(f"[DEBUG] Mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"[DEBUG] Mask True pixels: {np.sum(mask)}, Total: {mask.size}")
            print(f"[DEBUG] Background: {background}")
            
            # Verifica compatibilità dimensioni
            if anchor_array.shape[:2] != mask.shape:
                print(f"❌ Dimensioni incompatibili: anchor {anchor_array.shape[:2]} vs mask {mask.shape}")
                return anchor_img.copy()
            
            # Crea immagine risultato
            result_array = np.zeros_like(anchor_array)
            
            # Carica o crea background
            if background in ['black', 'white']:
                # Background solido
                bg_color = 0 if background == 'black' else 255
                bg_array = np.full_like(anchor_array, bg_color)
                print(f"[DEBUG] Background solido: {bg_color}")
            else:
                # Carica file background
                bg_path = os.path.join(BACKGROUNDS_DIR, background)
                if os.path.exists(bg_path):
                    bg_img = Image.open(bg_path).convert("RGB").resize(IMG_SIZE)
                    bg_array = np.array(bg_img)
                    print(f"[DEBUG] Background caricato: {bg_path}, shape: {bg_array.shape}")
                else:
                    print(f"⚠️ Background {background} non trovato, uso nero")
                    bg_array = np.zeros_like(anchor_array)
            
            # Applica maschera: dove mask=True usa anchor, altrimenti background
            result_array[mask] = anchor_array[mask]
            result_array[~mask] = bg_array[~mask]
            
            print(f"[DEBUG] Pixel conservati: {np.sum(mask)}, sostituiti: {np.sum(~mask)}")
            
            # Converti risultato in immagine
            result_img = Image.fromarray(result_array.astype(np.uint8))
            
            # Applica ombreggiatura se richiesta (per indoor)
            if apply_shadow:
                result_img = self.apply_indoor_shadow(result_img, mask)
            
            return result_img
            
        except Exception as e:
            print(f"❌ Errore in apply_selective_mask: {e}")
            return anchor_img.copy()
    
    def apply_indoor_shadow(self, img, obstacle_mask):
        """Applica effetto ombreggiatura per ambienti indoor"""
        try:
            result_array = np.array(img).astype(float)
            
            # Applica ombreggiatura solo dove ci sono ostacoli
            shadow_factor = 0.7  # 30% più scuro
            result_array[obstacle_mask] *= shadow_factor
            
            # Aggiungi tint blu/grigio per effetto indoor
            blue_tint = 10
            result_array[obstacle_mask, 2] = np.minimum(255, result_array[obstacle_mask, 2] + blue_tint)
            
            # Riconverti
            result_array = np.clip(result_array, 0, 255).astype(np.uint8)
            return Image.fromarray(result_array)
            
        except Exception as e:
            print(f"❌ Errore ombreggiatura indoor: {e}")
            return img
    
    def move_drone_randomly(self):
        """Muove il drone in modo casuale nell'ambiente"""
        # Ottiene la posizione attuale
        state = self.client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val
        
        # Calcola velocità per mantenere altitudine nel range
        if current_alt > MAX_ALTITUDE:
            vz = -1.0  # Scendi
        elif current_alt < MIN_ALTITUDE:
            vz = 1.0   # Sali
        else:
            vz = random.uniform(-0.5, 0.5)  # Movimento casuale verticale
        
        # Movimento orizzontale casuale
        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        yaw_rate = random.uniform(-30, 30)  # Rotazione casuale
        
        # Esegue il movimento
        self.client.moveByVelocityAsync(
            vx, vy, vz, CAPTURE_INTERVAL,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
    
    def save_anchor_set(self, anchor_img, positives, anchor_idx, segmentation_mask=None):
        """Salva un set completo: anchor + positivi + maschera di segmentazione"""
        # Crea cartella per questo anchor
        anchor_dir = os.path.join(DATASET_DIR, f"anchor_{anchor_idx:05d}")
        os.makedirs(anchor_dir, exist_ok=True)
        
        try:
            # Salva l'immagine anchor
            anchor_path = os.path.join(anchor_dir, "anchor.png")
            anchor_img.save(anchor_path)
            
            # Salva la maschera di segmentazione se disponibile
            if segmentation_mask is not None:
                # Salva come array numpy (formato .npy)
                mask_npy_path = os.path.join(anchor_dir, "segmentation_mask.npy")
                np.save(mask_npy_path, segmentation_mask)
                
                # Salva anche come immagine PNG per visualizzazione
                mask_png_path = os.path.join(anchor_dir, "segmentation_mask.png")
                
                # Normalizza la maschera per visualizzazione (0-255)
                if segmentation_mask.max() > 0:
                    mask_normalized = ((segmentation_mask / segmentation_mask.max()) * 255).astype(np.uint8)
                else:
                    mask_normalized = segmentation_mask.astype(np.uint8)
                
                mask_img = Image.fromarray(mask_normalized, mode='L')
                mask_img.save(mask_png_path)
                
                # Salva anche una versione colorata per debug
                mask_debug_path = os.path.join(anchor_dir, "segmentation_debug.png")
                self.save_colored_segmentation_mask(segmentation_mask, mask_debug_path)
                
                print(f"💾 Salvata maschera segmentazione (valori unici: {len(np.unique(segmentation_mask))})")
            
            # Salva le immagini positive
            for i, pos_img in enumerate(positives, 1):
                pos_path = os.path.join(anchor_dir, f"positive_{i}.png")
                pos_img.save(pos_path)
                
            print(f"💾 Salvato anchor_{anchor_idx:05d} con {len(positives)} positivi")
            return True
            
        except Exception as e:
            print(f"❌ Errore nel salvataggio anchor_{anchor_idx:05d}: {e}")
            return False
    
    def save_colored_segmentation_mask(self, segmentation_mask, output_path):
        """Salva una versione colorata della maschera di segmentazione per debug"""
        try:
            # Crea immagine RGB per visualizzazione colorata
            height, width = segmentation_mask.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Definisci colori per le categorie
            category_colors = {
                'sky': [135, 206, 235],      # Celeste
                'trees': [34, 139, 34],      # Verde foresta
                'buildings': [139, 69, 19],  # Marrone
                'ground': [160, 82, 45],     # Marrone terra
                'obstacles': [255, 0, 0]     # Rosso
            }
            
            # Colora ogni pixel in base alla categoria
            for category, color in category_colors.items():
                if category in SEGMENTATION_CATEGORIES:
                    for value in SEGMENTATION_CATEGORIES[category]:
                        mask = (segmentation_mask == value)
                        colored_mask[mask] = color
            
            # Pixel non categorizzati rimangono neri
            
            # Salva immagine colorata
            colored_img = Image.fromarray(colored_mask, 'RGB')
            colored_img.save(output_path)
            
        except Exception as e:
            print(f"⚠️ Errore salvataggio maschera colorata: {e}")

    def generate_dataset(self):
        """Funzione principale per generare il dataset"""
        print(f"🚀 Inizio generazione dataset - {N_SAMPLES} samples")
        print(f"📍 Altitudine: {MIN_ALTITUDE}m - {MAX_ALTITUDE}m")
        print(f"⏱️ Intervallo cattura: {CAPTURE_INTERVAL}s")
        print("-" * 50)
        
        # Connessione e setup
        if not self.connect_airsim():
            return False
            
        self.takeoff_and_setup()
        
        # Loop principale di generazione
        successful_captures = 0
        
        for i in range(N_SAMPLES):
            print(f"📸 Cattura {i+1}/{N_SAMPLES}", end=" - ")
            
            # Muovi il drone
            self.move_drone_randomly()
            
            # Aspetta che il movimento si stabilizzi
            time.sleep(0.5)
            
            # Cattura l'immagine anchor con segmentazione - USA SOLO COMPRESSED
            try:
                responses = self.client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, True),  # compressed=True
                    airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)  # compressed=True
                ])
                
                if len(responses) != 2 or any(len(r.image_data_uint8) == 0 for r in responses):
                    print("❌ Cattura fallita o dati vuoti")
                    continue
                    
                # Scene image
                anchor_img = Image.open(BytesIO(responses[0].image_data_uint8)).convert("RGB")
                anchor_img = anchor_img.resize(IMG_SIZE, Image.LANCZOS)
                
                # Segmentation mask  
                seg_img = Image.open(BytesIO(responses[1].image_data_uint8)).convert("RGB")
                seg_img = seg_img.resize(IMG_SIZE, Image.LANCZOS)
                segmentation_mask = np.array(seg_img)[:, :, 0]  # Solo canale R
                
                print(f"✅ Cattura {i+1}/{N_SAMPLES} - Valori unici: {len(np.unique(segmentation_mask))}")
                
            except Exception as capture_error:
                print(f"❌ Cattura {i+1}/{N_SAMPLES} - Errore: {capture_error}")
                continue
            
            # Analizza segmentazione se disponibile (solo primo sample per debug)
            if i == 0 and segmentation_mask is not None:
                print("\n🔍 Analizzo segmentation mask del primo sample...")
                self.analyze_segmentation_values(segmentation_mask)
            
            # Genera le immagini positive usando la segmentazione
            positives = self.generate_positives(anchor_img, segmentation_mask)
            
            # Salva il set completo
            if self.save_anchor_set(anchor_img, positives, i, segmentation_mask):
                successful_captures += 1
            
            # Piccola pausa prima del prossimo ciclo
            time.sleep(0.5)
        
        # Atterraggio e cleanup
        print("\n🛬 Atterraggio del drone...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
        print(f"✅ Generazione completata!")
        print(f"📊 {successful_captures}/{N_SAMPLES} anchor generati con successo")
        print(f"📁 Dataset salvato in: {DATASET_DIR}")
        
        return True

def main():
    """Funzione principale"""
    try:
        generator = DatasetGenerator()
        generator.generate_dataset()
    except KeyboardInterrupt:
        print("\n⏹️ Generazione interrotta dall'utente")
    except Exception as e:
        print(f"❌ Errore durante la generazione: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()