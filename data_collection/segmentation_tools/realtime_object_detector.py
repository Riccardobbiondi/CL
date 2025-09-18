#!/usr/bin/env python3
"""
Object detector semplificato per AirSim
Riconosce oggetti in modo oggettivo usando solo PIL e numpy:
- Analisi colori RGB/HSV
- Analisi posizionale
- Filtri per continuità regioni
- Segmentazione semantica basata su euristica
"""

import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Importazioni opzionali per features avanzate
try:
    import cv2
    HAS_OPENCV = True
    print("✅ OpenCV disponibile - usando features avanzate")
except ImportError:
    HAS_OPENCV = False
    print("ℹ️ OpenCV non disponibile - usando solo PIL e numpy")

class RealtimeObjectDetector:
    def __init__(self):
        """Inizializza detector semplificato con parametri ottimizzati"""
        print("🔧 Inizializzando Object Detector Semplificato...")
        
        # Valori per categorie semantiche
        self.category_values = {
            'sky': 1,
            'trees': 2, 
            'buildings': 3,
            'ground': 4,
            'water': 5,
            'roads': 6,
            'unknown': 0
        }
        
        # Soglie colori RGB per identificazione oggetti
        self.color_thresholds = {
            'sky': {
                'blue_sky': {'r_max': 180, 'g_max': 200, 'b_min': 150, 'b_dominance': 20},
                'gray_sky': {'r_min': 150, 'g_min': 150, 'b_min': 150, 'uniformity': 25},
                'white_sky': {'r_min': 200, 'g_min': 200, 'b_min': 200}
            },
            'trees': {
                'green_vegetation': {'g_min': 60, 'g_dominance': 15, 'r_max': 150, 'b_max': 150},
                'dark_green': {'g_min': 40, 'g_dominance': 10, 'brightness_max': 180}
            },
            'buildings': {
                'gray_concrete': {'uniformity': 30, 'brightness_min': 60, 'brightness_max': 180},
                'red_brick': {'r_min': 100, 'r_dominance': 20, 'brightness_max': 200}
            },
            'ground': {
                'brown_soil': {'r_min': 80, 'r_dominance': 10, 'g_min': 60, 'brightness_max': 150},
                'gray_pavement': {'uniformity': 25, 'brightness_min': 50, 'brightness_max': 130}
            }
        }
        
        print("✅ Object Detector Semplificato inizializzato")
        
    def process_airsim_image(self, airsim_image):
        """Processa immagine AirSim con analisi semplificata ma efficace"""
        try:
            # Converti in array numpy
            if isinstance(airsim_image, Image.Image):
                img_array = np.array(airsim_image)
            else:
                img_array = airsim_image
            
            h, w = img_array.shape[:2]
            print(f"🔍 Processando immagine {w}x{h}")
            
            # 1. ANALISI COLORI RGB
            color_masks = self.analyze_colors_rgb(img_array)
            
            # 2. ANALISI POSIZIONALE
            position_weights = self.create_position_weights(h, w)
            
            # 3. COMBINAZIONE INTELLIGENTE
            semantic_mask = self.combine_analysis(color_masks, position_weights, h, w)
            
            # 4. POST-PROCESSING SEMPLIFICATO
            semantic_mask = self.simple_post_process(semantic_mask)
            
            # 5. STATISTICHE
            stats = self.calculate_statistics(semantic_mask)
            
            # 6. GENERA DETECTION BOXES
            detections = self.generate_simple_detections(semantic_mask)
            
            result = {
                'semantic_mask': semantic_mask,
                'detections': detections,
                'statistics': stats,
                'category_mapping': {v: k for k, v in self.category_values.items()},
                'debug_info': {
                    'processing_method': 'simplified_rgb_analysis',
                    'opencv_available': HAS_OPENCV,
                    'categories_detected': len([k for k, v in stats.items() if k > 0 and v > 100])
                }
            }
            
            print(f"✅ Processamento completato: {len(stats)} categorie rilevate")
            return result
            
        except Exception as e:
            print(f"❌ Errore processing image: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback robusto
            h, w = 224, 224
            if isinstance(airsim_image, Image.Image):
                w, h = airsim_image.size
            elif hasattr(airsim_image, 'shape'):
                h, w = airsim_image.shape[:2]
                
            return {
                'semantic_mask': np.zeros((h, w), dtype=np.uint8),
                'detections': [],
                'statistics': {0: h * w},
                'category_mapping': {0: 'unknown'},
                'debug_info': {'error': str(e)}
            }
    
    def analyze_colors_rgb(self, img_array):
        """Analisi colori AGGRESSIVA per catturare meglio tutti gli oggetti"""
        h, w = img_array.shape[:2]
        color_masks = {}
        
        # Estrai canali RGB
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32)
        b = img_array[:, :, 2].astype(np.float32)
        
        # Calcola metriche avanzate
        brightness = (r + g + b) / 3
        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        uniformity = max_rgb - min_rgb
        
        # Saturazione approssimata
        saturation = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)
        
        # ANALISI CIELO - PIÙ RIGOROSA per evitare false positive
        sky_mask = np.zeros((h, w), dtype=bool)
        
        # Cielo blu: soglie più rigorose per evitare di prendere edifici
        blue_sky = (b > r + 25) & (b > g + 20) & (b > 120) & (brightness > 110) & (saturation > 0.15)
        
        # Cielo grigio: molto più rigoroso, solo aree molto uniformi e luminose
        gray_sky = (uniformity < 20) & (brightness > 170) & (saturation < 0.15) & (b >= r - 5) & (b >= g - 5)
        
        # Cielo bianco/chiaro: soglie alzate per evitare edifici bianchi
        white_sky = (r > 200) & (g > 200) & (b > 200) & (uniformity < 15) & (brightness > 200)
        
        # Cielo con nuvole: più conservativo, deve essere chiaramente celeste
        cloudy_sky = (brightness > 180) & (saturation < 0.2) & (b > r + 10) & (b > g + 10) & (uniformity < 25)
        
        sky_mask = blue_sky | gray_sky | white_sky | cloudy_sky
        color_masks['sky'] = sky_mask
        
        # ANALISI VEGETAZIONE - PIÙ AGGRESSIVA
        trees_mask = np.zeros((h, w), dtype=bool)
        
        # Verde classico: soglie più ampie
        classic_green = (g > r + 10) & (g > b + 10) & (g > 60)
        
        # Verde scuro (ombre/bosco)
        dark_green = (g > r + 5) & (g > b + 8) & (g > 30) & (brightness < 120)
        
        # Verde saturo 
        bright_green = (g > r + 15) & (g > b + 20) & (g > 80) & (saturation > 0.2)
        
        # Marrone degli alberi/tronchi
        brown_trees = (r > g + 5) & (r > b + 15) & (g > b + 5) & (brightness > 40) & (brightness < 120) & (saturation > 0.2)
        
        trees_mask = classic_green | dark_green | bright_green | brown_trees
        color_masks['trees'] = trees_mask
        
        # ANALISI EDIFICI - PIÙ AGGRESSIVA per catturare quello che il cielo non deve prendere  
        buildings_mask = np.zeros((h, w), dtype=bool)
        
        # Grigio cemento: soglie ampliate per catturare più variazioni
        concrete_gray = (uniformity < 50) & (brightness > 50) & (brightness < 190) & (saturation < 0.35)
        
        # Mattoni rossi e strutture colorate
        red_brick = (r > g + 10) & (r > b + 15) & (r > 70) & (brightness > 50) & (brightness < 180)
        
        # Superfici colorate edifici (più permissivo)
        colored_buildings = (saturation > 0.1) & (saturation < 0.7) & (brightness > 60) & (brightness < 190) & (uniformity < 60)
        
        # Materiali bianchi/chiari degli edifici (distinguere dal cielo)
        light_materials = (r > 140) & (g > 140) & (b > 140) & (brightness > 140) & (brightness < 220) & ~sky_mask
        
        # Superfici verticali/strutture (based on position hints)
        vertical_structures = (uniformity < 45) & (brightness > 40) & (brightness < 200) & (saturation < 0.5)
        
        # Escludi vegetazione e cielo, ma sii aggressivo nel prendere tutto il resto
        buildings_mask = (concrete_gray | red_brick | colored_buildings | light_materials | vertical_structures) & ~sky_mask & ~trees_mask
        color_masks['buildings'] = buildings_mask
        
        # ANALISI TERRENO - PIÙ AGGRESSIVA
        ground_mask = np.zeros((h, w), dtype=bool)
        
        # Asfalto/strade: grigio scuro
        asphalt = (uniformity < 30) & (brightness > 20) & (brightness < 120) & (saturation < 0.25)
        
        # Terra/suolo: marrone
        soil = (r > g) & (r > b + 5) & (g > b) & (brightness > 40) & (brightness < 150) & (saturation > 0.1)
        
        # Cemento/pavimenti
        pavement = (uniformity < 40) & (brightness > 70) & (brightness < 160) & (saturation < 0.2)
        
        # Superfici scure generiche
        dark_surfaces = (brightness < 80) & (saturation < 0.3) & (uniformity < 35)
        
        # Escludi altre categorie
        ground_mask = (asphalt | soil | pavement | dark_surfaces) & ~sky_mask & ~trees_mask & ~buildings_mask
        color_masks['ground'] = ground_mask
        
        # DEBUG: Stampa statistiche per tuning
        if np.random.random() < 0.1:  # 10% delle volte
            print(f"[DEBUG COLORI] Sky: {np.sum(sky_mask)/sky_mask.size:.1%}, Trees: {np.sum(trees_mask)/trees_mask.size:.1%}, Buildings: {np.sum(buildings_mask)/buildings_mask.size:.1%}, Ground: {np.sum(ground_mask)/ground_mask.size:.1%}")
        
        return color_masks
    
    def create_position_weights(self, h, w):
        """Crea pesi posizionali per ogni categoria"""
        position_weights = {}
        
        # Crea griglia Y normalizzata (0=top, 1=bottom)
        y_grid = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
        
        # Peso per cielo (più probabile in alto)
        sky_weight = np.maximum(0, 1.5 - 2 * y_grid)  # Alto peso in alto, decresce verso il basso
        position_weights['sky'] = sky_weight
        
        # Peso per terreno (più probabile in basso)
        ground_weight = np.maximum(0, 2 * y_grid - 0.5)  # Alto peso in basso
        position_weights['ground'] = ground_weight
        
        # Peso per alberi (centro-alto)
        trees_weight = 1.0 - np.abs(y_grid - 0.4) * 2  # Picco a y=0.4
        trees_weight = np.maximum(0, trees_weight)
        position_weights['trees'] = trees_weight
        
        # Peso per edifici (centro)
        buildings_weight = 1.0 - np.abs(y_grid - 0.5) * 3  # Picco a y=0.5
        buildings_weight = np.maximum(0, buildings_weight)
        position_weights['buildings'] = buildings_weight
        
        return position_weights
    
    def combine_analysis(self, color_masks, position_weights, h, w):
        """Combina analisi colori e posizione con logica ANTI-CONFLITTO SKY/BUILDINGS"""
        final_mask = np.zeros((h, w), dtype=np.uint8)
        
        # STRATEGIA ANTI-CONFLITTO: Prima identifica edifici, poi cielo
        
        # 1. PRIMA: Identifica edifici (priorità ALTA per evitare che il cielo li rubi)
        buildings_confidence = color_masks['buildings'].astype(np.float32) * position_weights['buildings']
        buildings_strong = buildings_confidence > 0.2  # Soglia bassa per catturare tutto
        final_mask[buildings_strong] = self.category_values['buildings']
        
        # 2. SECONDO: Identifica vegetazione 
        trees_confidence = color_masks['trees'].astype(np.float32) * position_weights['trees']
        trees_strong = (trees_confidence > 0.25) & (final_mask == 0)
        final_mask[trees_strong] = self.category_values['trees']
        
        # 3. TERZO: Identifica cielo (ma SOLO dove non ci sono già edifici/alberi)
        sky_confidence = color_masks['sky'].astype(np.float32) * position_weights['sky']
        sky_strong = (sky_confidence > 0.5) & (final_mask == 0)  # Soglia più alta per essere conservativi
        final_mask[sky_strong] = self.category_values['sky']
        
        # 4. QUARTO: Identifica terreno 
        ground_confidence = color_masks['ground'].astype(np.float32) * position_weights['ground']
        ground_strong = (ground_confidence > 0.2) & (final_mask == 0)
        final_mask[ground_strong] = self.category_values['ground']
        
        # 5. CORREZIONE: Risolvi conflitti sky/buildings nelle zone di confine
        final_mask = self.resolve_sky_building_conflicts(final_mask, color_masks, h, w)
        
        # 6. AGGIUNTA: Classificazione di backup per pixel rimanenti
        remaining_pixels = (final_mask == 0)
        
        # Classifica pixel rimanenti con preferenza per buildings su sky
        for category in ['buildings', 'trees', 'sky', 'ground']:
            if category in color_masks and category in self.category_values:
                low_confidence = color_masks[category] & remaining_pixels
                final_mask[low_confidence] = self.category_values[category]
                remaining_pixels = remaining_pixels & ~low_confidence
        
        # 7. REFINEMENT: Applica logica spaziale
        final_mask = self.spatial_refinement(final_mask, color_masks)
        
        return final_mask
    
    def resolve_sky_building_conflicts(self, mask, color_masks, h, w):
        """Risolve conflitti tra cielo e edifici"""
        sky_value = self.category_values['sky']
        buildings_value = self.category_values['buildings']
        
        # Trova pixel classificati come cielo
        sky_pixels = (mask == sky_value)
        
        # Per ogni pixel cielo, controlla se potrebbe essere un edificio
        for y in range(h):
            for x in range(w):
                if sky_pixels[y, x]:
                    # Se è nella parte bassa dell'immagine, probabilmente è un edificio
                    if y > h * 0.4:  # Sotto il 40% dell'immagine
                        # Controlla se ha caratteristiche di edificio
                        if color_masks['buildings'][y, x]:
                            mask[y, x] = buildings_value
                    
                    # Se ha molti vicini edifici, probabilmente è un edificio
                    building_neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] == buildings_value:
                                    building_neighbors += 1
                    
                    if building_neighbors >= 3:  # 3+ vicini edifici
                        mask[y, x] = buildings_value
        
        return mask
    
    def spatial_refinement(self, mask, color_masks):
        """Applica correzioni spaziali per migliorare la segmentazione"""
        h, w = mask.shape
        refined_mask = mask.copy()
        
        # REGOLA 1: Il cielo dovrebbe essere continuo nella parte superiore
        sky_pixels = np.where(mask == self.category_values['sky'])
        if len(sky_pixels[0]) > 0:
            # Trova la riga più bassa con cielo
            sky_bottom = np.max(sky_pixels[0])
            # Se il cielo è frammentato in alto, riempi i buchi
            for y in range(0, min(sky_bottom + 10, h//3)):  # Solo nel terzo superiore
                for x in range(w):
                    if mask[y, x] == 0 and color_masks['sky'][y, x]:
                        # Controlla se ha vicini cielo
                        neighbors_sky = 0
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if mask[ny, nx] == self.category_values['sky']:
                                        neighbors_sky += 1
                        if neighbors_sky >= 2:
                            refined_mask[y, x] = self.category_values['sky']
        
        # REGOLA 2: Gli edifici dovrebbero avere forme più regolari
        # Rimuovi piccoli cluster di edifici isolati
        buildings_mask = (mask == self.category_values['buildings'])
        if np.sum(buildings_mask) > 0:
            # Usa morphological operations se OpenCV disponibile
            if HAS_OPENCV:
                buildings_binary = buildings_mask.astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # Close per connettere parti vicine
                buildings_binary = cv2.morphologyEx(buildings_binary, cv2.MORPH_CLOSE, kernel)
                # Open per rimuovere noise
                buildings_binary = cv2.morphologyEx(buildings_binary, cv2.MORPH_OPEN, kernel)
                refined_mask[buildings_mask] = 0  # Reset
                refined_mask[buildings_binary > 0] = self.category_values['buildings']
        
        # REGOLA 3: La vegetazione dovrebbe essere più compatta
        trees_mask = (mask == self.category_values['trees'])
        if np.sum(trees_mask) > 0 and HAS_OPENCV:
            trees_binary = trees_mask.astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Chiudi buchi nella vegetazione
            trees_binary = cv2.morphologyEx(trees_binary, cv2.MORPH_CLOSE, kernel)
            refined_mask[trees_mask] = 0  # Reset
            refined_mask[trees_binary > 0] = self.category_values['trees']
        
        return refined_mask
    
    def simple_post_process(self, semantic_mask):
        """Post-processing semplificato senza OpenCV"""
        # Se OpenCV è disponibile, usa morphological operations
        if HAS_OPENCV:
            return self.opencv_post_process(semantic_mask)
        else:
            return self.numpy_post_process(semantic_mask)
    
    def numpy_post_process(self, semantic_mask):
        """Post-processing usando solo numpy"""
        h, w = semantic_mask.shape
        processed_mask = semantic_mask.copy()
        
        # Filtro semplice: rimuovi pixel isolati
        for category_value in self.category_values.values():
            if category_value == 0:
                continue
                
            category_mask = (semantic_mask == category_value)
            
            # Per ogni pixel di questa categoria, controlla i vicini
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if category_mask[y, x]:
                        # Conta vicini della stessa categoria
                        neighbors = semantic_mask[y-1:y+2, x-1:x+2]
                        same_category_count = np.sum(neighbors == category_value)
                        
                        # Se ha pochi vicini della stessa categoria, rimuovilo
                        if same_category_count < 3:  # Meno di 3 su 9 pixel
                            processed_mask[y, x] = 0
        
        return processed_mask
    
    def opencv_post_process(self, semantic_mask):
        """Post-processing avanzato con OpenCV"""
        try:
            for category_value in self.category_values.values():
                if category_value == 0:
                    continue
                    
                # Crea maschera binaria per questa categoria
                category_mask = (semantic_mask == category_value).astype(np.uint8) * 255
                
                # Morphological operations
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                
                # Close piccoli buchi
                category_mask = cv2.morphologyEx(category_mask, cv2.MORPH_CLOSE, kernel)
                
                # Rimuovi piccole regioni
                category_mask = cv2.morphologyEx(category_mask, cv2.MORPH_OPEN, kernel)
                
                # Riapplica alla maschera finale
                semantic_mask[category_mask > 0] = category_value
                semantic_mask[(semantic_mask == category_value) & (category_mask == 0)] = 0
            
            return semantic_mask
            
        except Exception as e:
            print(f"⚠️ Errore OpenCV post-processing: {e}")
            return self.numpy_post_process(semantic_mask)
    
    def calculate_statistics(self, semantic_mask):
        """Calcola statistiche e valida la segmentazione"""
        unique_values, counts = np.unique(semantic_mask, return_counts=True)
        stats = dict(zip(unique_values, counts))
        
        total_pixels = semantic_mask.size
        
        print("📊 STATISTICHE SEGMENTAZIONE:")
        for value, count in stats.items():
            category = next((k for k, v in self.category_values.items() if v == value), 'unknown')
            percentage = (count / total_pixels) * 100
            print(f"   {category.upper()}: {percentage:.1f}% ({count} pixel)")
        
        # VALIDAZIONE: controlla se la segmentazione è ragionevole (PIÙ PERMISSIVA)
        unknown_ratio = stats.get(0, 0) / total_pixels
        categorized_ratio = 1.0 - unknown_ratio
        
        if unknown_ratio > 0.85:  # Soglia alzata da 0.8 a 0.85
            print("⚠️ WARNING: Segmentazione troppo conservativa (>85% unknown)")
        elif unknown_ratio < 0.05:  # Soglia abbassata da 0.1 a 0.05  
            print("⚠️ WARNING: Segmentazione troppo aggressiva (<5% unknown)")
        else:
            print(f"✅ Segmentazione bilanciata ({categorized_ratio:.1%} categorizzato)")
        
        # Controlla proporzioni ragionevoli (con soglie più permissive)
        sky_ratio = stats.get(self.category_values['sky'], 0) / total_pixels
        trees_ratio = stats.get(self.category_values['trees'], 0) / total_pixels
        
        if sky_ratio > 0.8:
            print("⚠️ WARNING: Troppo cielo identificato (>80%)")
        if trees_ratio > 0.7:
            print("⚠️ WARNING: Troppa vegetazione identificata (>70%)")
        
        return stats
    
    def generate_simple_detections(self, semantic_mask):
        """Genera bounding boxes simulate semplici"""
        detections = []
        
        for category, value in self.category_values.items():
            if value == 0 or category == 'unknown':
                continue
                
            # Trova pixel di questa categoria
            category_pixels = np.where(semantic_mask == value)
            
            if len(category_pixels[0]) > 500:  # Solo regioni significative
                # Calcola bounding box
                y_min, y_max = category_pixels[0].min(), category_pixels[0].max()
                x_min, x_max = category_pixels[1].min(), category_pixels[1].max()
                
                area = len(category_pixels[0])
                confidence = min(0.9, area / (semantic_mask.size * 0.1))
                
                detections.append({
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': confidence,
                    'class_name': category,
                    'class_id': value,
                    'area': area
                })
        return detections