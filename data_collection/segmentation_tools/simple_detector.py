"""
Object Detector SEMPLIFICATO per AirSim
Approccio minimale e diretto senza complessità eccessive
"""

import numpy as np
from PIL import Image

class SimpleObjectDetector:
    def __init__(self):
        self.category_values = {
            'sky': 5,
            'trees': 50, 
            'buildings': 90,
            'ground': 130
        }
        print("✅ Simple Object Detector inizializzato")
    
    def process_airsim_image(self, airsim_image):
        """Processamento SEMPLIFICATO dell'immagine AirSim"""
        try:
            # Converti in array numpy
            if isinstance(airsim_image, Image.Image):
                img_array = np.array(airsim_image)
            else:
                img_array = airsim_image
            
            h, w = img_array.shape[:2]
            print(f"🔍 Processando immagine {w}x{h}")
            
            # Analisi SEMPLICE basata solo su colori e posizione
            semantic_mask = self.simple_segmentation(img_array)
            
            # Statistiche
            stats = self.calculate_statistics(semantic_mask)
            
            result = {
                'semantic_mask': semantic_mask,
                'detections': [],
                'statistics': stats,
                'category_mapping': {v: k for k, v in self.category_values.items()},
                'debug_info': {
                    'processing_method': 'simple_color_position',
                    'categories_detected': len([k for k, v in stats.items() if k > 0 and v > 100])
                }
            }
            
            print(f"✅ Processamento completato: {len(stats)} categorie rilevate")
            return result
            
        except Exception as e:
            print(f"❌ Errore processing image: {e}")
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
    
    def simple_segmentation(self, img_array):
        """Segmentazione SEMPLIFICATA - logica chiara e diretta"""
        h, w = img_array.shape[:2]
        
        # Estrai canali RGB
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32) 
        b = img_array[:, :, 2].astype(np.float32)
        
        # Crea griglia posizioni
        y_pos = np.arange(h).reshape(-1, 1) / h  # 0 = top, 1 = bottom
        x_pos = np.arange(w).reshape(1, -1) / w   # 0 = left, 1 = right
        
        # Inizializza maschera
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. CIELO: Blu dominante + posizione alta
        is_blue = (b > r + 20) & (b > g + 15) & (b > 100)
        is_top_area = y_pos < 0.7  # Solo nel 70% superiore
        sky_pixels = is_blue & is_top_area
        mask[sky_pixels] = self.category_values['sky']
        
        # 2. VEGETAZIONE: Verde dominante
        is_green = (g > r + 10) & (g > b + 10) & (g > 60) & (mask == 0)
        mask[is_green] = self.category_values['trees']
        
        # 3. EDIFICI: Colori neutri/grigi + non cielo + non vegetazione
        brightness = (r + g + b) / 3
        is_neutral = (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & (np.abs(r - b) < 30)
        is_medium_bright = (brightness > 80) & (brightness < 200)
        is_building = is_neutral & is_medium_bright & (mask == 0)
        mask[is_building] = self.category_values['buildings']
        
        # 4. TERRENO: Tutto il resto che è scuro + posizione bassa
        is_bottom_area = y_pos > 0.6  # Solo nel 40% inferiore
        is_dark = brightness < 100
        is_ground = is_dark & is_bottom_area & (mask == 0)
        mask[is_ground] = self.category_values['ground']
        
        return mask
    
    def calculate_statistics(self, semantic_mask):
        """Calcola statistiche semplici"""
        unique_values, counts = np.unique(semantic_mask, return_counts=True)
        stats = dict(zip(unique_values, counts))
        
        total_pixels = semantic_mask.size
        
        print("📊 STATISTICHE SEGMENTAZIONE:")
        for value, count in stats.items():
            category = next((k for k, v in self.category_values.items() if v == value), 'unknown')
            percentage = (count / total_pixels) * 100
            print(f"   {category.upper()}: {percentage:.1f}% ({count} pixel)")
        
        # Validazione semplice
        unknown_ratio = stats.get(0, 0) / total_pixels
        categorized_ratio = 1.0 - unknown_ratio
        
        if unknown_ratio > 0.9:
            print("⚠️ WARNING: Troppi pixel non categorizzati")
        elif unknown_ratio < 0.1:
            print("⚠️ WARNING: Forse troppo aggressivo")
        else:
            print(f"✅ Segmentazione OK ({categorized_ratio:.1%} categorizzato)")
        
        return stats