#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERATE V5 - APPROCCIO SUPER SEMPLICE
Usa solo computer vision di base per identificare ostacoli
"""

import airsim
import os
import time
import random
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# Parametri configurazione
IMG_SIZE = (224, 224)
DATASET_VERSION = "v5"
N_SAMPLES = 100  # Ridotto per test
CAPTURE_INTERVAL = 2
MIN_ALTITUDE = -8
MAX_ALTITUDE = 0

class SimpleDatasetGeneratorV5:
    def __init__(self):
        self.dataset_dir = os.path.join(os.path.dirname(__file__), "..", f"dataset_{DATASET_VERSION}")
        self.backgrounds_dir = os.path.join(os.path.dirname(__file__), "..", "backgrounds")
        self.client = None
        print(f"📁 Directory dataset: {self.dataset_dir}")
        
    def setup_directories(self):
        """Crea le directory necessarie"""
        os.makedirs(self.dataset_dir, exist_ok=True)
        
    def connect_airsim(self):
        """Connessione ad AirSim"""
        print("🔌 Connessione ad AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("✅ Connesso ad AirSim")
        
    def takeoff_and_setup(self):
        """Decollo e setup iniziale"""
        print("🚁 Decollo del drone...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(1)
        
    def simple_obstacle_detection(self, img):
        """
        APPROCCIO SUPER SEMPLICE: usa HSV e edge detection
        """
        # Converti PIL in OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 1. Converti in HSV per identificare meglio il cielo
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # 2. Maschera per il cielo (tonalità blu/cyan, alta saturazione)
        # Cielo tipicamente ha H tra 90-130, S bassa-alta, V alta
        lower_sky = np.array([90, 30, 100])   # Blu/cyan chiaro
        upper_sky = np.array([130, 255, 255]) # Blu/cyan scuro
        sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)
        
        # 3. Maschera per il ground (tonalità marroni/verdi, saturazione varia)
        # Ground tipicamente ha H tra 10-60 (marroni/verdi)
        lower_ground = np.array([10, 20, 20])
        upper_ground = np.array([60, 255, 180])
        ground_mask = cv2.inRange(hsv, lower_ground, upper_ground)
        
        # 4. Combina cielo e ground
        background_mask = cv2.bitwise_or(sky_mask, ground_mask)
        
        # 5. Dilata la maschera di background per catturare bordi sfumati
        kernel = np.ones((5,5), np.uint8)
        background_mask = cv2.dilate(background_mask, kernel, iterations=2)
        
        # 6. Gli ostacoli sono tutto ciò che NON è background
        obstacle_mask = cv2.bitwise_not(background_mask)
        
        # 7. Rimuovi rumore con operazioni morfologiche
        kernel_small = np.ones((3,3), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel_small)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel_small)
        
        # 8. Converti in PIL
        mask_pil = Image.fromarray(obstacle_mask).resize(IMG_SIZE, Image.NEAREST)
        
        # Debug
        obstacle_pixels = np.sum(obstacle_mask > 0)
        total_pixels = obstacle_mask.size
        percentage = (obstacle_pixels / total_pixels) * 100
        print(f"[DEBUG] Ostacoli rilevati: {percentage:.1f}% dell'immagine")
        
        return mask_pil
        
    def capture_scene_and_mask(self, anchor_idx=None):
        """Cattura scena e genera maschera con computer vision"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
        ])
        
        if not responses or len(responses[0].image_data_uint8) == 0:
            print("[DEBUG] Errore: immagine scene vuota")
            return None, None
            
        # Processa immagine scene
        img_bytes = responses[0].image_data_uint8
        scene_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        scene_img = scene_img.resize(IMG_SIZE, Image.LANCZOS)
        
        # Genera maschera ostacoli
        mask = self.simple_obstacle_detection(scene_img)
        
        return scene_img, mask
        
    def generate_positives(self, anchor_img, mask, num_positives=6):
        """Genera immagini positive con sfondi diversi"""
        bg_files = [f for f in os.listdir(self.backgrounds_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(bg_files) == 0:
            print("[WARNING] Nessuno sfondo trovato, uso sfondi colorati")
            # Genera sfondi colorati semplici
            colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
            bg_files = colors[:num_positives]
        
        positives = []
        chosen_bgs = random.sample(bg_files, min(num_positives, len(bg_files)))
        
        for bg in chosen_bgs:
            if isinstance(bg, tuple):  # Colore generato
                bg_img = Image.new("RGB", IMG_SIZE, bg)
            else:  # File esistente
                bg_img = Image.open(os.path.join(self.backgrounds_dir, bg)).convert("RGB")
                bg_img = bg_img.resize(IMG_SIZE, Image.LANCZOS)
            
            # Componi: mantieni solo ostacoli
            anchor_rgba = anchor_img.convert("RGBA")
            bg_rgba = bg_img.convert("RGBA")
            
            # Usa maschera come alpha
            mask_array = np.array(mask)
            alpha = Image.fromarray(mask_array).convert('L')
            
            anchor_rgba.putalpha(alpha)
            result = Image.alpha_composite(bg_rgba, anchor_rgba).convert("RGB")
            positives.append(result)
            
        return positives
        
    def save_anchor_set(self, anchor_img, positives, mask, idx):
        """Salva anchor e positivi con debug"""
        anchor_dir = os.path.join(self.dataset_dir, f"anchor_{idx:05d}")
        os.makedirs(anchor_dir, exist_ok=True)
        
        # Salva anchor
        anchor_img.save(os.path.join(anchor_dir, "anchor.png"))
        
        # Salva maschera per debug
        mask.save(os.path.join(anchor_dir, "mask_debug.png"))
        
        # Salva positivi
        for i, pos_img in enumerate(positives, start=1):
            pos_img.save(os.path.join(anchor_dir, f"positive_{i}.png"))
            
        print(f"💾 Salvato anchor_{idx:05d} con {len(positives)} positivi")
        
    def move_drone_randomly(self):
        """Movimento casuale del drone"""
        state = self.client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val
        
        # Controlla altitudine
        if current_alt > MAX_ALTITUDE:
            vz = -1.0
        elif current_alt < MIN_ALTITUDE:
            vz = 1.0
        else:
            vz = random.uniform(-0.5, 0.5)
            
        # Movimento orizzontale e rotazione
        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        yaw_rate = random.uniform(-45, 45)
        
        self.client.moveByVelocityAsync(
            vx, vy, vz, CAPTURE_INTERVAL,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        )
        
    def generate_dataset(self):
        """Generazione del dataset"""
        print("🚀 GENERATE V5 - APPROCCIO SEMPLIFICATO")
        print(f"📊 Generazione {N_SAMPLES} samples")
        print(f"📍 Altitudine: {MIN_ALTITUDE}m - {MAX_ALTITUDE}m")
        print("-" * 50)
        
        self.setup_directories()
        self.connect_airsim()
        self.takeoff_and_setup()
        
        successful_captures = 0
        
        for i in range(N_SAMPLES):
            try:
                # Movimento
                self.move_drone_randomly()
                
                # Cattura con computer vision
                anchor_img, mask = self.capture_scene_and_mask(successful_captures)
                
                if anchor_img and mask:
                    # Genera positivi
                    positives = self.generate_positives(anchor_img, mask)
                    
                    # Salva
                    self.save_anchor_set(anchor_img, positives, mask, successful_captures)
                    successful_captures += 1
                    
                    print(f"📸 Cattura {i+1}/{N_SAMPLES} completata")
                else:
                    print(f"⚠️ Cattura {i+1} fallita - riprovando...")
                    
                time.sleep(CAPTURE_INTERVAL)
                
            except KeyboardInterrupt:
                print("⏹️ Generazione interrotta dall'utente")
                break
            except Exception as e:
                print(f"❌ Errore cattura {i+1}: {e}")
                continue
                
        # Atterraggio
        print("🛬 Atterraggio...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
        print(f"✅ Dataset V5 generato! {successful_captures} anchor salvati")

def main():
    print("🔥 DATASET GENERATOR V5 - SEMPLICE E DIRETTO")
    generator = SimpleDatasetGeneratorV5()
    generator.generate_dataset()

if __name__ == "__main__":
    main()