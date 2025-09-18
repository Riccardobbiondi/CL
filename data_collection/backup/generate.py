#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script SEMPLIFICATO per la generazione del dataset usando AirSim
Basato sulla logica di capture_images.py - PULITO E DIRETTO
"""

import airsim
import os
import time
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO
import cv2
from sklearn.cluster import KMeans

# Parametri configurazione
IMG_SIZE = (224, 224)
DATASET_VERSION = "v4"
N_SAMPLES = 2500
CAPTURE_INTERVAL = 2
MIN_ALTITUDE = -8
MAX_ALTITUDE = 0

class SimpleDatasetGenerator:
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
        
        # Posiziona a quota casuale
        target_alt = random.uniform(MIN_ALTITUDE, MAX_ALTITUDE)
        state = self.client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val
        print(f"✅ Drone posizionato ad altitudine: {current_alt:.2f}m")
        
    def capture_scene_and_mask(self, anchor_idx=None):
        """Cattura SOLO la scena e usa OpenCV per detection ostacoli"""
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
        ])
        
        if len(responses) != 1:
            print("[DEBUG] Errore: non ho ricevuto l'immagine scene")
            return None, None
            
        # Processa immagine scene
        scene_response = responses[0]
        if not scene_response.image_data_uint8 or len(scene_response.image_data_uint8) == 0:
            print("[DEBUG] Errore: immagine scene vuota")
            return None, None
            
        scene_img = Image.open(BytesIO(scene_response.image_data_uint8)).convert("RGB")
        scene_img = scene_img.resize(IMG_SIZE, Image.LANCZOS)
        
        # USA IL NUOVO APPROCCIO SEMPLICE
        mask_img = self.simple_obstacle_detection(scene_img)
        
        return scene_img, mask_img
        print(f"[DEBUG] Min: {mask_np.min()}, Max: {mask_np.max()}")
        
        # Mostra distribuzione dei valori
        values, counts = np.unique(mask_np, return_counts=True)
        for val, count in zip(values, counts):
            percentage = count / mask_np.size * 100
            print(f"[DEBUG] Valore {val}: {count} pixel ({percentage:.1f}%)")
        
    def simple_obstacle_detection(self, scene_img):
        """APPROCCIO SEMPLICE: usa OpenCV per trovare ostacoli automaticamente"""
        
        # Converti in array numpy
        img_np = np.array(scene_img)
        
        # 1. EDGE DETECTION: trova i bordi degli edifici
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 2. CLUSTERING COLORI: separa cielo/ground dagli ostacoli
        h, w, c = img_np.shape
        img_flat = img_np.reshape((-1, 3))
        
        # K-means per trovare 3 cluster principali (cielo, ground, ostacoli)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(img_flat)
        labels = labels.reshape((h, w))
        
        # 3. ANALISI SPAZIALE: identifica quale cluster è cosa
        cluster_analysis = []
        for cluster_id in range(3):
            mask = (labels == cluster_id)
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) > 0:
                avg_y = np.mean(y_coords)
                top_ratio = np.sum(y_coords < h * 0.3) / len(y_coords)
                bottom_ratio = np.sum(y_coords > h * 0.7) / len(y_coords)
                
                # Luminosità media del cluster
                cluster_pixels = img_flat[labels.flatten() == cluster_id]
                avg_brightness = np.mean(cluster_pixels)
                
                cluster_analysis.append({
                    'id': cluster_id,
                    'avg_y': avg_y,
                    'top_ratio': top_ratio,
                    'bottom_ratio': bottom_ratio,
                    'brightness': avg_brightness,
                    'size': len(y_coords)
                })
        
        # Identifica cielo (più luminoso + in alto) e ground (in basso)
        cluster_analysis.sort(key=lambda x: x['brightness'], reverse=True)
        sky_cluster = cluster_analysis[0]['id']  # Più luminoso = cielo
        
        cluster_analysis.sort(key=lambda x: x['bottom_ratio'], reverse=True)
        ground_cluster = cluster_analysis[0]['id']  # Più in basso = ground
        
        # Se sky e ground sono uguali, prendi il secondo più luminoso come cielo
        if sky_cluster == ground_cluster:
            cluster_analysis.sort(key=lambda x: x['brightness'], reverse=True)
            sky_cluster = cluster_analysis[1]['id']
        
        print(f"[DEBUG] Sky cluster: {sky_cluster}, Ground cluster: {ground_cluster}")
        
        # 4. CREA MASCHERA: tutto tranne cielo e ground = ostacoli
        obstacle_mask = np.ones((h, w), dtype=np.uint8) * 255
        obstacle_mask[labels == sky_cluster] = 0      # Cielo = nero
        obstacle_mask[labels == ground_cluster] = 0   # Ground = nero
        
        # 5. MORPHOLOGICAL OPERATIONS: pulisci la maschera
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        
        # 6. COMBINA CON EDGE DETECTION: rinforza i bordi degli edifici
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        obstacle_mask = cv2.bitwise_or(obstacle_mask, edges_dilated)
        
        # Converti in PIL Image
        mask_pil = Image.fromarray(obstacle_mask)
        
        # Debug
        obstacle_pixels = np.sum(obstacle_mask == 255)
        total_pixels = h * w
        print(f"[DEBUG] Ostacoli: {obstacle_pixels} pixel ({obstacle_pixels/total_pixels*100:.1f}%)")
        
        return mask_pil
        
    def generate_positives(self, anchor_img, mask, num_positives=6):
        """Genera immagini positive con sfondi diversi"""
        # Lista sfondi disponibili
        bg_files = [f for f in os.listdir(self.backgrounds_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Seleziona sfondi diversificati
        ind_files = [f for f in bg_files if f.startswith('ind_')]
        black_white = [f for f in bg_files if f in ['black.png', 'white.png']]
        other_bgs = [f for f in bg_files if f not in ind_files + black_white]
        
        chosen_ind = random.choice(ind_files) if ind_files else None
        chosen_bw = black_white
        exclude = set([chosen_ind] + chosen_bw if chosen_ind else chosen_bw)
        available_others = [f for f in other_bgs if f not in exclude]
        chosen_others = random.sample(available_others, min(3, len(available_others)))
        
        chosen_bgs = []
        if chosen_ind:
            chosen_bgs.append(chosen_ind)
        chosen_bgs.extend(chosen_bw)
        chosen_bgs.extend(chosen_others)
        
        # Completa fino a 6 sfondi
        while len(chosen_bgs) < num_positives:
            chosen_bgs.append(random.choice(bg_files))
            
        positives = []
        for bg_file in chosen_bgs[:num_positives]:
            bg_img = Image.open(os.path.join(self.backgrounds_dir, bg_file)).convert("RGB")
            
            # Componi: mantieni solo ostacoli usando la maschera
            anchor_rgba = anchor_img.resize(IMG_SIZE).convert("RGBA")
            bg_rgba = bg_img.resize(IMG_SIZE).convert("RGBA")
            mask_fixed = mask.resize(IMG_SIZE, Image.NEAREST).convert('L')
            alpha = mask_fixed.point(lambda x: 255 if x > 128 else 0)
            
            anchor_rgba.putalpha(alpha)
            result = Image.alpha_composite(bg_rgba, anchor_rgba).convert("RGB")
            positives.append(result)
            
        return positives
        
    def save_anchor_set(self, anchor_img, positives, mask, idx):
        """Salva anchor e positivi"""
        anchor_dir = os.path.join(self.dataset_dir, f"anchor_{idx:05d}")
        os.makedirs(anchor_dir, exist_ok=True)
        
        # Salva anchor
        anchor_path = os.path.join(anchor_dir, "anchor.png")
        anchor_img.save(anchor_path)
        
        # Salva solo la maschera generata da OpenCV
        debug_bin_name = os.path.join(anchor_dir, "obstacle_mask.png")
        mask.save(debug_bin_name)
        
        # Salva positivi
        for i, pos_img in enumerate(positives, start=1):
            pos_path = os.path.join(anchor_dir, f"positive_{i}.png")
            pos_img.save(pos_path)
            
        print(f"💾 Salvato anchor_{idx:05d} con {len(positives)} positivi + maschera ostacoli")
        
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
        print("🚀 Inizio generazione dataset - {} samples".format(N_SAMPLES))
        print("📍 Altitudine: {}m - {}m".format(MIN_ALTITUDE, MAX_ALTITUDE))
        print("⏱️ Intervallo cattura: {}s".format(CAPTURE_INTERVAL))
        print("-" * 50)
        
        self.setup_directories()
        self.connect_airsim()
        self.takeoff_and_setup()
        
        successful_captures = 0
        
        for i in range(N_SAMPLES):
            try:
                # Movimento
                self.move_drone_randomly()
                
                # Cattura SINCRONIZZATA
                anchor_img, mask = self.capture_scene_and_mask(successful_captures)
                
                if anchor_img and mask:
                    # Genera positivi
                    positives = self.generate_positives(anchor_img, mask)
                    
                    # Salva
                    self.save_anchor_set(anchor_img, positives, mask, successful_captures)
                    successful_captures += 1
                    
                    state = self.client.getMultirotorState()
                    current_alt = state.kinematics_estimated.position.z_val
                    print(f"📸 Cattura {i+1}/{N_SAMPLES} - altitudine {current_alt:.2f}m")
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
        
        print(f"✅ Dataset generato! {successful_captures} anchor salvati in {self.dataset_dir}")

def main():
    generator = SimpleDatasetGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main()