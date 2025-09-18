# ==================================================================================================
# SCRIPT PER LA GENERAZIONE DI UN DATASET CONTRASTIVO TRAMITE AIRSIM
# ==================================================================================================
#
# Generare un dataset di immagini per l'addestramento di un modello di contrastive learning,
# finalizzato al riconoscimento robusto di ostacoli in ambienti simulati.
# Implementa un processo automatizzato che si articola nelle seguenti fasi:
#
# 1. CALIBRAZIONE DELL'AMBIENTE:
#    - In fase di inizializzazione, viene eseguita una calibrazione per identificare la
#      "palette" di colori che costituiscono lo sfondo (es. cielo, terreno).
#    - Questa operazione analizza la maschera di segmentazione di AirSim da diverse
#      prospettive per collezionare i valori RGB unici dello sfondo.
#
# 2. NAVIGAZIONE AUTONOMA:
#    - Un agente autonomo controlla il drone, facendolo decollare e muovere verso
#      destinazioni casuali all'interno dell'ambiente per garantire una copertura
#      varia dello scenario.
#
# 3. ACQUISIZIONE E AUGMENTATION:
#    - A intervalli regolari, lo script acquisisce una coppia di immagini: la scena
#      visuale (denominata "ancora") e la corrispondente maschera di segmentazione.
#    - Utilizzando la palette di colori di sfondo, viene generata una maschera binaria
#      per isolare con precisione gli ostacoli presenti nell'immagine.
#    - L'immagine "ancora" viene salvata e, a partire da essa, vengono generate le
#      immagini "positive" attraverso tecniche di data augmentation:
#        - Sostituzione dello sfondo con immagini predefinite (tinta unita, texture).
#        - Applicazione di effetti di illuminazione (es. ombre).
#
# 4. ORGANIZZAZIONE DEL DATASET:
#    - Il dataset viene strutturato in directory separate per ogni campione. Ogni
#      directory contiene l'immagine "ancora" e le relative "positive", facilitando
#      il caricamento in fase di addestramento.
#
# 5. IMMAGINI
# 1 anchor + 6 positive:
# black, white, shadow+indoor+ground MAGENTA, 3x outdoor
#
# PREREQUISITI DI ESECUZIONE:
# - Il simulatore AirSim deve essere in esecuzione con l'ambiente desiderato.
# - La variabile `ENV_ID` deve essere configurata per indicizzare correttamente i dati
#   generati ed evitare sovrascritture.
# ==================================================================================================

import airsim
import cv2
import numpy as np
import os
import random
from PIL import Image
import time

# --- CONFIGURAZIONE PRINCIPALE ---
# !!! MODIFICA QUESTO VALORE PER OGNI AMBIENTE !!!
ENV_ID = 0

DATASET_PATH = "../dataset_wm_final"
NUM_ANCHORS_PER_RUN = 2500
CAPTURE_INTERVAL = 1.0 # Scatta una foto ogni secondo

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144
BACKGROUNDS_PATH = "../backgrounds"

# --- CONFIGURAZIONE MOVIMENTO ---
MAX_HORIZONTAL_SPEED = 5.0 # Aumentata per viaggi più rapidi
MAX_VERTICAL_SPEED = 1.0
TAKEOFF_ALTITUDE = -10 # Altitudine minima (negativa = verso l'alto)
MAX_ALTITUDE = -40   # Altitudine massima

# --- Inizializzazione AirSim ---
client = airsim.MultirotorClient()
client.confirmConnection()

# --- Funzioni Utility ---

def add_shadow_effect(image, factor=0.7):
    """Applica un effetto ombra scurendo l'immagine."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * factor, 0, 255)
    shadowed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return shadowed_image

def process_image_response(response):
    """Converte una singola ImageResponse in un array NumPy."""
    if response.width == 0 or response.height == 0: return None
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    channels = len(img1d) // (response.height * response.width)
    if channels == 3: return img1d.reshape(response.height, response.width, 3)
    elif channels == 4: return img1d.reshape(response.height, response.width, 4)[..., :3]
    else: return None

def get_synchronized_images():
    """Ottiene scena e maschera di segmentazione in una singola chiamata."""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)
    ])
    return process_image_response(responses[0]), process_image_response(responses[1])

def find_most_frequent_color(img):
    """Trova il colore (pixel) più frequente in un'immagine."""
    if img is None: return None
    pixels = img.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    if not unique_colors.any(): return None
    return unique_colors[counts.argmax()]

def calibrate_environment():
    """
    Calibrazione automatica per identificare gli ID del cielo e dello sfondo (terreno, ecc.).
    Cattura i colori presenti nella metà superiore (cielo) e inferiore (terreno/sfondo) della vista.
    """
    print("Inizio calibrazione ambiente...")
    
    # 1. Calibrazione Cielo
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, -10), airsim.to_quaternion(0, 0, 0)))
    time.sleep(1)
    _, seg_mask_sky = get_synchronized_images()
    if seg_mask_sky is None: raise ConnectionError("Calibrazione fallita: impossibile ottenere la maschera di segmentazione.")
    
    sky_colors = np.unique(seg_mask_sky[:IMAGE_HEIGHT // 2, :, :].reshape(-1, 3), axis=0)
    print(f"Trovati {len(sky_colors)} colori per il cielo.")

    # 2. Calibrazione Sfondo (Terreno e altro)
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, -10), airsim.to_quaternion(0.3, 0, 0))) # Leggermente inclinato verso il basso
    time.sleep(1)
    _, seg_mask_ground = get_synchronized_images()
    if seg_mask_ground is None: raise ConnectionError("Calibrazione fallita: impossibile ottenere la maschera di segmentazione.")

    ground_colors = np.unique(seg_mask_ground[IMAGE_HEIGHT // 2:, :, :].reshape(-1, 3), axis=0)
    print(f"Trovati {len(ground_colors)} colori per lo sfondo/terreno.")

    # 3. Resetta la posa e combina i colori
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
    time.sleep(1)
    
    background_colors = np.unique(np.concatenate((sky_colors, ground_colors)), axis=0)
    
    print(f"Calibrazione completata: {len(background_colors)} colori di sfondo totali identificati.")
    return background_colors

def create_obstacle_mask(seg_mask, background_colors):
    """Crea una maschera binaria che isola gli ostacoli, escludendo i colori di sfondo."""
    if seg_mask is None: return None
    
    # Inizializza una maschera di sfondo come 'falsa'
    background_mask = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=bool)
    
    # Itera sui colori di sfondo e aggiorna la maschera
    for color in background_colors:
        background_mask = np.logical_or(background_mask, np.all(seg_mask == color, axis=-1))

    obstacle_mask = np.logical_not(background_mask)
    obstacle_mask = (obstacle_mask * 255).astype(np.uint8)

    # Pulizia morfologica della maschera
    kernel = np.ones((3,3), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return obstacle_mask

def create_ground_mask(seg_mask, ground_id):
    """Crea una maschera binaria che isola il terreno."""
    if seg_mask is None: return None
    ground_condition = np.all(seg_mask == ground_id, axis=-1)
    ground_mask = (ground_condition * 255).astype(np.uint8)
    return ground_mask

def replace_ground_with_magenta(image, ground_mask):
    """Sostituisce il terreno nell'immagine con il colore magenta."""
    if ground_mask is None: return image
    magenta_color = np.array([255, 0, 255], dtype=np.uint8)
    # Crea un'immagine magenta delle stesse dimensioni
    magenta_layer = np.full(image.shape, magenta_color, dtype=np.uint8)
    # Usa la maschera per combinare l'immagine originale e il layer magenta
    # La maschera deve essere espansa a 3 canali per np.where
    mask_3d = ground_mask[..., np.newaxis] > 0
    return np.where(mask_3d, magenta_layer, image)

def apply_background(scene_img, obstacle_mask, background_path):
    """Applica un nuovo sfondo all'immagine."""
    background = Image.open(background_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    scene = Image.fromarray(scene_img)
    mask = Image.fromarray(obstacle_mask)
    return Image.composite(scene, background, mask)

# --- Logica Principale ---
def main():
    # Setup
    print("Abilitazione controllo API e decollo...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(TAKEOFF_ALTITUDE, 5).join()
    time.sleep(1)

    background_colors = calibrate_environment()

    # Preparazione sfondi
    all_bgs = [os.path.join(BACKGROUNDS_PATH, f) for f in os.listdir(BACKGROUNDS_PATH) if f.endswith(('.png', '.jpg'))]
    indoor_bgs = [p for p in all_bgs if 'ind_' in p]
    outdoor_bgs = [p for p in all_bgs if 'b_' in p]
    black_bg = os.path.join(BACKGROUNDS_PATH, 'black.png')
    white_bg = os.path.join(BACKGROUNDS_PATH, 'white.png')
    if not all([os.path.exists(black_bg), os.path.exists(white_bg), indoor_bgs, outdoor_bgs]):
        raise FileNotFoundError("File di sfondo mancanti.")

    os.makedirs(DATASET_PATH, exist_ok=True)
    
    start_anchor_index = ENV_ID * NUM_ANCHORS_PER_RUN
    total_duration = NUM_ANCHORS_PER_RUN * CAPTURE_INTERVAL
    
    print(f"Inizio raccolta dati per l'ambiente {ENV_ID} per circa {total_duration} secondi.")
    print(f"Verranno generate {NUM_ANCHORS_PER_RUN} ancore, da indice {start_anchor_index}.")

    movement_future = None
    last_destination_time = 0
    DESTINATION_CHANGE_INTERVAL = 15 # secondi prima di cambiare destinazione

    anchor_count = 0
    last_capture_time = 0

    # Loop di movimento e cattura
    while anchor_count < NUM_ANCHORS_PER_RUN:
        current_time = time.time()

        # 1. Gestione Movimento: Scegli una nuova destinazione se necessario
        if movement_future is None or (current_time - last_destination_time) > DESTINATION_CHANGE_INTERVAL:
            if movement_future:
                client.cancelLastTask() # Annulla il movimento precedente se non è ancora finito
                print("Timeout destinazione, scelgo un nuovo punto...")

            current_pos = client.getMultirotorState().kinematics_estimated.position
            
            # Scegli una destinazione casuale in un raggio di 50-100 metri
            radius = random.uniform(50, 100)
            angle = random.uniform(0, 2 * np.pi)
            target_x = current_pos.x_val + radius * np.cos(angle)
            target_y = current_pos.y_val + radius * np.sin(angle)
            
            # Scegli un'altitudine casuale nel range consentito
            target_z = random.uniform(TAKEOFF_ALTITUDE, MAX_ALTITUDE)

            print(f"Nuova destinazione: (X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f})")
            movement_future = client.moveToPositionAsync(target_x, target_y, target_z, MAX_HORIZONTAL_SPEED)
            last_destination_time = current_time
            time.sleep(0.1) # Piccola pausa per far iniziare il movimento

        # 2. Cattura a intervalli regolari durante il volo
        if (current_time - last_capture_time) >= CAPTURE_INTERVAL:
            last_capture_time = current_time
            current_anchor_index = start_anchor_index + anchor_count
            
            print(f"\n--- Cattura Ancora {anchor_count + 1}/{NUM_ANCHORS_PER_RUN} (Indice: {current_anchor_index}) ---")
            
            anchor_img, seg_mask = get_synchronized_images()
            if anchor_img is None or seg_mask is None:
                print("Immagine non valida, salto.")
                continue

            anchor_folder = os.path.join(DATASET_PATH, f"anchor_{current_anchor_index:06d}")
            os.makedirs(anchor_folder, exist_ok=True)
            Image.fromarray(anchor_img).save(os.path.join(anchor_folder, "anchor.png"))

            obstacle_mask = create_obstacle_mask(seg_mask, background_colors)
            if obstacle_mask is None or np.all(obstacle_mask == 0):
                print("Maschera non valida o vuota, salto.")
                continue

            # Genera positivi
            apply_background(anchor_img, obstacle_mask, black_bg).save(os.path.join(anchor_folder, "positive_0.png"))
            apply_background(anchor_img, obstacle_mask, white_bg).save(os.path.join(anchor_folder, "positive_1.png"))
            
            shadowed_anchor = add_shadow_effect(anchor_img)
            apply_background(shadowed_anchor, obstacle_mask, random.choice(indoor_bgs)).save(os.path.join(anchor_folder, "positive_2.png"))
            
            random_outdoor_bgs = random.sample(outdoor_bgs, 3)
            for i in range(3):
                apply_background(anchor_img, obstacle_mask, random_outdoor_bgs[i]).save(os.path.join(anchor_folder, f"positive_{3+i}.png"))

            print(f"Ancora salvata in {anchor_folder}")
            anchor_count += 1
        
        time.sleep(0.1) # Loop check per non sovraccaricare la CPU
    
    # Fine
    print("\nGenerazione completata. Atterraggio...")
    client.hoverAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Drone a terra e disarmato.")

if __name__ == "__main__":
    main()