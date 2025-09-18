import airsim
import cv2
import numpy as np
import os
import random
from PIL import Image
import time

# --- CONFIGURAZIONE PRINCIPALE ---
# !!! MODIFICA QUESTO VALORE PER OGNI AMBIENTE !!!
ENV_ID = 3

DATASET_PATH = "../dataset_final"
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
    """Calibrazione automatica per identificare gli ID di cielo e terreno."""
    print("Inizio calibrazione ambiente...")
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(-np.pi/2, 0, 0)))
    time.sleep(1)
    _, sky_mask = get_synchronized_images()
    sky_id = find_most_frequent_color(sky_mask)

    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(np.pi/2, 0, 0)))
    time.sleep(1)
    _, ground_mask = get_synchronized_images()
    ground_id = find_most_frequent_color(ground_mask)
    
    client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)))
    time.sleep(1)
    
    if sky_id is None or ground_id is None: raise ConnectionError("Calibrazione fallita.")
    print(f"Calibrazione completata: Sky ID={sky_id}, Ground ID={ground_id}")
    return sky_id, ground_id

def create_obstacle_mask(seg_mask, sky_id, ground_id):
    """Crea una maschera binaria che isola gli ostacoli."""
    if seg_mask is None: return None
    sky_condition = np.all(seg_mask == sky_id, axis=-1)
    ground_condition = np.all(seg_mask == ground_id, axis=-1)
    background_condition = np.logical_or(sky_condition, ground_condition)
    obstacle_mask = np.logical_not(background_condition)
    obstacle_mask = (obstacle_mask * 255).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return obstacle_mask

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

    sky_id, ground_id = calibrate_environment()

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

            obstacle_mask = create_obstacle_mask(seg_mask, sky_id, ground_id)
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