import airsim
import cv2
import numpy as np
import os
import random
from PIL import Image
import time
import csv
import argparse # Aggiunto per gli argomenti da riga di comando

# --- CONFIGURAZIONE PRINCIPALE ---
# !!! MODIFICA QUESTO VALORE PER OGNI AMBIENTE !!!
ENV_ID = 4
ENVIRONMENT_NAME = "AirsimNH" # !!! MODIFICA QUESTO VALORE CON IL NOME DELL'AMBIENTE !!!

DATASET_PATH = "../dataset_plus"
NUM_ANCHORS_PER_RUN = 2500
CAPTURE_INTERVAL = 1.0 # Scatta una foto ogni secondo

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 144
BACKGROUNDS_PATH = "../backgrounds"

# --- CONFIGURAZIONE MOVIMENTO ---
MAX_HORIZONTAL_SPEED = 3.0
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

def get_environment_name(client):
    """Estrae il nome della scena corrente in modo robusto."""
    try:
        # Prova a ottenere il nome dell'ambiente direttamente dalle impostazioni
        # Questo è spesso il metodo più affidabile
        settings_str = client.simGetSettingsString()
        if settings_str:
            # Cerca una corrispondenza per "LevelName" o simili
            # Questo è un approccio euristico e potrebbe necessitare di aggiustamenti
            import re
            match = re.search(r'"LevelName"\s*:\s*"([^"]+)"', settings_str)
            if match:
                level_path = match.group(1)
                # Estrai solo il nome del file, senza percorso o estensione
                level_name = os.path.basename(level_path).split('.')[0]
                if level_name:
                    return level_name

        # Fallback: prova con getSceneString(), che a volte funziona
        scene_name = client.getSceneString()
        if scene_name:
            return scene_name
            
    except Exception as e:
        print(f"Non è stato possibile determinare il nome dell'ambiente in modo dinamico: {e}")
    
    # Se tutti i tentativi falliscono, ritorna un valore di default
    # per garantire la coerenza dei dati.
    return "DefaultEnv"

def get_privileged_data(client):
    """Recupera dati privilegiati sullo stato del drone."""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    orientation = state.kinematics_estimated.orientation
    linear_velocity = state.kinematics_estimated.linear_velocity
    angular_velocity = state.kinematics_estimated.angular_velocity
    collision = state.collision.has_collided

    return [
        pos.x_val, pos.y_val, pos.z_val,
        orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val,
        linear_velocity.x_val, linear_velocity.y_val, linear_velocity.z_val,
        angular_velocity.x_val, angular_velocity.y_val, angular_velocity.z_val,
        collision
    ]

# --- Logica Principale ---
def main():
    # --- Parsing degli argomenti da riga di comando ---
    parser = argparse.ArgumentParser(description="Raccoglie dati di training da AirSim.")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Forza un nome specifico per l'ambiente.")
    args = parser.parse_args()

    # Setup
    print("Abilitazione controllo API e decollo...")
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    client.moveToZAsync(TAKEOFF_ALTITUDE, 5).join()
    time.sleep(1)

    sky_id, ground_id = calibrate_environment()
    
    env_name = ENVIRONMENT_NAME

    # Preparazione sfondi
    all_bgs = [os.path.join(BACKGROUNDS_PATH, f) for f in os.listdir(BACKGROUNDS_PATH) if f.endswith(('.png', '.jpg'))]
    indoor_bgs = [p for p in all_bgs if 'ind_' in p]
    outdoor_bgs = [p for p in all_bgs if 'b_' in p]
    black_bg = os.path.join(BACKGROUNDS_PATH, 'black.png')
    white_bg = os.path.join(BACKGROUNDS_PATH, 'white.png')
    if not all([os.path.exists(black_bg), os.path.exists(white_bg), indoor_bgs, outdoor_bgs]):
        raise FileNotFoundError("File di sfondo mancanti.")

    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # --- Setup del file CSV per i dati privilegiati ---
    CSV_PATH = os.path.join(DATASET_PATH, "privileged_data.csv")
    csv_header = [
        "anchor_id", "env_name",
        "pos_x", "pos_y", "pos_z",
        "q_w", "q_x", "q_y", "q_z",
        "vel_x", "vel_y", "vel_z",
        "ang_vel_x", "ang_vel_y", "ang_vel_z",
        "has_collided"
    ]
    
    file_exists = os.path.isfile(CSV_PATH)
    csv_file = open(CSV_PATH, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not file_exists:
        csv_writer.writerow(csv_header)
        print(f"Creato file CSV per dati privilegiati in: {CSV_PATH}")

    start_anchor_index = ENV_ID * NUM_ANCHORS_PER_RUN
    total_duration = NUM_ANCHORS_PER_RUN * CAPTURE_INTERVAL
    
    print(f"Inizio raccolta dati per l'ambiente {ENV_ID} per circa {total_duration} secondi.")
    print(f"Verranno generate {NUM_ANCHORS_PER_RUN} ancore, da indice {start_anchor_index}.")

    start_time = time.time()
    last_capture_time = 0
    anchor_count = 0

    # Loop di movimento e cattura
    while anchor_count < NUM_ANCHORS_PER_RUN:
        # Controllo altitudine e impostazione velocità verticale
        current_pos = client.getMultirotorState().kinematics_estimated.position
        vz = random.uniform(-MAX_VERTICAL_SPEED, MAX_VERTICAL_SPEED)
        if current_pos.z_val < MAX_ALTITUDE: # Troppo alto (z è più negativo)
            vz = abs(vz) # Forza la discesa
        elif current_pos.z_val > TAKEOFF_ALTITUDE: # Troppo basso
            vz = -abs(vz) # Forza la salita

        # Imposta velocità orizzontale e muovi
        vx = random.uniform(-MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
        vy = random.uniform(-MAX_HORIZONTAL_SPEED, MAX_HORIZONTAL_SPEED)
        client.moveByVelocityAsync(vx, vy, vz, duration=0.5).join()

        # Cattura a intervalli
        if (time.time() - last_capture_time) >= CAPTURE_INTERVAL:
            last_capture_time = time.time()
            current_anchor_index = start_anchor_index + anchor_count
            
            print(f"\n--- Cattura Ancora {anchor_count + 1}/{NUM_ANCHORS_PER_RUN} (Indice: {current_anchor_index}) ---")
            
            # Recupera immagini e dati privilegiati
            anchor_img, seg_mask = get_synchronized_images()
            privileged_data = get_privileged_data(client)

            if anchor_img is None or seg_mask is None:
                print("Immagine non valida, salto.")
                continue

            # Scrivi i dati privilegiati nel CSV
            csv_writer.writerow([current_anchor_index, env_name] + privileged_data)

            anchor_folder = os.path.join(DATASET_PATH, f"anchor_{current_anchor_index:06d}")
            os.makedirs(anchor_folder, exist_ok=True)
            Image.fromarray(anchor_img).save(os.path.join(anchor_folder, "anchor.png"))

            obstacle_mask = create_obstacle_mask(seg_mask, sky_id, ground_id)
            if obstacle_mask is None:
                print("Maschera non valida, salto.")
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
    
    # Fine
    csv_file.close() # Chiudi il file CSV
    print("\nGenerazione completata. Atterraggio...")
    client.hoverAsync().join()
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Drone a terra e disarmato.")

if __name__ == "__main__":
    main()
