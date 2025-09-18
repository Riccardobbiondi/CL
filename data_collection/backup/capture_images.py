# Funzione semplice: usa la maschera di AirSim così com'è, binarizzando tutto ciò che non è nero
def get_segmentation_mask(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)
    ])
    if responses and len(responses) > 0:
        img_bytes = responses[0].image_data_uint8
        if len(img_bytes) == 0:
            print("[DEBUG] Nessun dato nella maschera di segmentazione.")
            return None
        mask = Image.open(BytesIO(img_bytes)).convert("L").resize(IMG_SIZE, Image.NEAREST)
        # Salva la maschera grezza per debug
        mask.save("debug_mask.png")
        mask_np = np.array(mask)
        print(f"[DEBUG] Valori unici nella maschera: {np.unique(mask_np)}")
        # Binarizza: considera ostacolo tutto ciò che supera una soglia (es. 10)
        mask_bin = mask.point(lambda x: 255 if x > 10 else 0)
        return mask_bin
    print("[DEBUG] Nessuna risposta da AirSim per la segmentazione.")
    return None
import airsim
import os
import sys
import time
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from io import BytesIO

# Parametri dataset
IMG_SIZE = (224, 224)  # dimensione immagini
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset_v4")
n_env = 0  # numero di ambienti già processati
offset = 2500 * n_env  # offset per evitare conflitti con immagini esistenti
N_SAMPLES = 2500             # numero di triplette da salvare
CAPTURE_INTERVAL = 1      # secondi tra uno scatto e l'altro


def save_triplet(anchor, positive, negative, idx):
    anchor.save(os.path.join(SAVE_DIR, "anchor", f"img_{idx:04d}.png"))
    positive.save(os.path.join(SAVE_DIR, "positive", f"img_{idx:04d}.png"))
    negative.save(os.path.join(SAVE_DIR, "negative", f"img_{idx:04d}.png"))

def save_anchor_and_positives(anchor, positives, idx):
    anchor_dir = os.path.join(SAVE_DIR, f"anchor_{idx:04d}")
    print(f"[DEBUG] SAVE_DIR: {SAVE_DIR}")
    try:
        os.makedirs(anchor_dir, exist_ok=True)
    except Exception as e:
        print(f"Errore nella creazione della cartella {anchor_dir}: {e}")
    anchor_path = os.path.join(anchor_dir, "anchor.png")
    try:
        print(f"Salvataggio anchor: {anchor_path} (in SAVE_DIR: {SAVE_DIR})")
        anchor.save(anchor_path)
    except Exception as e:
        print(f"Errore nel salvataggio anchor: {e}")
    for i, pos_img in enumerate(positives, start=1):
        pos_path = os.path.join(anchor_dir, f"positive_{i}.png")
        try:
            print(f"Salvataggio positive: {pos_path} (in SAVE_DIR: {SAVE_DIR})")
            pos_img.save(pos_path)
        except Exception as e:
            print(f"Errore nel salvataggio positive {i}: {e}")

def crop_center(img, crop_size):
    w, h = img.size
    cw, ch = crop_size
    left = (w - cw) // 2
    top = (h - ch) // 2
    right = left + cw
    bottom = top + ch
    return img.crop((left, top, right, bottom))

def capture_image(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, True)
    ])
    if responses and len(responses) > 0:
        img_bytes = responses[0].image_data_uint8
        if len(img_bytes) == 0:
            return None
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img.resize(IMG_SIZE, Image.LANCZOS)
    return None

def main():
    # Connessione ad AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    time.sleep(1)

    target_alt = random.uniform(-3, -8)

    for i in range(offset, N_SAMPLES + offset):
        state = client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val

        if current_alt > -3:
            vz = -1.0
        elif current_alt < -8:
            vz = 1.0
        else:
            vz = random.uniform(-0.5, 0.5)

        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        yaw_rate = random.uniform(-45, 45)

        client.moveByVelocityAsync(vx, vy, vz, CAPTURE_INTERVAL,
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))


        img_anchor = capture_image(client)
        mask = None
        # Prova a ottenere la maschera di segmentazione (se esiste la funzione nel file)
        if 'get_segmentation_mask' in globals():
            try:
                mask = get_segmentation_mask(client)
            except Exception as e:
                print(f"Errore nel calcolo della maschera: {e}")

        if img_anchor and mask is not None:
            backgrounds_dir = os.path.join(os.path.dirname(__file__), "..", "backgrounds")
            bg_files = [f for f in os.listdir(backgrounds_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
            while len(chosen_bgs) < 6:
                chosen_bgs.append(random.choice(bg_files))

            positives = []
            for bg_file in chosen_bgs[:6]:
                bg_img = Image.open(os.path.join(backgrounds_dir, bg_file)).convert("RGB")
                # Componi: mantieni solo ostacoli usando la maschera
                anchor_rgba = img_anchor.resize(IMG_SIZE).convert("RGBA")
                bg_rgba = bg_img.resize(IMG_SIZE).convert("RGBA")
                mask_fixed = mask.resize(IMG_SIZE, Image.NEAREST).convert('L')
                alpha = mask_fixed.point(lambda x: 255 if x > 128 else 0)
                print(f"[DEBUG] anchor_rgba size: {anchor_rgba.size}, mode: {anchor_rgba.mode}")
                print(f"[DEBUG] bg_rgba size: {bg_rgba.size}, mode: {bg_rgba.mode}")
                print(f"[DEBUG] alpha size: {alpha.size}, mode: {alpha.mode}")
                try:
                    anchor_rgba.putalpha(alpha)
                except Exception as e:
                    print(f"[ERROR] putalpha failed: {e}")
                try:
                    result = Image.alpha_composite(bg_rgba, anchor_rgba).convert("RGB")
                    positives.append(result)
                except Exception as e:
                    print(f"[ERROR] alpha_composite failed: {e}")
            save_anchor_and_positives(img_anchor, positives, i)

        print(f"[{i+1}/{N_SAMPLES+offset}] anchor e positivi salvati (altitudine {current_alt:.2f} m)")
        time.sleep(CAPTURE_INTERVAL)

    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Raccolta completata!")

if __name__ == "__main__":
    main()
