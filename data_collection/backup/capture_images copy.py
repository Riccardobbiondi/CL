import airsim
import os
import sys
import time
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# Parametri dataset
IMG_SIZE = (224, 224)  # dimensione immagini
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
offset = 0  # offset per evitare conflitti con immagini esistenti
N_SAMPLES = 5000             # numero di triplette da salvare
CAPTURE_INTERVAL = 0.5      # secondi tra uno scatto e l'altro

# Funzioni di trasformazione
def transform_image(img, type='positive'):
    if type == 'positive':
        img = img.rotate(random.uniform(-5, 5))  # rotazione leggera
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
    elif type == 'negative':
        img = img.rotate(random.uniform(20, 45))  # rotazione più grande
        img = ImageOps.autocontrast(img)
    return img

def save_triplet(anchor, positive, negative, idx):
    anchor.save(os.path.join(SAVE_DIR, "anchor", f"img_{idx:04d}.png"))
    positive.save(os.path.join(SAVE_DIR, "positive", f"img_{idx:04d}.png"))
    negative.save(os.path.join(SAVE_DIR, "negative", f"img_{idx:04d}.png"))

def capture_image(client):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ])
    if responses and len(responses) > 0:
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
        return Image.fromarray(img_rgb).resize(IMG_SIZE)
    return None

def main():
    # Creazione cartelle
    os.makedirs(os.path.join(SAVE_DIR, "anchor"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "positive"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "negative"), exist_ok=True)

    # Connessione ad AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    # Decollo
    client.takeoffAsync().join()
    time.sleep(1)

    # Imposta altitudine target iniziale (tra -3 e -8 m rispetto al livello di partenza)
    target_alt = random.uniform(-3, -8)

    for i in range(offset, N_SAMPLES + offset):
        # Ottieni stato attuale
        state = client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val  # negativo = in basso

        # Se troppo alto o troppo basso, inverti direzione (bounce)
        if current_alt > -3:   # troppo vicino al suolo
            vz = -1.0
        elif current_alt < -8:  # troppo in basso
            vz = 1.0
        else:
            vz = random.uniform(-0.5, 0.5)

        # Movimento casuale orizzontale
        vx = random.uniform(-3, 3)
        vy = random.uniform(-3, 3)
        yaw_rate = random.uniform(-45, 45)

        client.moveByVelocityAsync(vx, vy, vz, CAPTURE_INTERVAL,
                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                   yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate))

        # Cattura immagine
        img_anchor = capture_image(client)
        if img_anchor:
            img_positive = transform_image(img_anchor.copy(), 'positive')
            img_negative = transform_image(img_anchor.copy(), 'negative')
            save_triplet(img_anchor, img_positive, img_negative, i)

        print(f"[{i+1}/{N_SAMPLES+offset}] tripla salvata (altitudine {current_alt:.2f} m)")
        time.sleep(CAPTURE_INTERVAL)


    # Atterraggio e disconnessione
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Raccolta completata!")

if __name__ == "__main__":
    main()
