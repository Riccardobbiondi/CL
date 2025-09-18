import airsim
import os
import sys
import time
import random
import numpy as np
from PIL import Image
from io import BytesIO

# Directory containing background images
BACKGROUND_DIR = os.path.join(os.path.dirname(__file__), "..", "backgrounds")

# Parametri dataset
IMG_SIZE = (224, 224)  # dimensione immagini
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset_v3")
n_env = 10  # numero di ambienti già processati
offset = 2500 * n_env  # offset per evitare conflitti con immagini esistenti
N_SAMPLES = 2500             # numero di triplette da salvare
CAPTURE_INTERVAL = 1      # secondi tra uno scatto e l'altro

def get_segmentation_mask(client):
    # Request segmentation image from AirSim
    # Each pixel value in the segmentation image corresponds to an object ID in the scene
    # Typically, obstacles are assigned a specific ID in the AirSim environment
    # Here we keep only trees, buildings, structures, etc. and mask out sky, ground, moving objects
    # Set your allowed IDs here (example: 1=tree, 2=building, 3=structure)
    ALLOWED_IDS = [0, 64, 110,150, 151, 165, 171, 191]  # <-- Replace with your actual IDs
    NOT_USED_IDS = [115]  # IDs to ignore (e.g., ground=0)

    # 64 TREES
    # 110 BUILDINGS
    # 115 ANIMATED OBJECTS (ignore these)
    # 150 STRUCTURES 1
    # 151 STRUCTURES 2
    # 161 STRUCTURES 3
    # 165 STRUCTURES 4
    # 171 STRUCTURES 5
    # 191 STRUCTURES 6
    # 205 OBJECTS (e.g., boxes, barrels)
    

    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, True)
    ])
    if responses and len(responses) > 0:
        img_bytes = responses[0].image_data_uint8
        if len(img_bytes) == 0:
            return None
    # Convert to grayscale mask and then to numpy array
    mask = Image.open(BytesIO(img_bytes)).convert("L").resize(IMG_SIZE, Image.NEAREST)
    mask_np = np.array(mask)
    # Create binary mask: keep only allowed IDs
    filtered_mask = np.isin(mask_np, ALLOWED_IDS).astype(np.uint8) * 255
    mask_img = Image.fromarray(filtered_mask, mode="L")
    return mask_img
    return None

def composite_obstacle_on_bg(anchor_img, mask, bg_img):
    # The mask is a grayscale image where obstacle pixels have high values (e.g., 255)
    # and background pixels have low values (e.g., 0)
    anchor_img = anchor_img.resize(IMG_SIZE)
    bg_img = bg_img.resize(IMG_SIZE)
    mask = mask.resize(IMG_SIZE)
    # Convert mask to binary alpha: obstacle=255 (opaque), background=0 (transparent)
    alpha = mask.point(lambda x: 255 if x > 128 else 0)
    # Convert images to RGBA for compositing
    anchor_rgba = anchor_img.convert("RGBA")
    bg_rgba = bg_img.convert("RGBA")
    # Apply alpha mask to anchor image (obstacles remain, background becomes transparent)
    anchor_rgba.putalpha(alpha)
    # Composite: overlay obstacles onto new background
    result = Image.alpha_composite(bg_rgba, anchor_rgba)
    # Return composited image as RGB
    return result.convert("RGB")

def save_triplet(anchor, positive, negative, idx):
    anchor.save(os.path.join(SAVE_DIR, "anchor", f"img_{idx:04d}.png"))
    positive.save(os.path.join(SAVE_DIR, "positive", f"img_{idx:04d}.png"))
    negative.save(os.path.join(SAVE_DIR, "negative", f"img_{idx:04d}.png"))

def save_anchor_and_positives(anchor, positives, idx):
    anchor_dir = os.path.join(SAVE_DIR, f"anchor_{idx:05d}")
    os.makedirs(anchor_dir, exist_ok=True)
    anchor.save(os.path.join(anchor_dir, "anchor.png"))
    for i, pos_img in enumerate(positives, start=1):
        pos_img.save(os.path.join(anchor_dir, f"positive_{i}.png"))

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

def replace_background(original_img, mask, new_bg):
    # Ensure all images are the same size
    original_img = original_img.resize(new_bg.size)
    mask = mask.resize(new_bg.size)
    # Convert mask to binary (obstacle=1, background=0)
    mask = mask.convert('L').point(lambda x: 255 if x > 128 else 0)
    # Composite: keep obstacle from original, background from new_bg
    result = Image.composite(original_img, new_bg, mask)
    return result

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
        mask = get_segmentation_mask(client)
        if img_anchor and mask:
            # Always use black.png and white.png, plus 3 random others
            bg_files = [f for f in os.listdir(BACKGROUND_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            fixed_bgs = ['black.png', 'white.png']
            # Exclude fixed backgrounds from random selection
            random_bgs = [f for f in bg_files if f not in fixed_bgs]
            chosen_random = random.sample(random_bgs, min(3, len(random_bgs)))
            chosen_bgs = fixed_bgs + chosen_random
            positives = []
            for bg_file in chosen_bgs:
                bg_img = Image.open(os.path.join(BACKGROUND_DIR, bg_file)).convert("RGB")
                pos_img = composite_obstacle_on_bg(img_anchor, mask, bg_img)
                positives.append(pos_img)
            save_anchor_and_positives(img_anchor, positives, i)

        print(f"[{i+1}/{N_SAMPLES+offset}] anchor e positivi salvati (altitudine {current_alt:.2f} m)")
        time.sleep(CAPTURE_INTERVAL)

    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("✅ Raccolta completata!")

if __name__ == "__main__":
    main()
