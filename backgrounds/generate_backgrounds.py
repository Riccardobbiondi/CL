import os
import random
from PIL import Image

# Paths
DATASET_V2 = os.path.join(os.path.dirname(__file__), '..', 'dataset_v2_preview')
BG_DIR = os.path.dirname(__file__)
IMG_SIZE = (224, 224)
N_RANDOM = 10  # Number of random backgrounds to generate

# 1. Create white and black backgrounds
white_bg = Image.new('RGB', IMG_SIZE, (255, 255, 255))
white_bg.save(os.path.join(BG_DIR, 'white.png'))

black_bg = Image.new('RGB', IMG_SIZE, (0, 0, 0))
black_bg.save(os.path.join(BG_DIR, 'black.png'))

# 2. Randomly select images from dataset_v2_preview
anchor_dirs = [d for d in os.listdir(DATASET_V2) if d.startswith('anchor_')]
all_imgs = []
for anchor_dir in anchor_dirs:
    dir_path = os.path.join(DATASET_V2, anchor_dir)
    imgs = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_imgs.extend(imgs)

random_imgs = random.sample(all_imgs, min(N_RANDOM, len(all_imgs)))
for i, img_path in enumerate(random_imgs, start=1):
    try:
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
        img.save(os.path.join(BG_DIR, f'b_{i}.png'))
    except Exception as e:
        print(f'Error processing {img_path}: {e}')

print('Backgrounds generated in', BG_DIR)
