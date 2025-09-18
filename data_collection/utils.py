# data_collection/utils.py

from PIL import ImageEnhance, ImageOps
import os

def transform_image(img, type='positive'):
    if type == 'positive':
        img = img.rotate(5)  # rotazione leggera
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
    elif type == 'negative':
        img = img.rotate(30)
        img = ImageOps.autocontrast(img)
    return img

def save_triplet(anchor, positive, negative, save_dir, idx):
    anchor.save(os.path.join(save_dir, "anchor", f"img_{idx:04d}.png"))
    positive.save(os.path.join(save_dir, "positive", f"img_{idx:04d}.png"))
    negative.save(os.path.join(save_dir, "negative", f"img_{idx:04d}.png"))
