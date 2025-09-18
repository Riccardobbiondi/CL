import os

BG_DIR = os.path.dirname(__file__)

# Rename random backgrounds
for fname in os.listdir(BG_DIR):
    if fname.startswith('background_random_') and fname.endswith('.png'):
        num = fname.split('_')[-1].replace('.png', '')
        new_name = f'b_{num}.png'
        os.rename(os.path.join(BG_DIR, fname), os.path.join(BG_DIR, new_name))
    elif fname == 'background_white.png':
        os.rename(os.path.join(BG_DIR, fname), os.path.join(BG_DIR, 'white.png'))
    elif fname == 'background_black.png':
        os.rename(os.path.join(BG_DIR, fname), os.path.join(BG_DIR, 'black.png'))

print('Rinominati!')
