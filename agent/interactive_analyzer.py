import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import os

# Prova a importare tqdm per la barra di progresso.
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# --- Funzione di calcolo della similarità ---
def SA(row1, row2, Wp, Wv, Wpos, Wrot):
    """
    Calcola la "Similarità Attesa" (SA) tra due stati del drone.
    """
    if row1['env_name'] != row2['env_name']:
        return 0.0

    pos1 = row1[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    pos2 = row2[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    vel1_mag = np.linalg.norm(row1[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    vel2_mag = np.linalg.norm(row2[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    
    pos_distance = np.linalg.norm(pos1 - pos2)
    avg_velocity = (vel1_mag + vel2_mag) / 2.0
    dynamic_scale = Wp / (1 + avg_velocity * Wv)
    pos_similarity = np.exp(-dynamic_scale * pos_distance)

    q1 = row1[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    q2 = row2[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    
    norm_q1 = np.linalg.norm(q1)
    norm_q2 = np.linalg.norm(q2)
    if norm_q1 > 0: q1 /= norm_q1
    if norm_q2 > 0: q2 /= norm_q2
    
    dot_product = np.abs(np.dot(q1, q2))
    rot_s = (1-np.clip(dot_product, 0.0, 1) )* 10
    rot_similarity = 1-rot_s

    return (pos_similarity * Wpos) + (rot_similarity * Wrot)

# --- Caricamento e preparazione dei dati ---
RAW_DATA_FILE = 'data_collection/prova.csv'
ANCHOR_ID = 1

print(f"Loading raw data from: {RAW_DATA_FILE}...")
if not os.path.exists(RAW_DATA_FILE):
    print(f"Error: Raw data file not found at '{RAW_DATA_FILE}'")
    exit()

df = pd.read_csv(RAW_DATA_FILE)

if ANCHOR_ID not in df['anchor_id'].values:
    print(f"Error: Anchor ID {ANCHOR_ID} not found in the raw data file.")
    exit()

anchor_row = df[df['anchor_id'] == ANCHOR_ID].iloc[0]
other_rows = df[df['anchor_id'] != ANCHOR_ID]
print("Data loaded and prepared.")

# --- Creazione dell'interfaccia grafica ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35)

# Valori iniziali
initial_Wp = 0.25
initial_Wv = 0.75
initial_Wpos = 0.5
initial_Wrot = 0.5

# --- Assi per gli slider ---
ax_Wp = plt.axes([0.25, 0.20, 0.65, 0.03])
ax_Wv = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_Wpos = plt.axes([0.25, 0.10, 0.65, 0.03])
ax_Wrot_display = plt.axes([0.25, 0.05, 0.65, 0.03]) # Per mostrare Wrot

# --- Creazione degli slider ---
slider_Wp = Slider(ax=ax_Wp, label='Wp (Dist Sens)', valmin=0.01, valmax=1.0, valinit=initial_Wp)
slider_Wv = Slider(ax=ax_Wv, label='Wv (Vel Tol)', valmin=0.0, valmax=2.0, valinit=initial_Wv)
slider_Wpos = Slider(ax=ax_Wpos, label='Wpos (Pos Weight)', valmin=0.0, valmax=1.0, valinit=initial_Wpos)

# --- Funzione di aggiornamento ---
def update(val):
    Wp = slider_Wp.val
    Wv = slider_Wv.val
    Wpos = slider_Wpos.val
    Wrot = 1.0 - Wpos  # Wrot è derivato da Wpos

    # Aggiorna il display di Wrot
    ax_Wrot_display.clear()
    ax_Wrot_display.text(0.5, 0.5, f'Wrot (Rot Weight): {Wrot:.2f}', ha='center', va='center')
    ax_Wrot_display.set_xticks([])
    ax_Wrot_display.set_yticks([])

    # Calcola le similarità
    results = []
    iterator = tqdm(other_rows.iterrows(), total=len(other_rows), desc="Calculating similarities") if TQDM_AVAILABLE else other_rows.iterrows()
    
    for index, current_row in iterator:
        similarity = SA(anchor_row, current_row, Wp=Wp, Wv=Wv, Wpos=Wpos, Wrot=Wrot)
        results.append(similarity)
    
    # Aggiorna il plot
    ax.clear()
    ax.hist(results, bins=50, density=True, alpha=0.7, label='Similarity Distribution')
    
    mean_score = np.mean(results)
    ax.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
    
    ax.set_title(f'Interactive Similarity Distribution for Anchor ID: {ANCHOR_ID}', fontsize=16)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()
    ax.set_xlim(0, 1)
    fig.canvas.draw_idle()

# Connetti la funzione di aggiornamento agli slider
slider_Wp.on_changed(update)
slider_Wv.on_changed(update)
slider_Wpos.on_changed(update)

# Chiamata iniziale per disegnare il primo grafico
update(None)

plt.show()
