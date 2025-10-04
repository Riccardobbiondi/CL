import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Prova a importare tqdm per la barra di progresso.
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# --- Funzione di calcolo della similarità (copiata da similarity_calculator.py) ---
def SA(row1, row2, Wp, Wv, Wpos, Wrot):
    """
    Calcola la "Similarità Attesa" (SA) tra due stati del drone basandosi su dati privilegiati.
    """
    # Se l'ID è lo stesso, la similarità è massima (ma lo escluderemo dal calcolo)
    if row1['anchor_id'] == row2['anchor_id']:
        return 1.0
    # Se l'ambiente è diverso, la similarità è nulla
    if row1['env_name'] != row2['env_name']:
        return 0.0

    # Similarità di Posizione
    pos1 = row1[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    pos2 = row2[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    vel1_mag = np.linalg.norm(row1[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    vel2_mag = np.linalg.norm(row2[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    
    pos_distance = np.linalg.norm(pos1 - pos2)
    avg_velocity = (vel1_mag + vel2_mag) / 2.0
    dynamic_scale = Wp / (1 + avg_velocity * Wv)
    pos_similarity = np.exp(-dynamic_scale * pos_distance)

    # Similarità di Rotazione
    q1 = row1[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    q2 = row2[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    
    norm_q1 = np.linalg.norm(q1)
    norm_q2 = np.linalg.norm(q2)
    if norm_q1 > 0: q1 /= norm_q1
    if norm_q2 > 0: q2 /= norm_q2
    
    dot_product = np.abs(np.dot(q1, q2))
    rot_s = (1-np.clip(dot_product, 0.0, 1) )* 10
    rot_similarity = 1-rot_s

    # Calcolo del punteggio finale
    expected_similarity = (pos_similarity * Wpos) + (rot_similarity * Wrot)
    
    return expected_similarity

def analyze_similarity_on_the_fly(raw_data_path: str, anchor_id: int, output_dir: str = 'agent'):
    """
    Calcola le similarità per un'ancora al volo, ne analizza la distribuzione e salva i risultati.

    Args:
        raw_data_path (str): Il percorso al file CSV con i dati grezzi (es. prova.csv).
        anchor_id (int): L'ID dell'immagine ancora da analizzare.
        output_dir (str): La directory dove salvare il grafico e il file di debug.
    """
    # --- 1. Impostazione degli iperparametri bilanciati (v2) ---
    Wp = 0.25      # Aumentato per maggiore sensibilità alla distanza
    Wv = 0.75      # Invariato
    Wpos = 0.4     # Aumentato per dare più peso alla posizione
    Wrot = 1 - Wpos     # Diminuito per bilanciare con la posizione

    print(f"Loading raw data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at '{raw_data_path}'")
        return

    try:
        # --- 2. Caricamento e preparazione dei dati ---
        df = pd.read_csv(raw_data_path)

        if anchor_id not in df['anchor_id'].values:
            print(f"Error: Anchor ID {anchor_id} not found in the raw data file.")
            print(f"Available IDs range from {df['anchor_id'].min()} to {df['anchor_id'].max()}.")
            return
            
        anchor_row = df[df['anchor_id'] == anchor_id].iloc[0]
        print(f"Anchor row for ID {anchor_id} selected.")

        # --- 3. Calcolo delle similarità al volo ---
        results = []
        num_rows = len(df)
        iterator = tqdm(df.iterrows(), total=num_rows, desc=f"Calculating similarities for Anchor {anchor_id}") if TQDM_AVAILABLE else df.iterrows()

        print("Calculating similarities on-the-fly...")
        for index, current_row in iterator:
            if current_row['anchor_id'] == anchor_id:
                continue
            
            similarity = SA(anchor_row, current_row, Wp=Wp, Wv=Wv, Wpos=Wpos, Wrot=Wrot)
            results.append((current_row['anchor_id'], similarity))

        similarity_scores = [res[1] for res in results]
        print(f"Calculation complete. Found {len(similarity_scores)} similarity scores.")

        # --- 4. Salvataggio su file di debug (in modalità append) ---
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        debug_filename = os.path.join(output_dir, 'debug.txt')
        
        print(f"Appending results to {debug_filename}...")
        with open(debug_filename, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n--- Analysis for Anchor ID: {anchor_id} @ {timestamp} ---\n")
            f.write("Compared_ID\tSimilarity\n")
            for compared_id, score in results:
                f.write(f"{compared_id}\t{score:.4f}\n")
        print("Debug file updated.")

        # --- 5. Plot della distribuzione ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 7))

        sns.histplot(similarity_scores, bins=50, kde=True, ax=ax)

        ax.set_title(f'On-the-fly Similarity Distribution for Anchor ID: {anchor_id}', fontsize=16)
        ax.set_xlabel('Similarity Score', fontsize=12)
        ax.set_ylabel('Frequency (Occurrence)', fontsize=12)
        
        mean_score = np.mean(similarity_scores)
        ax.axvline(mean_score, color='r', linestyle='--', label=f'Mean: {mean_score:.2f}')
        ax.legend()

        output_filename = os.path.join(output_dir, f'similarity_distribution_anchor_{anchor_id}.png')
        plt.savefig(output_filename)
        print(f"Plot saved successfully to: {output_filename}")
        plt.close(fig)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Percorso al file con i dati grezzi per il calcolo
    RAW_DATA_FILE = 'data_collection/prova.csv'

    try:
        anchor_id_input = input("Please enter the Anchor ID to analyze: ")
        anchor_id_to_check = int(anchor_id_input)

        analyze_similarity_on_the_fly(
            raw_data_path=RAW_DATA_FILE,
            anchor_id=anchor_id_to_check
        )

    except ValueError:
        print("Invalid input. Please enter a valid integer for the Anchor ID.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
