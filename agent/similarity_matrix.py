import pandas as pd
import numpy as np
import time

# Prova a importare SciPy per calcoli di distanza ottimizzati.
try:
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Libreria 'scipy' non trovata. L'ottimizzazione non può essere eseguita.")
    print("Per prestazioni massime, installala con: pip install scipy")
    exit()

def calculate_similarity_matrix(df, Wp, Wv, Wpos, Wrot):
    """
    Calcola la matrice di similarità per l'intero dataset usando un approccio vettorizzato e ottimizzato.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati privilegiati.
        Wp (float): Parametro di sensibilità alla posizione.
        Wv (float): Parametro di tolleranza alla velocità.
        Wpos (float): Peso della similarità di posizione.
        Wrot (float): Peso della similarità di rotazione.

    Returns:
        pd.DataFrame: Una matrice di similarità n x n come DataFrame pandas.
    """
    n = len(df)
    print(f"Inizio estrazione dati in blocco per {n} righe...")

    # Estrai tutti i dati necessari in matrici NumPy
    positions = df[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    velocities = df[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float)
    quaternions = df[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    env_names = df['env_name'].to_numpy()
    anchor_ids = df['anchor_id'].tolist()

    # --- Calcolo Vettorizzato ---
    
    # A) Similarità di Posizione
    print("Calcolo similarità di posizione...")
    pos_dist_matrix = squareform(pdist(positions, 'euclidean'))
    vel_magnitudes = np.linalg.norm(velocities, axis=1)
    avg_vel_matrix = np.add.outer(vel_magnitudes, vel_magnitudes) / 2.0
    dynamic_scale_matrix = Wp / (1 + avg_vel_matrix * Wv)
    pos_similarity_matrix = np.exp(-dynamic_scale_matrix * pos_dist_matrix)

    # B) Similarità di Rotazione
    print("Calcolo similarità di rotazione...")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions_normalized = quaternions / norms
    rot_similarity_matrix = np.abs(quaternions_normalized @ quaternions_normalized.T)

    # C) Combinazione e finalizzazione
    print("Combinazione dei risultati...")
    final_similarity_matrix = (pos_similarity_matrix * Wpos) + (rot_similarity_matrix * Wrot)
    
    # D) Applica regole di business (ambienti diversi e diagonale)
    env_mismatch_mask = np.not_equal.outer(env_names, env_names)
    final_similarity_matrix[env_mismatch_mask] = 0.0
    np.fill_diagonal(final_similarity_matrix, 1.0)

    # Crea il DataFrame finale usando gli anchor_id per indici e colonne
    matrix_df = pd.DataFrame(final_similarity_matrix, index=anchor_ids, columns=anchor_ids)
    
    return matrix_df


if __name__ == '__main__':
    # --- 1. Impostazione degli iperparametri ---
    Wp = 0.25      # Sensibilità alla distanza
    Wv = 0.75      # Tolleranza alla velocità
    Wpos = 0.6    # Peso posizione
    Wrot = 0.4    # Peso rotazione

    print("--- Inizio Calcolo Matrice di Similarità (Versione Ottimizzata) ---")
    start_time = time.time()
    
    # --- 2. Caricamento dei dati ---
    try:
        df = pd.read_csv("../data_collection/prova.csv")
    except FileNotFoundError:
        print("Errore: File '../data_collection/prova.csv' non trovato.")
        exit()

    # --- 3. Calcolo della matrice ---
    matrix_df = calculate_similarity_matrix(df, Wp, Wv, Wpos, Wrot)

    calculation_time = time.time() - start_time
    print(f"Calcolo completato in {calculation_time:.2f} secondi.")

    # --- 4. Salvataggio della matrice ---
    output_filename = 'prova_similarity_matrix.csv'
    matrix_df.to_csv(output_filename, index=True, header=True, float_format='%.4f')

    print(f"Matrice di similarità salvata con successo in '{output_filename}'.")

