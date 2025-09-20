import pandas as pd
import numpy as np

# Prova a importare tqdm per la barra di progresso.
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Libreria 'tqdm' non trovata. Verrà mostrato un progresso testuale.")
    print("Per una barra di progresso visuale, installala con: pip install tqdm")


def SA(row1, row2, Wp, Wv, Wpos, Wrot):
    """
    Calcola la "Similarità Attesa" (SA) tra due stati del drone basandosi su dati privilegiati.
    """
    # 1. Controllo preliminare
    if row1['anchor_id'] == row2['anchor_id']:
        return 1.0
    if row1['env_name'] != row2['env_name']:
        return 0.0

    # --- 2. Similarità di Posizione ---
    pos1 = row1[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    pos2 = row2[['pos_x', 'pos_y', 'pos_z']].to_numpy(dtype=float)
    vel1_mag = np.linalg.norm(row1[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    vel2_mag = np.linalg.norm(row2[['vel_x', 'vel_y', 'vel_z']].to_numpy(dtype=float))
    
    pos_distance = np.linalg.norm(pos1 - pos2)
    avg_velocity = (vel1_mag + vel2_mag) / 2.0
    dynamic_scale = Wp / (1 + avg_velocity * Wv)
    pos_similarity = np.exp(-dynamic_scale * pos_distance)

    # --- 3. Similarità di Rotazione ---
    q1 = row1[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    q2 = row2[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy(dtype=float)
    
    # Normalizza i quaternioni per sicurezza
    norm_q1 = np.linalg.norm(q1)
    norm_q2 = np.linalg.norm(q2)
    if norm_q1 > 0: q1 /= norm_q1
    if norm_q2 > 0: q2 /= norm_q2
    
    dot_product = np.abs(np.dot(q1, q2))
    rot_similarity = np.clip(dot_product, 0.0, 1.0)

    # --- 4. Calcolo del punteggio finale (SA) ---
    expected_similarity = (pos_similarity * Wpos) + (rot_similarity * Wrot)
    
    return expected_similarity

if __name__ == '__main__':
    # --- 1. Impostazione degli iperparametri ---
    Wp = 0.25      # Sensibilità alla distanza
    Wv = 0.75      # Tolleranza alla velocità
    Wpos = 0.6    # Peso posizione
    Wrot = 0.4    # Peso rotazione

    print("--- Inizio Calcolo Matrice di Similarità ---")
    
    # --- 2. Caricamento dei dati ---
    try:
        df = pd.read_csv("../data_collection/prova.csv")
    except FileNotFoundError:
        print("Errore: File '../data_collection/prova.csv' non trovato.")
        exit()

    n = len(df)
    print(f"Trovate {n} righe nel file. Verrà creata una matrice {n}x{n}.")
    
    # --- 3. Inizializzazione della matrice ---
    similarity_matrix = np.zeros((n, n))

    # --- 4. Calcolo ottimizzato con barra di progresso ---
    print("Calcolo della matrice di similarità...")
    
    iterator = tqdm(range(n), desc="Calcolo Similarità") if TQDM_AVAILABLE else range(n)

    for i in iterator:
        # Calcola solo la matrice triangolare superiore
        for j in range(i, n):
            row1 = df.iloc[i]
            row2 = df.iloc[j]
            
            similarity = SA(row1, row2, Wp=Wp, Wv=Wv, Wpos=Wpos, Wrot=Wrot)
            
            # Sfrutta la simmetria della matrice
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
            
        if not TQDM_AVAILABLE and (i + 1) % 100 == 0:
             print(f"Processate {i + 1}/{n} righe...")


    print("Calcolo completato.")

    # --- 5. Salvataggio della matrice ---
    output_filename = 'similarity_matrix.csv'
    matrix_df = pd.DataFrame(similarity_matrix)
    # Aggiunto float_format per arrotondare a 4 cifre decimali nell'output
    matrix_df.to_csv(output_filename, index=False, header=False, float_format='%.4f')

    print(f"Matrice di similarità salvata con successo in '{output_filename}'.")

