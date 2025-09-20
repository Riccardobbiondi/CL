1.  **`anchor_id`**
    *   **Descrizione**: Identificativo numerico univoco per ogni campione "ancora". Permette di associare i dati di questa riga all'immagine corrispondente nella cartella `dataset_plus/anchor_XXXXXX/`.
    *   **Tipo**: Intero

2.  **`env_name`**
    *   **Descrizione**: Il nome del livello/mappa di AirSim in cui è stato raccolto il dato (es. "Blocks", "Neighborhood"). Viene estratto automaticamente dalle impostazioni del simulatore.
    *   **Tipo**: Stringa

3.  **`pos_x`, `pos_y`, `pos_z`**
    *   **Descrizione**: Le coordinate (X, Y, Z) del drone nel sistema di riferimento globale del simulatore.
    *   **Unità**: Metri
    *   **Tipo**: Float

4.  **`q_w`, `q_x`, `q_y`, `q_z`**
    *   **Descrizione**: L'orientamento del drone rappresentato come un quaternione. I quaternioni sono un sistema a 4 valori per descrivere una rotazione nello spazio 3D senza ambiguità.
    *   **Tipo**: Float

5.  **`vel_x`, `vel_y`, `vel_z`**
    *   **Descrizione**: La velocità lineare del drone lungo gli assi X, Y, e Z del sistema di riferimento globale.
    *   **Unità**: Metri al secondo
    *   **Tipo**: Float

6.  **`ang_vel_x`, `ang_vel_y`, `ang_vel_z`**
    *   **Descrizione**: La velocità angolare del drone, ovvero quanto velocemente sta ruotando attorno ai propri assi (roll, pitch, yaw).
    *   **Unità**: Radianti al secondo
    *   **Tipo**: Float

7.  **`has_collided`**
    *   **Descrizione**: Un indicatore booleano che diventa `True` se il drone ha registrato una collisione con un oggetto dell'ambiente.
    *   **Tipo**: Booleano (`True` o `False`)
