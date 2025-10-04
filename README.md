# CL
Project for thesis: Contrastive Learning for robust flight control in UAVs

## Struttura delle directory

Questo repository è organizzato nelle seguenti cartelle principali:

- **`CLEncoderRobotNav/`**: **Questa è la cartella principale per l'addestramento del modello.** Contiene il codice per il training vero e proprio del sistema di contrastive learning.
- **`agent/`**: Contiene lo script `ai_agent.py` per il movimento autonomo e vari script di utilità e analisi per la similarità dei dati.
- **`backgrounds/`**: Raccolta di immagini di sfondo utilizzate per generare dati di addestramento sintetici e variati. Contiene anche gli script per la generazione.
- **`data_collection/`**: Include script e utility per la raccolta di dati dal simulatore AirSim e la loro elaborazione.
- **`dataset_vX/`**: Contiene il dataset grezzo, suddiviso in campioni `anchor`, `positive` e `negative`, secondo l'approccio di contrastive learning.
- **`dataset_final/`**: Contiene una prima versione del dataset finale, ma senza privileged data.
- **`dataset_plus/`**: Contiene una versione migliorata del dataset finale, con dati privilegiati inclusi. USATO PER L'ADDESTRAMENTO FINALE.
- **`learning_v1/`**: Cartella contenente una prima prova di addestramento. **Da considerare solo come un test iniziale.**
- **`learning_v2/`**: Cartella contenente una seconda prova di addestramento. **Da considerare solo come un test.**

