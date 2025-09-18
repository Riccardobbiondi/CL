# CL
Project for thesis: Contrastive Learning for robust flight control in UAVs

## Struttura delle directory

Questo repository è organizzato nelle seguenti cartelle principali:

- **`agent/`**: Contiene lo script `ai_agent.py`, che definisce l'agente AI per il movimento autonomo del drone nel simulatore.
- **`backgrounds/`**: Raccolta di immagini di sfondo utilizzate per generare dati di addestramento sintetici e variati. Contiene anche gli script per la generazione.
- **`data_collection/`**: Include script e utility per la raccolta di dati dal simulatore AirSim e la loro elaborazione.
- **`dataset_vX/`**: Contiene il dataset grezzo, suddiviso in campioni `anchor`, `positive` e `negative`, secondo l'approccio di contrastive learning.
- **`dataset_final/`**: Il dataset finale, elaborato e strutturato, pronto per essere utilizzato durante la fase di addestramento del modello.
- **`learning_v1/`**: Contiene gli script per l'addestramento del modello di contrastive learning, incluso il `contrastive_trainer.py`.
