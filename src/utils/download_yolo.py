from ultralytics import YOLO
import os

def download_yolo11x_seg():
    model_name = "yolo11x-seg.pt"
    
    print(f"--- AVVIO DOWNLOAD: {model_name} ---")
    print("Nota: Il file è grande (~100-200 MB), attendi...")

    try:
        # Inizializzando la classe YOLO con il nome del file, 
        # Ultralytics controlla se esiste. Se non esiste, lo scarica automaticamente.
        model = YOLO(model_name)
        
        # Facciamo una info() per confermare che sia caricato in memoria
        model.info()
        
        print(f"\n[SUCCESSO] '{model_name}' è stato scaricato correttamente!")
        print(f"Percorso file: {os.path.abspath(model_name)}")
        
    except Exception as e:
        print(f"\n[ERRORE] Impossibile scaricare il modello.")
        print(f"Dettaglio: {e}")
        print("Suggerimento: Prova a fare 'pip install -U ultralytics'")

if __name__ == "__main__":
    download_yolo11x_seg()