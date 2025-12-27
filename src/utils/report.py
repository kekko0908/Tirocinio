import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURAZIONE ---
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "Yolo_vs_VLM"
JSON_PATH = DATA_DIR / "FINAL.json"
OUTPUT_IMG = DATA_DIR / "THESIS_PROFESSIONAL_REPORT.png"

# Colori Tesi (Professionali)
COLOR_YOLO = '#1f77b4'  # Blu Accademico
COLOR_VLM = '#d62728'   # Rosso Accademico
COLOR_NONE = '#7f7f7f'  # Grigio Neutro

def autolabel(rects, ax):
    """Aggiunge etichette numeriche sopra le barre"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

def generate_graphs():
    if not JSON_PATH.exists():
        print(f"[ERR] File non trovato: {JSON_PATH}")
        print("Esegui prima 'src/18_benchmark_thesis.py' per generare i dati.")
        return

    print(f"[REPORT] Leggo dati da {JSON_PATH}...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Estrazione Dati Sicura
    details = data.get("details", [])
    if not details:
        print("[ERR] Il JSON è vuoto o non ha dettagli.")
        return

    # Calcoliamo il totale dinamicamente contando gli elementi nella lista dettagli
    total_simulations = len(details)
    yolo_wins = data.get("yolo_wins", 0)
    vlm_wins = data.get("vlm_wins", 0)
    
    # Recupera i dati per le barre
    objects = [d.get("obj", "Unknown") for d in details]
    
    # Gestione sicura nel caso manchino chiavi annidate
    iou_yolo = [d.get("yolo", {}).get("iou", 0.0) for d in details]
    iou_vlm = [d.get("vlm", {}).get("iou", 0.0) for d in details]
    time_yolo = [d.get("yolo", {}).get("time", 0.0) for d in details]
    time_vlm = [d.get("vlm", {}).get("time", 0.0) for d in details]

    # --- SETUP DASHBOARD (2x2 Grid) ---
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Benchmark Navigazione Semantica: YOLOv8-Seg vs VLM (Gemma-3-4b)', fontsize=16, fontweight='bold')
    
    # 1. GRAFICO IoU PER OGGETTO (Bar Chart)
    ax1 = axs[0, 0]
    x = np.arange(len(objects))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, iou_vlm, width, label='VLM (Gemma)', color=COLOR_VLM, alpha=0.8)
    rects2 = ax1.bar(x + width/2, iou_yolo, width, label='YOLOv8', color=COLOR_YOLO, alpha=0.8)
    
    ax1.set_ylabel('IoU Score (Precisione)')
    ax1.set_title('Precisione Bounding Box per Oggetto')
    ax1.set_xticks(x)
    ax1.set_xticklabels(objects, rotation=15)
    ax1.set_ylim(0, 1.15) # Spazio per le label
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # 2. GRAFICO TEMPI DI INFERENZA (Bar Chart)
    ax2 = axs[0, 1]
    rects3 = ax2.bar(x - width/2, time_vlm, width, label='VLM', color=COLOR_VLM, alpha=0.6)
    rects4 = ax2.bar(x + width/2, time_yolo, width, label='YOLO', color=COLOR_YOLO, alpha=0.6)
    
    ax2.set_ylabel('Tempo (secondi)')
    ax2.set_title('Latenza di Inferenza (Basso è meglio)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(objects, rotation=15)
    ax2.set_yscale('log') # Scala logaritmica
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    for i, v in enumerate(time_vlm):
        ax2.text(i - width/2, v * 1.1, f"{v:.1f}s", ha='center', fontsize=8)
    for i, v in enumerate(time_yolo):
        ax2.text(i + width/2, v * 1.1, f"{v:.2f}s", ha='center', fontsize=8)

    # 3. GRAFICO VITTORIE (Pie Chart)
    ax3 = axs[1, 0]
    
    # Calcolo Pareggi/Nulla
    draws = total_simulations - (yolo_wins + vlm_wins)
    wins = [yolo_wins, vlm_wins, draws]
    
    labels = ['YOLO Vince', 'VLM Vince', 'Pareggio/Nulla']
    colors = [COLOR_YOLO, COLOR_VLM, COLOR_NONE]
    explode = (0.05, 0, 0)
    
    # Filtra slice vuote
    final_wins = []
    final_labels = []
    final_colors = []
    final_explode = []
    
    for w, l, c, e in zip(wins, labels, colors, explode):
        if w > 0:
            final_wins.append(w)
            final_labels.append(l)
            final_colors.append(c)
            final_explode.append(e)

    if sum(final_wins) > 0:
        ax3.pie(final_wins, explode=final_explode, labels=final_labels, colors=final_colors,
                autopct='%1.1f%%', shadow=True, startangle=140, textprops={'fontsize': 12, 'fontweight': 'bold'})
    else:
        ax3.text(0.5, 0.5, "Nessun dato valido", ha='center')
        
    ax3.set_title('Distribuzione Vittorie (IoU Dominante)')

    # 4. GRAFICO MEDIE GENERALI (Horizontal Bar)
    ax4 = axs[1, 1]
    
    categories = ['Media IoU']
    y_pos = np.arange(len(categories))
    
    # Gestione chiavi diverse (avg_vlm o average_iou_vlm)
    avg_iou_vlm = data.get("avg_vlm", data.get("average_iou_vlm", 0))
    avg_iou_yolo = data.get("avg_yolo", data.get("average_iou_yolo", 0))
    
    bar_width = 0.3
    ax4.barh(y_pos - bar_width/2, avg_iou_vlm, bar_width, color=COLOR_VLM, label='VLM')
    ax4.barh(y_pos + bar_width/2, avg_iou_yolo, bar_width, color=COLOR_YOLO, label='YOLO')
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(categories)
    ax4.set_xlim(0, 1.0)
    ax4.set_title('Performance Media Complessiva')
    ax4.legend(loc='lower right')
    
    ratio = avg_iou_yolo/(avg_iou_vlm+1e-6)
    
    stats_text = (
        f"STATISTICHE TOTALI:\n\n"
        f"Simulazioni: {total_simulations}\n"
        f"Media IoU YOLO: {avg_iou_yolo:.3f}\n"
        f"Media IoU VLM:  {avg_iou_vlm:.3f}\n\n"
        f"YOLO è mediamente {ratio:.1f}x più preciso\n"
    )
    ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # --- SALVATAGGIO ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"[SUCCESS] Report grafico salvato in: {OUTPUT_IMG}")

if __name__ == "__main__":
    generate_graphs()