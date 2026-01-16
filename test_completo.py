import torch
import time

print("\n" + "="*40)
print("     DIAGNOSTICA GPU RDNA 4     ")
print("="*40)

# 1. Verifica Software
print(f"PyTorch Version: {torch.__version__}")
try:
    print(f"ROCm Version:    {torch.version.hip}")
except:
    print("ROCm Version:    Non rilevato")

print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 2. Verifica Hardware
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n✅ GPU Rilevata: {gpu_name}")
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    print(f"💾 VRAM Totale: {vram_gb:.2f} GB")
    
    # 3. Stress Test Matematico
    print("\n[Avvio Stress Test...]")
    try:
        # Allocazione
        print(" -> Allocazione tensori (4096 x 4096)...", end=" ")
        x = torch.randn(4096, 4096, device=device)
        y = torch.randn(4096, 4096, device=device)
        print("OK.")
        
        # Calcolo
        print(" -> Moltiplicazione matrici...", end=" ")
        start = time.time()
        z = torch.mm(x, y) # Moltiplicazione pura
        torch.cuda.synchronize() # Aspetta che la GPU finisca davvero
        end = time.time()
        print("OK.")
        
        elapsed = end - start
        ops = (2 * 4096**3) / elapsed / 1e12 # TFLOPS approssimativi
        
        print(f"\n⏱️  Tempo calcolo: {elapsed:.4f} secondi")
        print(f"🚀 Performance stimate: {ops:.2f} TFLOPS")
        print("\n=== ESITO: SUCCESSO! LA TUA 9070 XT È PRONTA ===")
        
    except Exception as e:
        print(f"\n❌ ERRORE NEL CALCOLO: {e}")
else:
    print("\n❌ ERRORE: Nessuna GPU rilevata da PyTorch.")
