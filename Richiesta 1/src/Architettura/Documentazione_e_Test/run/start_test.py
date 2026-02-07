"""Esegue N simulazioni per T minuti ciascuna e genera documentazione."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from subprocess import Popen


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print("Valore non valido, uso default.")
        return default


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip().replace(",", ".")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Valore non valido, uso default.")
        return default


def _prompt_text(label: str, default: str = "") -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def _run_simulation(
    start_script: Path,
    goal: str,
    duration_sec: float,
    env: dict,
    cwd: Path,
    idx: int,
    total: int,
    log_interval: int,
) -> None:
    args = [sys.executable, start_script.as_posix()]
    if goal:
        args.append(goal)
    print(f"\n[TEST {idx}] Simulazione {idx}/{total} avviata (durata {duration_sec:.0f}s)...")
    proc = Popen(args, cwd=cwd.as_posix(), env=env)
    start_time = time.time()
    deadline = start_time + duration_sec
    last_log = 0.0
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            now = time.time()
            elapsed = int(now - start_time)
            remaining = max(0, int(deadline - now))
            if elapsed - last_log >= log_interval:
                em, es = divmod(elapsed, 60)
                rm, rs = divmod(remaining, 60)
                print(
                    f"[TEST {idx}] tempo trascorso {em:02d}:{es:02d} | "
                    f"rimasti {rm:02d}:{rs:02d}"
                )
                last_log = elapsed
            time.sleep(1)
        if proc.poll() is None:
            print("[TEST] Tempo scaduto: invio SIGINT (Ctrl+C).")
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except Exception:
                print("[TEST] Arresto forzato (SIGTERM).")
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:
                    print("[TEST] Kill processo.")
                    proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
        print(f"[TEST {idx}] Simulazione {idx}/{total} terminata.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Esegue test automatici con più simulazioni."
    )
    parser.add_argument("--count", type=int, default=0, help="Numero simulazioni")
    parser.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="Minuti per simulazione",
    )
    parser.add_argument(
        "--goal",
        default="",
        help="Testo obiettivo per tutte le simulazioni (es. 'cerco la mela')",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Secondi tra i log di progresso (default: 1)",
    )
    args = parser.parse_args()

    count = args.count or _prompt_int("Quante simulazioni vuoi fare?", 5)
    minutes = args.minutes or _prompt_float("Minuti per simulazione?", 3.0)
    goal = args.goal or _prompt_text("Testo obiettivo (opzionale)", "")
    log_interval = args.log_interval
    if log_interval <= 0:
        log_interval = 1

    if count <= 0:
        print("Numero simulazioni non valido.")
        return 1
    if minutes <= 0:
        print("Durata non valida.")
        return 1

    total_minutes = count * minutes
    print(f"\n[TEST] Totale simulazioni: {count}")
    print(f"[TEST] Durata per simulazione: {minutes} minuti")
    print(f"[TEST] Tempo totale stimato: {total_minutes:.1f} minuti\n")
    start_ok = input("Iniziare il test? [Y/n]: ").strip().lower()
    if start_ok and start_ok not in {"y", "yes"}:
        print("[TEST] Annullato dall'utente.")
        return 0

    src_root = Path(__file__).resolve().parents[3]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else str(src_root)
    )

    start_script = src_root / "Architettura" / "app" / "start.py"
    duration_sec = minutes * 60.0

    for idx in range(1, count + 1):
        env = env.copy()
        env["TEST_INDEX"] = str(idx)
        _run_simulation(
            start_script=start_script,
            goal=goal,
            duration_sec=duration_sec,
            env=env,
            cwd=src_root,
            idx=idx,
            total=count,
            log_interval=log_interval,
        )
        if idx < count:
            print(f"[NEXT]: passo a Test {idx + 1}")
            time.sleep(2)

    print("\n[TEST] Tutte le simulazioni completate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
