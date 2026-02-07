"""Benchmark multi-modello per start.py con report CSV e grafico."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.paths import ARTIFACTS_ROOT
from Architettura.vlm.client import resolve_model_id

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except Exception:
    plt = None
    Patch = None

ALLOWED_MODELS = ("qwen-vl2.5", "gemma-3")
ALLOWED_MODEL_SET = set(ALLOWED_MODELS)
MM_PROJ_BY_MODEL = {
    "qwen-vl2.5": "qwen-vl-mmproj-2.5",
}
MODEL_COLORS = {
    "qwen-vl2.5": "#1f77b4",
    "gemma-3": "#ff7f0e",
}
TARGET_NEAR_THRESHOLD_M = 0.8
BENCHMARK_FOLDER_NAME = "benchmark_data_X"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sanitize_name(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return clean.strip("_") or "model"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_models(text: str) -> list[str]:
    raw = [item.strip() for item in str(text).split(",")]
    models = [item for item in raw if item]
    dedup: list[str] = []
    seen: set[str] = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        dedup.append(model)
    return dedup


def _build_base_env(src_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pp}" if existing_pp else str(src_root)
    )
    return env


def _format_clock(seconds: float, *, always_hours: bool = False) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if always_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _is_missing_local_model(model_alias: str) -> bool:
    resolved = resolve_model_id(model_alias)
    if str(resolved).lower().endswith(".gguf"):
        return not Path(resolved).exists()
    return False


def _run_start_process(
    *,
    start_script: Path,
    goal: str,
    timeout_sec: float,
    startup_delay_sec: float,
    phase_path: Path,
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    run_index: int,
    total_runs: int,
    bench_started_ts: float,
    bench_total_budget_sec: float,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, start_script.as_posix()]
    if goal:
        args.append(goal)
    started_ts = time.time()
    timed_out = False
    sent_sigint = False
    returncode = None

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            args,
            cwd=cwd.as_posix(),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        expected_active_start_ts = started_ts + max(0.0, startup_delay_sec)
        active_phase_start_ts: float | None = None
        deadline = expected_active_start_ts + timeout_sec
        try:
            while True:
                now = time.time()
                returncode = proc.poll()
                if returncode is not None:
                    break
                phase_data = _read_json(phase_path)
                if active_phase_start_ts is None and isinstance(phase_data, dict):
                    active_raw = _safe_float(phase_data.get("active_run_started_ts"))
                    if active_raw is not None and active_raw > 0:
                        active_phase_start_ts = active_raw
                        deadline = active_phase_start_ts + timeout_sec
                phase_start_for_display = (
                    active_phase_start_ts if active_phase_start_ts is not None else expected_active_start_ts
                )
                in_warmup = now < phase_start_for_display
                run_elapsed = max(0.0, now - phase_start_for_display)
                run_remaining = max(0.0, timeout_sec - run_elapsed)
                warmup_elapsed = max(0.0, now - started_ts)
                warmup_remaining = max(0.0, phase_start_for_display - now)
                bench_elapsed = max(0.0, now - bench_started_ts)
                bench_remaining = max(0.0, bench_total_budget_sec - bench_elapsed)
                if in_warmup:
                    print(
                        (
                            f"[TEST {run_index}/{total_runs}] "
                            f"warmup {_format_clock(warmup_elapsed)} / {_format_clock(startup_delay_sec)} "
                            f"(restanti {_format_clock(warmup_remaining)}) | "
                            f"totale {_format_clock(bench_elapsed, always_hours=True)} / "
                            f"{_format_clock(bench_total_budget_sec, always_hours=True)} "
                            f"(restanti {_format_clock(bench_remaining, always_hours=True)})"
                        ),
                        flush=True,
                    )
                else:
                    print(
                        (
                            f"[TEST {run_index}/{total_runs}] "
                            f"run {_format_clock(run_elapsed)} / {_format_clock(timeout_sec)} "
                            f"(restanti {_format_clock(run_remaining)}) | "
                            f"totale {_format_clock(bench_elapsed, always_hours=True)} / "
                            f"{_format_clock(bench_total_budget_sec, always_hours=True)} "
                            f"(restanti {_format_clock(bench_remaining, always_hours=True)})"
                        ),
                        flush=True,
                    )
                if now >= deadline:
                    break
                time.sleep(1)
            if proc.poll() is None and time.time() >= deadline:
                timed_out = True
                sent_sigint = True
                logf.write("[BENCH] Timeout raggiunto: invio SIGINT.\n")
                logf.flush()
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except Exception:
                    logf.write("[BENCH] Arresto forzato: SIGTERM.\n")
                    logf.flush()
                    proc.terminate()
                    try:
                        proc.wait(timeout=15)
                    except Exception:
                        logf.write("[BENCH] Arresto forzato: SIGKILL.\n")
                        logf.flush()
                        proc.kill()
            returncode = proc.poll()
        finally:
            if proc.poll() is None:
                proc.kill()
            ended_ts = time.time()

    return {
        "started_ts": started_ts,
        "ended_ts": ended_ts,
        "runtime_sec": max(0.0, ended_ts - started_ts),
        "returncode": returncode,
        "timed_out": timed_out,
        "sent_sigint": sent_sigint,
        "log_path": log_path.as_posix(),
    }


def _run_documentation_process(
    *,
    documentation_script: Path,
    sim_name: str,
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
) -> dict[str, Any]:
    started_ts = time.time()
    args = [
        sys.executable,
        documentation_script.as_posix(),
        "--sim-name",
        sim_name,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[BENCH] Avvio documentazione per {sim_name}\n")
        logf.flush()
        returncode = subprocess.run(
            args,
            cwd=cwd.as_posix(),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).returncode
    ended_ts = time.time()
    status = "ok" if returncode == 0 else "failed"
    return {
        "status": status,
        "returncode": returncode,
        "runtime_sec": max(0.0, ended_ts - started_ts),
    }


def _compute_distance_gain(start_distance: Any, end_distance: Any) -> float | None:
    start = _safe_float(start_distance)
    end = _safe_float(end_distance)
    if start is None or end is None:
        return None
    return max(0.0, start - end)


def _enrich_run_metrics(run_data: dict[str, Any]) -> None:
    initial = _safe_float(run_data.get("initial_target_distance_m"))
    minimum = _safe_float(run_data.get("min_target_distance_m"))
    final = _safe_float(run_data.get("final_target_distance_m"))

    run_data["success_rate_m"] = _compute_distance_gain(initial, minimum)
    run_data["final_approach_m"] = _compute_distance_gain(initial, final)
    run_data["target_within_0_8m"] = bool(
        minimum is not None and float(minimum) <= TARGET_NEAR_THRESHOLD_M
    )


def _fallback_summary(
    *,
    model_alias: str,
    goal: str,
    started_ts: float,
    ended_ts: float,
    exit_reason: str,
    returncode: int | None,
    documentation_sim_name: str,
) -> dict[str, Any]:
    return {
        "model_alias": model_alias,
        "resolved_model_id": resolve_model_id(model_alias),
        "goal": goal,
        "run_started_at": datetime.fromtimestamp(started_ts, timezone.utc).isoformat(),
        "run_ended_at": datetime.fromtimestamp(ended_ts, timezone.utc).isoformat(),
        "runtime_sec": max(0.0, ended_ts - started_ts),
        "exit_reason": exit_reason,
        "steps_total": 0,
        "initial_target_distance_m": None,
        "min_target_distance_m": None,
        "final_target_distance_m": None,
        "target_reached": False,
        "success": False,
        "action_failures_count": 0,
        "json_parse_failures_count": 0,
        "semantic_validation_failures_count": 0,
        "retry_attempts_total": 0,
        "vlm_inference_mean_sec": None,
        "vlm_inference_p95_sec": None,
        "model_load_sec": None,
        "documentation_sim_name": documentation_sim_name,
        "documentation_status": None,
        "documentation_returncode": None,
        "documentation_runtime_sec": None,
        "process_returncode": returncode,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if field_order is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    else:
        extra = set()
        for row in rows:
            extra.update(row.keys())
        extras_sorted = sorted(k for k in extra if k not in field_order)
        fieldnames = list(field_order) + extras_sorted
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_by_model(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        model = str(run.get("model_alias", "")).strip() or "unknown"
        grouped.setdefault(model, []).append(run)

    summary_rows: list[dict[str, Any]] = []
    for model, rows in grouped.items():
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                _safe_int(item.get("run_index")),
                _safe_int(item.get("model_run_index")),
            ),
        )
        evaluated = [row for row in rows_sorted if row.get("status") != "skipped"]
        evaluated_count = len(evaluated)

        success_rate_m_vals = [
            v
            for row in evaluated
            if (v := _safe_float(row.get("success_rate_m"))) is not None
        ]
        final_approach_vals = [
            v
            for row in evaluated
            if (v := _safe_float(row.get("final_approach_m"))) is not None
        ]
        runtimes = [
            v for row in evaluated if (v := _safe_float(row.get("runtime_sec"))) is not None
        ]
        within_count = sum(1 for row in evaluated if bool(row.get("target_within_0_8m")))
        legacy_success_count = sum(1 for row in evaluated if bool(row.get("success")))

        summary_rows.append(
            {
                "model_alias": model,
                "runs_total": len(rows_sorted),
                "runs_evaluated": evaluated_count,
                "runs_skipped": len(rows_sorted) - evaluated_count,
                "success_rate_m_mean": _mean(success_rate_m_vals),
                "success_rate_m_median": _median(success_rate_m_vals),
                "final_approach_m_mean": _mean(final_approach_vals),
                "target_within_0_8m_pct": (
                    within_count * 100.0 / evaluated_count if evaluated_count > 0 else 0.0
                ),
                "legacy_success_rate_pct": (
                    legacy_success_count * 100.0 / evaluated_count
                    if evaluated_count > 0
                    else 0.0
                ),
                "median_runtime_sec": _median(runtimes),
            }
        )
    order = {model: idx for idx, model in enumerate(ALLOWED_MODELS)}
    return sorted(summary_rows, key=lambda row: order.get(str(row.get("model_alias", "")), 999))


def _plot_overview(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib non disponibile: salto grafico benchmark.")
        return
    if not summary_rows:
        return

    models = [str(row.get("model_alias", "")) for row in summary_rows]
    raw_values = [_safe_float(row.get("success_rate_m_mean")) for row in summary_rows]
    values = [0.0 if v is None else v for v in raw_values]
    colors = [MODEL_COLORS.get(model, "#666666") for model in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, values, color=colors)
    ax.set_title("Benchmark score medio (success_rate_m)")
    ax.set_ylabel("Metri avvicinamento medi (m)")
    ax.set_xlabel("Modello")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for idx, bar in enumerate(bars):
        text = "n/a" if raw_values[idx] is None else f"{raw_values[idx]:.2f} m"
        y_pos = bar.get_height() + 0.02
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if Patch is not None:
        handles = [
            Patch(facecolor=MODEL_COLORS.get(model, "#666666"), label=model)
            for model in models
        ]
        ax.legend(handles=handles, title="Legenda modelli")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def _plot_score_per_run(run_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    evaluated = [row for row in run_rows if row.get("status") != "skipped"]
    if not evaluated:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for model in ALLOWED_MODELS:
        model_rows = [row for row in evaluated if str(row.get("model_alias")) == model]
        model_rows = sorted(model_rows, key=lambda row: _safe_int(row.get("model_run_index")))
        xs: list[int] = []
        ys: list[float] = []
        for row in model_rows:
            score = _safe_float(row.get("success_rate_m"))
            if score is None:
                continue
            xs.append(_safe_int(row.get("model_run_index")))
            ys.append(score)
        if not xs:
            continue
        plotted = True
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            color=MODEL_COLORS.get(model, "#666666"),
            label=model,
        )
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Score per run")
    ax.set_xlabel("Run index per modello")
    ax.set_ylabel("success_rate_m (m)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(title="Modello")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def _plot_score_distribution(run_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    labels: list[str] = []
    data: list[list[float]] = []
    for model in ALLOWED_MODELS:
        vals = [
            _safe_float(row.get("success_rate_m"))
            for row in run_rows
            if row.get("status") != "skipped" and str(row.get("model_alias")) == model
        ]
        clean_vals = [v for v in vals if v is not None]
        if not clean_vals:
            continue
        labels.append(model)
        data.append(clean_vals)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot(data, labels=labels, patch_artist=True)
    for idx, patch in enumerate(box["boxes"]):
        model = labels[idx]
        patch.set_facecolor(MODEL_COLORS.get(model, "#666666"))
        patch.set_alpha(0.6)
    ax.set_title("Distribuzione score per modello")
    ax.set_ylabel("success_rate_m (m)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def _plot_distance_scatter(run_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    max_axis = 0.0
    for model in ALLOWED_MODELS:
        model_rows = [
            row
            for row in run_rows
            if row.get("status") != "skipped" and str(row.get("model_alias")) == model
        ]
        init_min_x: list[float] = []
        init_min_y: list[float] = []
        init_final_x: list[float] = []
        init_final_y: list[float] = []
        for row in model_rows:
            initial = _safe_float(row.get("initial_target_distance_m"))
            min_dist = _safe_float(row.get("min_target_distance_m"))
            final_dist = _safe_float(row.get("final_target_distance_m"))
            if initial is None:
                continue
            if min_dist is not None:
                init_min_x.append(initial)
                init_min_y.append(min_dist)
                max_axis = max(max_axis, initial, min_dist)
            if final_dist is not None:
                init_final_x.append(initial)
                init_final_y.append(final_dist)
                max_axis = max(max_axis, initial, final_dist)
        if init_min_x:
            plotted = True
            ax.scatter(
                init_min_x,
                init_min_y,
                color=MODEL_COLORS.get(model, "#666666"),
                marker="o",
                alpha=0.8,
                label=f"{model} - min",
            )
        if init_final_x:
            plotted = True
            ax.scatter(
                init_final_x,
                init_final_y,
                color=MODEL_COLORS.get(model, "#666666"),
                marker="x",
                alpha=0.9,
                label=f"{model} - final",
            )
    if not plotted:
        plt.close(fig)
        return
    if max_axis > 0:
        ax.plot([0, max_axis], [0, max_axis], linestyle="--", color="#555555", linewidth=1)
    ax.set_title("Scatter distanze (iniziale vs minima/finale)")
    ax.set_xlabel("Distanza iniziale (m)")
    ax.set_ylabel("Distanza osservata (m)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def _plot_threshold_rate(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None or not summary_rows:
        return
    models = [str(row.get("model_alias", "")) for row in summary_rows]
    values = [
        0.0 if _safe_float(row.get("target_within_0_8m_pct")) is None
        else float(row.get("target_within_0_8m_pct"))
        for row in summary_rows
    ]
    colors = [MODEL_COLORS.get(model, "#666666") for model in models]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, values, color=colors)
    ax.set_title("Run entro soglia 0.8m")
    ax.set_ylabel("Percentuale run (%)")
    ax.set_xlabel("Modello")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-modello VLM con report CSV e grafico."
    )
    parser.add_argument("--count", type=int, default=1, help="Run per modello.")
    parser.add_argument("--minutes", type=float, default=3.0, help="Minuti per run.")
    parser.add_argument(
        "--startup-delay-sec",
        type=float,
        default=10.0,
        help="Secondi di warmup iniziale prima di avviare il cronometro run.",
    )
    parser.add_argument("--goal", default="", help="Goal unico da passare a start.py.")
    parser.add_argument(
        "--models",
        default="qwen-vl2.5,gemma-3",
        help="Lista alias modelli separati da virgola.",
    )
    parser.add_argument("--preset", default="accurate", help="Valore VLM_PRESET.")
    parser.add_argument("--max-steps", type=int, default=170, help="Valore VLM_MAX_STEPS.")
    parser.add_argument(
        "--output-root",
        default=(ARTIFACTS_ROOT / "Documentazione" / BENCHMARK_FOLDER_NAME).as_posix(),
        help="Cartella padre output benchmark (modelli + dati).",
    )
    parser.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        default=True,
        help="Continua benchmark anche se una run fallisce.",
    )
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Interrompe benchmark al primo errore.",
    )
    parser.add_argument(
        "--skip-missing-models",
        dest="skip_missing_models",
        action="store_true",
        default=True,
        help="Marca skipped i modelli locali mancanti invece di fallire.",
    )
    parser.add_argument(
        "--no-skip-missing-models",
        dest="skip_missing_models",
        action="store_false",
        help="Non saltare modelli mancanti.",
    )
    args = parser.parse_args()

    if args.count <= 0:
        print("Count non valido.")
        return 1
    if args.minutes <= 0:
        print("Minutes non valido.")
        return 1
    if args.startup_delay_sec < 0:
        print("Startup delay non valido.")
        return 1
    if args.max_steps <= 0:
        print("Max steps non valido.")
        return 1

    models = _parse_models(args.models)
    if not models:
        print("Nessun modello valido in --models.")
        return 1
    invalid_models = [model for model in models if model not in ALLOWED_MODEL_SET]
    if invalid_models:
        print(
            "Modelli non supportati: "
            + ", ".join(invalid_models)
            + f". Modelli consentiti: {', '.join(ALLOWED_MODELS)}."
        )
        return 1

    timeout_sec = float(args.minutes) * 60.0
    startup_delay_sec = float(args.startup_delay_sec)
    total_runs_planned = len(models) * int(args.count)
    bench_started_ts = time.time()
    bench_total_budget_sec = (timeout_sec + startup_delay_sec) * float(total_runs_planned)
    bench_name = f"BENCH_{_timestamp_tag()}"
    output_root = Path(args.output_root).expanduser().resolve()
    bench_data_root = output_root / "dati" / bench_name
    output_root.mkdir(parents=True, exist_ok=True)
    bench_data_root.mkdir(parents=True, exist_ok=True)

    start_script = SRC_ROOT / "Architettura" / "app" / "start.py"
    documentation_script = (
        SRC_ROOT / "Architettura" / "Documentazione_e_Test" / "run" / "start_documentation.py"
    )
    base_env = _build_base_env(SRC_ROOT)
    vlm_prompt_raw = os.environ.get("VLMPROMPT", "").strip()
    vlm_prompt_selected = vlm_prompt_raw if vlm_prompt_raw else "default"

    config_payload = {
        "benchmark_name": bench_name,
        "created_at": _utc_now_iso(),
        "count_per_model": args.count,
        "minutes_per_run": args.minutes,
        "startup_delay_sec": startup_delay_sec,
        "goal": args.goal,
        "models": models,
        "preset": args.preset,
        "max_steps": args.max_steps,
        "continue_on_error": args.continue_on_error,
        "skip_missing_models": args.skip_missing_models,
        "start_script": start_script.as_posix(),
        "documentation_script": documentation_script.as_posix(),
        "allowed_models": list(ALLOWED_MODELS),
        "vlm_prompt": vlm_prompt_selected,
        "output_root": output_root.as_posix(),
        "data_dir": bench_data_root.as_posix(),
        "model_dirs": {
            model: (output_root / model / bench_name).as_posix()
            for model in models
        },
    }
    _write_json(bench_data_root / "config.json", config_payload)
    input_payload = {
        "benchmark_name": bench_name,
        "created_at": _utc_now_iso(),
        "goal": args.goal,
        "vlm_prompt": vlm_prompt_selected,
        "count": args.count,
        "minutes": args.minutes,
        "startup_delay_sec": startup_delay_sec,
        "models": models,
        "preset": args.preset,
        "max_steps": args.max_steps,
        "continue_on_error": args.continue_on_error,
        "skip_missing_models": args.skip_missing_models,
        "total_runs_planned": total_runs_planned,
        "total_time_budget_sec": bench_total_budget_sec,
        "total_time_budget_hms": _format_clock(bench_total_budget_sec, always_hours=True),
        "output_root": output_root.as_posix(),
        "data_dir": bench_data_root.as_posix(),
        "model_dirs": {
            model: (output_root / model / bench_name).as_posix()
            for model in models
        },
    }
    _write_json(bench_data_root / "input_params.json", input_payload)

    run_rows: list[dict[str, Any]] = []
    global_idx = 0
    for model_alias in models:
        model_slug = _sanitize_name(model_alias)
        model_bench_dir = output_root / model_alias / bench_name
        model_log_dir = model_bench_dir / "logs"
        model_summary_dir = model_bench_dir / "summaries"
        model_bench_dir.mkdir(parents=True, exist_ok=True)
        model_log_dir.mkdir(parents=True, exist_ok=True)
        model_summary_dir.mkdir(parents=True, exist_ok=True)

        model_missing = _is_missing_local_model(model_alias)
        for model_run_idx in range(1, args.count + 1):
            global_idx += 1
            doc_sim_name = f"{bench_name}_{model_slug}_run_{model_run_idx:03d}"
            summary_path = model_summary_dir / f"run_{model_run_idx:03d}.json"
            phase_path = model_summary_dir / f"run_{model_run_idx:03d}_phase.json"
            log_path = model_log_dir / f"run_{model_run_idx:03d}.log"
            if phase_path.exists():
                try:
                    phase_path.unlink()
                except Exception:
                    pass

            print(
                f"[BENCH] Run {global_idx}: model={model_alias} "
                f"({model_run_idx}/{args.count})",
                flush=True,
            )

            if model_missing and args.skip_missing_models:
                now = time.time()
                skipped = _fallback_summary(
                    model_alias=model_alias,
                    goal=args.goal,
                    started_ts=now,
                    ended_ts=now,
                    exit_reason="missing_model",
                    returncode=None,
                    documentation_sim_name=doc_sim_name,
                )
                skipped["status"] = "skipped"
                skipped["run_index"] = global_idx
                skipped["model_run_index"] = model_run_idx
                skipped["log_path"] = log_path.as_posix()
                skipped["summary_path"] = summary_path.as_posix()
                skipped["phase_path"] = phase_path.as_posix()
                _enrich_run_metrics(skipped)
                _write_json(summary_path, skipped)
                run_rows.append(skipped)
                print(f"[BENCH] Skipped: modello locale mancante ({model_alias})", flush=True)
                continue

            run_env = base_env.copy()
            run_env["VLM_MODEL"] = model_alias
            run_env["VLM_PRESET"] = str(args.preset)
            run_env["VLM_MAX_STEPS"] = str(args.max_steps)
            run_env["BENCHMARK_SUMMARY_PATH"] = summary_path.as_posix()
            run_env["BENCHMARK_PHASE_PATH"] = phase_path.as_posix()
            run_env["DOC_SIM_NAME"] = doc_sim_name
            run_env["BENCHMARK_SKIP_AUTO_DOCUMENTATION"] = "1"
            run_env["VLM_BENCH_START_DELAY_SEC"] = str(startup_delay_sec)
            mmproj_alias = MM_PROJ_BY_MODEL.get(model_alias)
            if mmproj_alias:
                run_env["VLM_MMPROJ"] = mmproj_alias
            else:
                run_env.pop("VLM_MMPROJ", None)

            proc_info = _run_start_process(
                start_script=start_script,
                goal=args.goal,
                timeout_sec=timeout_sec,
                startup_delay_sec=startup_delay_sec,
                phase_path=phase_path,
                env=run_env,
                cwd=SRC_ROOT,
                log_path=log_path,
                run_index=global_idx,
                total_runs=total_runs_planned,
                bench_started_ts=bench_started_ts,
                bench_total_budget_sec=bench_total_budget_sec,
            )

            summary_data = _read_json(summary_path)
            if summary_data is None:
                fallback_reason = "timeout" if proc_info["timed_out"] else "exception"
                summary_data = _fallback_summary(
                    model_alias=model_alias,
                    goal=args.goal,
                    started_ts=proc_info["started_ts"],
                    ended_ts=proc_info["ended_ts"],
                    exit_reason=fallback_reason,
                    returncode=proc_info["returncode"],
                    documentation_sim_name=doc_sim_name,
                )

            if proc_info["timed_out"]:
                summary_data["exit_reason"] = "timeout"
                summary_data["success"] = False

            returncode = proc_info.get("returncode")
            if returncode not in (0, None) and summary_data.get("exit_reason") not in {
                "timeout",
                "exception",
            }:
                summary_data["exit_reason"] = "exception"
                summary_data["success"] = False

            summary_data["status"] = "completed" if bool(summary_data.get("success")) else "failed"
            if summary_data.get("exit_reason") == "missing_model":
                summary_data["status"] = "skipped"
            summary_data["run_index"] = global_idx
            summary_data["model_run_index"] = model_run_idx
            summary_data["log_path"] = log_path.as_posix()
            summary_data["summary_path"] = summary_path.as_posix()
            summary_data["phase_path"] = phase_path.as_posix()
            summary_data["timed_out"] = bool(proc_info.get("timed_out"))
            summary_data["process_returncode"] = returncode
            doc_info = {
                "status": None,
                "returncode": None,
                "runtime_sec": None,
            }
            if summary_data["status"] != "skipped":
                doc_info = _run_documentation_process(
                    documentation_script=documentation_script,
                    sim_name=doc_sim_name,
                    env=base_env.copy(),
                    cwd=SRC_ROOT,
                    log_path=log_path,
                )
            summary_data["documentation_status"] = doc_info["status"]
            summary_data["documentation_returncode"] = doc_info["returncode"]
            summary_data["documentation_runtime_sec"] = doc_info["runtime_sec"]
            _enrich_run_metrics(summary_data)

            _write_json(summary_path, summary_data)
            run_rows.append(summary_data)

            if summary_data["status"] == "failed" and not args.continue_on_error:
                print("[BENCH] Fail-fast attivo: interruzione benchmark.", flush=True)
                break
        else:
            continue
        break

    runs_csv = bench_data_root / "runs.csv"
    summary_csv = bench_data_root / "summary_by_model.csv"
    _write_csv(
        runs_csv,
        run_rows,
        field_order=[
            "run_index",
            "model_alias",
            "model_run_index",
            "status",
            "success",
            "exit_reason",
            "runtime_sec",
            "steps_total",
            "initial_target_distance_m",
            "min_target_distance_m",
            "final_target_distance_m",
            "success_rate_m",
            "final_approach_m",
            "target_within_0_8m",
            "time_to_target_sec",
            "action_failures_count",
            "json_parse_failures_count",
            "semantic_validation_failures_count",
            "retry_attempts_total",
            "vlm_inference_mean_sec",
            "vlm_inference_p95_sec",
            "model_load_sec",
            "timed_out",
            "process_returncode",
            "documentation_sim_name",
            "documentation_status",
            "documentation_returncode",
            "documentation_runtime_sec",
            "summary_path",
            "log_path",
        ],
    )

    summary_rows = _aggregate_by_model(run_rows)
    _write_csv(summary_csv, summary_rows)
    if plt is None:
        print("[WARN] matplotlib non disponibile: salto grafici benchmark.", flush=True)
    else:
        plot_jobs: list[tuple[str, Any, Path]] = [
            ("overview", _plot_overview, bench_data_root / "benchmark_overview.png"),
            ("score_per_run", _plot_score_per_run, bench_data_root / "score_per_run.png"),
            (
                "score_distribution",
                _plot_score_distribution,
                bench_data_root / "score_distribution_boxplot.png",
            ),
            ("distance_scatter", _plot_distance_scatter, bench_data_root / "distance_scatter.png"),
            ("threshold_rate", _plot_threshold_rate, bench_data_root / "threshold_rate.png"),
        ]
        for label, plot_fn, out_path in plot_jobs:
            try:
                if label in {"score_per_run", "score_distribution", "distance_scatter"}:
                    plot_fn(run_rows, out_path)
                else:
                    plot_fn(summary_rows, out_path)
            except Exception as exc:
                print(f"[WARN] Grafico '{label}' fallito: {exc}", flush=True)

    print(f"[BENCH] Completato. Root: {output_root}", flush=True)
    print(f"[BENCH] Cartella dati: {bench_data_root}", flush=True)
    print(f"[BENCH] Runs CSV: {runs_csv}", flush=True)
    print(f"[BENCH] Summary CSV: {summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
