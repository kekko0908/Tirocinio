"""Genera benchmark sintetico e dashboard grafico Gemma vs Qwen."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Architettura.paths import ARTIFACTS_ROOT

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

MODELS = ("qwen-vl2.5", "gemma-3")
COLORS = {
    "qwen-vl2.5": "#1f77b4",
    "gemma-3": "#ff7f0e",
}


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea un benchmark sintetico con Gemma-3 dominante."
    )
    parser.add_argument(
        "--runs-per-model",
        type=int,
        default=30,
        help="Numero di simulazioni finte per modello.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed RNG per risultati riproducibili.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Se vuoto usa Artefatti/benchmark_data_X/synthetic_*.",
    )
    return parser


def _sample_run(rng: random.Random, model: str, model_run_index: int, run_index: int) -> dict[str, Any]:
    # Distanza iniziale plausibile in AI2-THOR per target nel campo visivo esteso.
    initial = _clamp(rng.gauss(2.5, 0.55), 1.2, 4.2)

    if model == "gemma-3":
        near_05 = rng.random() < 0.70
        near_08 = (not near_05) and (rng.random() < 0.45)
        if near_05:
            min_dist = _clamp(rng.gauss(0.48, 0.035), 0.38, 0.50)
        elif near_08:
            min_dist = _clamp(rng.gauss(0.68, 0.07), 0.52, 0.80)
        else:
            min_dist = _clamp(rng.gauss(1.08, 0.22), 0.82, 1.80)
        action_fail = rng.randint(0, 3)
        json_fail = 1 if rng.random() < 0.015 else 0
        semantic_fail = rng.randint(0, 2) if rng.random() < 0.20 else 0
        retries = rng.randint(0, 2)
        steps = int(_clamp(rng.gauss(72, 15), 35, 130))
    else:
        near_05 = rng.random() < 0.42
        near_08 = (not near_05) and (rng.random() < 0.30)
        if near_05:
            min_dist = _clamp(rng.gauss(0.49, 0.025), 0.40, 0.50)
        elif near_08:
            min_dist = _clamp(rng.gauss(0.74, 0.08), 0.58, 0.80)
        else:
            min_dist = _clamp(rng.gauss(1.26, 0.30), 0.82, 2.10)
        action_fail = rng.randint(1, 5)
        json_fail = 1 if rng.random() < 0.04 else 0
        semantic_fail = rng.randint(0, 3) if rng.random() < 0.32 else 0
        retries = rng.randint(1, 4)
        steps = int(_clamp(rng.gauss(88, 18), 40, 160))

    # Finale leggermente peggiore o uguale alla minima distanza toccata.
    final_dist = _clamp(min_dist + abs(rng.gauss(0.10, 0.08)), min_dist, initial)
    success_rate_m = _clamp(initial - min_dist, 0.0, 5.0)
    final_approach_m = _clamp(initial - final_dist, 0.0, 5.0)
    target_within_0_5m = min_dist <= 0.5

    # Delta coerente con il successo: se resti lontano dalla soglia 0.5m,
    # il progresso "utile" viene attenuato anche se il delta grezzo e' alto.
    miss_over_threshold = max(0.0, min_dist - 0.5)
    coherent_factor = _clamp(1.0 - (miss_over_threshold / 1.8), 0.2, 1.0)
    coherent_success_rate_m = _clamp(success_rate_m * coherent_factor, 0.0, 5.0)

    success = bool(target_within_0_5m and json_fail == 0)

    miss_m = max(0.0, min_dist - 0.5)
    # Penalita' condivise: errori e distanza finale influenzano sempre la qualita'.
    penalty = (
        action_fail * 2.8
        + semantic_fail * 4.8
        + json_fail * 10.0
        + retries * 1.7
        + miss_m * 16.0
    )

    # Range target richiesti:
    # - Gemma: box ~60-75, massimo 90
    # - Qwen:  box ~50-62, massimo 72
    if model == "gemma-3":
        decision_quality = 76.5 - (penalty * 0.90) + rng.gauss(0.4, 5.8)
        if success and action_fail <= 1 and semantic_fail == 0 and retries <= 1:
            decision_quality += 4.5
        decision_quality = _clamp(decision_quality, 42.0, 90.0)
    else:
        decision_quality = 71.2 - (penalty * 0.82) + rng.gauss(0.0, 4.1)
        if success and action_fail <= 2 and semantic_fail <= 1 and retries <= 2:
            decision_quality += 2.5
        decision_quality = _clamp(decision_quality, 34.0, 72.0)

    # Coerenza: una run fallita resta sotto il top range.
    if not success:
        fail_cap = 84.0 if model == "gemma-3" else 68.5
        decision_quality = min(decision_quality, fail_cap)

    delta_vs_05 = 0.5 - min_dist

    return {
        "run_index": run_index,
        "model_run_index": model_run_index,
        "model_alias": model,
        "status": "completed",
        "steps_total": steps,
        "initial_target_distance_m": round(initial, 4),
        "min_target_distance_m": round(min_dist, 4),
        "final_target_distance_m": round(final_dist, 4),
        "success_rate_m": round(success_rate_m, 4),
        "success_rate_m_coherent": round(coherent_success_rate_m, 4),
        "final_approach_m": round(final_approach_m, 4),
        "delta_vs_0_5m": round(delta_vs_05, 4),
        "target_within_0_5m": target_within_0_5m,
        "action_failures_count": action_fail,
        "json_parse_failures_count": json_fail,
        "semantic_validation_failures_count": semantic_fail,
        "retry_attempts_total": retries,
        "decision_quality_score": round(decision_quality, 2),
        "success": success,
    }


def _build_runs(runs_per_model: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    global_idx = 0
    for model in MODELS:
        for model_run_idx in range(1, runs_per_model + 1):
            global_idx += 1
            rows.append(_sample_run(rng, model=model, model_run_index=model_run_idx, run_index=global_idx))
    _enforce_quality_max(rows)
    return rows


def _enforce_quality_max(rows: list[dict[str, Any]]) -> None:
    """Forza il massimo quality su una run riuscita per ciascun modello."""
    target_max = {
        "gemma-3": 90.0,
        "qwen-vl2.5": 72.0,
    }
    for model, target in target_max.items():
        candidates = [
            idx for idx, row in enumerate(rows)
            if row.get("model_alias") == model and bool(row.get("success"))
        ]
        if not candidates:
            continue
        best_idx = max(candidates, key=lambda idx: float(rows[idx].get("decision_quality_score", 0.0)))
        rows[best_idx]["decision_quality_score"] = round(target, 2)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [r for r in rows if r["model_alias"] == model]
        n = len(model_rows)
        within_05 = sum(1 for r in model_rows if bool(r["target_within_0_5m"]))
        success_n = sum(1 for r in model_rows if bool(r["success"]))
        delta_vals = [float(r.get("success_rate_m_coherent", r["success_rate_m"])) for r in model_rows]
        quality_vals = [float(r["decision_quality_score"]) for r in model_rows]
        out.append(
            {
                "model_alias": model,
                "runs_total": n,
                "target_within_0_5m_pct": round((within_05 * 100.0 / n) if n else 0.0, 2),
                "success_pct": round((success_n * 100.0 / n) if n else 0.0, 2),
                "decision_quality_mean": round(_mean(quality_vals), 3),
                "decision_quality_median": round(_median(quality_vals), 3),
                "delta_success_rate_m_mean": round(_mean(delta_vals), 4),
                "delta_success_rate_m_median": round(_median(delta_vals), 4),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_dashboard(run_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    if plt is None:
        print("[WARN] matplotlib non disponibile, grafico non generato.", flush=True)
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    models = [row["model_alias"] for row in summary_rows]
    x = list(range(len(models)))

    within05 = [row["target_within_0_5m_pct"] for row in summary_rows]
    ax1.bar(x, within05, width=0.52, color="#17becf", label="<= 0.5m")
    ax1.set_xticks(x, models)
    ax1.set_ylim(0, 100)
    ax1.set_title("Success Rate")
    ax1.set_ylabel("Run (%)")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.legend()

    quality_values = []
    quality_labels = []
    for model in MODELS:
        vals = [float(r["decision_quality_score"]) for r in run_rows if r["model_alias"] == model]
        if vals:
            quality_values.append(vals)
            quality_labels.append(model)
    if quality_values:
        box = ax2.boxplot(quality_values, tick_labels=quality_labels, patch_artist=True)
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(COLORS.get(quality_labels[i], "#666666"))
            patch.set_alpha(0.65)
        # Mostra esplicitamente il massimo per evidenziare gli estremi superiori.
        max_points = [max(vals) for vals in quality_values]
        ax2.scatter(
            list(range(1, len(max_points) + 1)),
            max_points,
            color="#222222",
            marker="D",
            s=28,
            zorder=4,
            label="Massimo",
        )
    ax2.set_title("Qualita' Scelte (Decision Quality)")
    ax2.set_ylabel("Score (0-100)")
    ax2.grid(axis="y", linestyle="--", alpha=0.25)

    delta_values = []
    delta_labels = []
    for model in MODELS:
        vals = [
            float(r.get("success_rate_m_coherent", r["success_rate_m"]))
            for r in run_rows
            if r["model_alias"] == model
        ]
        if vals:
            delta_values.append(vals)
            delta_labels.append(model)
    if delta_values:
        violin = ax3.violinplot(delta_values, showmeans=True, showmedians=True)
        for idx, body in enumerate(violin["bodies"]):
            body.set_facecolor(COLORS.get(delta_labels[idx], "#666666"))
            body.set_alpha(0.55)
    ax3.set_xticks(list(range(1, len(delta_labels) + 1)), delta_labels)
    ax3.set_title("Delta Avvicinamento Coerente")
    ax3.set_ylabel("Metri guadagnati")
    ax3.grid(axis="y", linestyle="--", alpha=0.25)

    success_pct = [row["success_pct"] for row in summary_rows]
    quality_mean = [row["decision_quality_mean"] for row in summary_rows]
    width = 0.35
    ax4.bar([v - width / 2 for v in x], success_pct, width=width, color="#9467bd", label="Successo complessivo %")
    ax4.bar([v + width / 2 for v in x], quality_mean, width=width, color="#8c564b", label="Quality medio")
    ax4.set_xticks(x, models)
    ax4.set_ylim(0, 100)
    ax4.set_title("Successo vs Qualita' Scelte")
    ax4.grid(axis="y", linestyle="--", alpha=0.25)
    ax4.legend()

    fig.suptitle("Dashboard Benchmark - Gemma-3 vs Qwen2.5", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), dpi=180)
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()
    if args.runs_per_model < 1:
        raise SystemExit("--runs-per-model deve essere >= 1")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = (
            ARTIFACTS_ROOT
            / "benchmark_data_X"
            / f"synthetic_gemma_vs_qwen_{_utc_tag()}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = _build_runs(runs_per_model=args.runs_per_model, seed=args.seed)
    summary_rows = _aggregate(run_rows)

    runs_csv = out_dir / "runs_synthetic.csv"
    summary_csv = out_dir / "summary_synthetic.csv"
    summary_json = out_dir / "summary_synthetic.json"
    dashboard_png = out_dir / "synthetic_dashboard.png"

    _write_csv(runs_csv, run_rows)
    _write_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=True), encoding="utf-8")
    _plot_dashboard(run_rows, summary_rows, dashboard_png)

    print(f"[SYNTH] Output dir: {out_dir}")
    print(f"[SYNTH] Runs CSV: {runs_csv}")
    print(f"[SYNTH] Summary CSV: {summary_csv}")
    print(f"[SYNTH] Summary JSON: {summary_json}")
    if plt is not None:
        print(f"[SYNTH] Dashboard: {dashboard_png}")
    else:
        print("[SYNTH] Dashboard non creato: matplotlib non disponibile.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
