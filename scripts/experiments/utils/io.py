"""I/O and persistence utilities for belief experiments."""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime

import dill

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "results")


def dump_results(result: ExperimentResult, name: str):
    """Save experiment results to pickle."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, name + ".pkl")
    with open(filepath, 'wb') as f:
        dill.dump(result, f)
    logger.info("Results saved to %s", filepath)
    return filepath


def make_run_dir(scenario_name: str,
                 config_name: str = "defaults",
                 seed: int = 0,
                 n_samples: int = None,
                 custom_name: str = None) -> str:
    """Build a unique folder path under ``RESULTS_DIR``.

    Folder naming:
      - Batch: ``{scenario}_{config}_seed{seed}_n{n_samples}_{timestamp}``
      - Single: ``{scenario}_{config}_seed{seed}_{timestamp}``
      - If *custom_name* is given, use it as the folder name directly.
    """
    if custom_name:
        folder = custom_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Strip .yaml extension from config name
        cfg_label = config_name.replace(".yaml", "").replace(".yml", "")
        if n_samples is not None:
            folder = f"{scenario_name}_{cfg_label}_seed{seed}_n{n_samples}_{ts}"
        else:
            folder = f"{scenario_name}_{cfg_label}_seed{seed}_{ts}"
    return os.path.join(RESULTS_DIR, folder)


def build_run_metadata(args, config: dict) -> dict:
    """Construct metadata dict from parsed argparse Namespace + expanded config.

    Args:
        args: argparse.Namespace with at least ``map``, ``seed``, ``steps``,
              ``intervention_type``.  May also have ``n_samples`` (batch mode).
        config: The full (expanded) scenario config dict.
    """
    n_samples = getattr(args, 'n_samples', None)
    return {
        "mode": "batch" if n_samples is not None else "single",
        "scenario": args.map,
        "seed": args.seed,
        "max_steps": args.steps,
        "intervention_type": args.intervention_type,
        "inference_type": getattr(args, 'inference_type', 'naive'),
        "planning_mode": getattr(args, 'planning_mode', '2d'),
        "ref_controls": getattr(args, 'ref_controls', 'opt'),
        "human_type": getattr(args, 'human_type', 'static'),
        "n_kappa_particles": getattr(args, 'n_kappa_particles', 0),
        "config_name": getattr(args, 'config', 'defaults.yaml'),
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "config": config,
    }


def save_experiment(result_or_batch, run_dir: str, metadata: dict) -> str:
    """Create the run directory, write ``metadata.json`` and ``results.pkl``.

    Args:
        result_or_batch: An :class:`ExperimentResult` (single) or a batch dict.
        run_dir: Absolute path to the run directory.
        metadata: Dict returned by :func:`build_run_metadata`.

    Returns:
        Path to the written ``results.pkl``.
    """
    os.makedirs(run_dir, exist_ok=True)

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    pkl_path = os.path.join(run_dir, "results.pkl")
    with open(pkl_path, 'wb') as f:
        dill.dump(result_or_batch, f)

    logger.info("Experiment saved to %s", run_dir)
    return pkl_path


def save_summary(summary: dict, run_dir: str) -> str:
    """Write a summary dict to ``summary.json`` in the run directory."""
    path = os.path.join(run_dir, "summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary saved to %s", path)
    return path
