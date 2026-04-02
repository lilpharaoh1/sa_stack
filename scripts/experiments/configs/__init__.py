"""Experiment configuration loading with YAML defaults."""

import os
import yaml

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    """Load a YAML config, resolving ``_base`` inheritance.

    If the config contains a ``_base`` key, the base config is loaded first
    and the current config's values are merged on top (shallow per-section).

    Args:
        path: Path to the YAML file (absolute or relative to configs dir).

    Returns:
        Merged config dict.
    """
    if not os.path.isabs(path):
        path = os.path.join(_CONFIGS_DIR, path)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_path = cfg.pop("_base", None)
    if base_path is not None:
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base = load_config(base_path)
        # Shallow merge: each top-level section is updated independently
        for section, values in cfg.items():
            if isinstance(values, dict) and section in base and isinstance(base[section], dict):
                base[section].update(values)
            else:
                base[section] = values
        cfg = base

    return cfg


def apply_config(cfg: dict, args) -> None:
    """Set method parameters on args namespace from the YAML config.

    These parameters are solely controlled by the config file — there
    are no corresponding CLI flags.

    Args:
        cfg: Loaded YAML config dict.
        args: argparse.Namespace from parse_args().
    """
    mapping = {
        # inference
        ("inference", "type"):              "inference_type",
        ("inference", "relevance_method"):  "relevance_method",
        # intervention
        ("intervention", "type"):           "intervention_type",
        ("intervention", "ref_controls"):   "ref_controls",
        # planning
        ("planning", "mode"):               "planning_mode",
        # velocity
        ("velocity", "n_kappa_particles"):  "n_kappa_particles",
        ("velocity", "kappa_min"):          "kappa_min",
        ("velocity", "b_kappa"):            "b_kappa",
        ("velocity", "q_kappa"):            "q_kappa",
    }

    for (section, key), attr in mapping.items():
        if section in cfg and key in cfg[section]:
            setattr(args, attr, cfg[section][key])
