"""
Shared utilities for belief experiments.

This package provides dataclasses, data collection, I/O, scenario management,
and summary statistics used by experiment runners and analysis scripts.

Modules:
    data      -- StepRecord, ExperimentResult dataclasses
    collect   -- Per-step diagnostic collection
    io        -- Results persistence and run directory management
    scenario  -- Agent creation, config expansion, spawn utilities
    summary   -- Summary statistics and batch aggregation
"""

from .data import StepRecord, ExperimentResult

from .collect import collect_step

from .io import (
    RESULTS_DIR,
    dump_results,
    make_run_dir,
    build_run_metadata,
    save_experiment,
    save_summary,
)

from .scenario import (
    generate_random_frame,
    create_agent,
    is_new_format,
    expand_new_config,
    expand_static_groups,
    check_viability,
    sample_viable_config,
    print_scene_summary,
    plot_spawn_preview,
)

from .summary import (
    build_summary,
    build_batch_summary,
)
