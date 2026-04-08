"""Episode data recorder for car-following experiments."""

import json
import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class EpisodeRecorder:
    """Collects per-step statistics for post-hoc analysis."""

    def __init__(self, true_vel_errors: Dict[int, float],
                 belief_candidates: np.ndarray):
        self._true_vel_errors = true_vel_errors
        self._candidates = belief_candidates

        # Build oracle distribution
        self._oracle_dist = {}
        for aid, ve in true_vel_errors.items():
            oracle = np.zeros(len(belief_candidates))
            oracle[np.argmin(np.abs(belief_candidates - ve))] = 1.0
            self._oracle_dist[aid] = oracle

        self.steps = []
        self.human_accel = []
        self.executed_accel = []
        self.jerk = []
        self.distance = []
        self.ego_speed = []
        self.lead_speed = []
        self.intervened = []
        self.kl_divergence = []
        self.belief_dists = []
        self.human_vel_err = []
        self.kf_kappa = []
        self.kf_P = []
        self.frames = []

    def record(self, step: int, info: dict, frame: dict):
        self.steps.append(step)
        self.human_accel.append(info.get("accel", 0.0))
        exec_a = info.get("executed_accel", info.get("accel", 0.0))
        self.executed_accel.append(exec_a)
        self.distance.append(info.get("distance"))
        self.ego_speed.append(info.get("ego_speed", 0.0))
        self.lead_speed.append(info.get("lead_speed", 0.0))
        self.intervened.append(info.get("intervened", False))

        if len(self.executed_accel) >= 2:
            self.jerk.append(self.executed_accel[-1] - self.executed_accel[-2])
        else:
            self.jerk.append(0.0)

        # KL divergence
        inferred_dist = info.get("inferred_dist")
        if inferred_dist is not None:
            probs = np.array(list(inferred_dist.values()))
            self.belief_dists.append(probs.tolist())
            oracle = list(self._oracle_dist.values())[0]
            eps = 1e-12
            kl = float(np.sum(oracle * np.log((oracle + eps) / (probs + eps))))
            self.kl_divergence.append(kl)
        else:
            self.belief_dists.append(None)
            self.kl_divergence.append(None)

        self.human_vel_err.append(info.get("human_vel_err"))
        self.kf_kappa.append(info.get("kf_kappa"))
        self.kf_P.append(info.get("kf_P"))

        # Frame snapshot
        frame_snap = {}
        for aid, state in frame.items():
            frame_snap[int(aid)] = {
                "position": state.position.tolist(),
                "velocity": (state.velocity.tolist()
                             if hasattr(state.velocity, 'tolist')
                             else float(state.velocity)),
                "heading": float(state.heading),
                "speed": float(state.speed),
            }
        self.frames.append(frame_snap)

    def save(self, path: str):
        data = {
            "steps": self.steps,
            "human_accel": self.human_accel,
            "executed_accel": self.executed_accel,
            "jerk": self.jerk,
            "distance": self.distance,
            "ego_speed": self.ego_speed,
            "lead_speed": self.lead_speed,
            "intervened": self.intervened,
            "kl_divergence": self.kl_divergence,
            "belief_dists": self.belief_dists,
            "belief_candidates": self._candidates.tolist(),
            "true_vel_errors": {str(k): v
                                for k, v in self._true_vel_errors.items()},
            "human_vel_err": self.human_vel_err,
            "kf_kappa": self.kf_kappa,
            "kf_P": self.kf_P,
            "frames": self.frames,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Episode data saved to %s", path)

    def print_summary(self):
        n = len(self.steps)
        if n == 0:
            return
        accel_dev = np.array(self.human_accel) - np.array(self.executed_accel)
        jerk_arr = np.array(self.jerk)
        kl_vals = [v for v in self.kl_divergence if v is not None]
        intv_count = sum(self.intervened)

        print(f"\n{'='*60}")
        print(f"  Episode Summary  ({n} steps)")
        print(f"{'='*60}")
        print(f"  Accel deviation  |  mean={np.mean(accel_dev):.4f}  "
              f"std={np.std(accel_dev):.4f}  "
              f"max={np.max(np.abs(accel_dev)):.4f}")
        print(f"  Jerk             |  mean={np.mean(jerk_arr):.4f}  "
              f"std={np.std(jerk_arr):.4f}  "
              f"max={np.max(np.abs(jerk_arr)):.4f}")
        if kl_vals:
            print(f"  KL(oracle||inf)  |  mean={np.mean(kl_vals):.4f}  "
                  f"final={kl_vals[-1]:.4f}")
        print(f"  Interventions    |  {intv_count}/{n} steps "
              f"({100*intv_count/n:.1f}%)")
        print(f"{'='*60}\n")
