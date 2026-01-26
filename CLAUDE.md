# CLAUDE.md - Project Guide for IGP2

## Project Overview

IGP2 (Interpretable Goal-based Prediction and Planning) is an autonomous driving framework that combines goal recognition with Monte Carlo Tree Search (MCTS) planning. This fork extends the original IGP2 with neural network trajectory prediction (PGP) and experimental scenario testing features.

## Branch Structure

This repository uses three main branches for different purposes:

| Branch | Purpose | Key Features |
|--------|---------|--------------|
| `master` | Stable framework | Core IGP2, simulators, MCTS planning, goal recognition |
| `soa-tests` | Scenario experiments | master + KeyboardAgent, SOA scenarios, failure zone testing |
| `pgp` | Neural network prediction | master + PGP module, vector maps, pgp_drive/pgp_control |

### Branch Details

**master** - Use this as the stable base for new features
- Clean CARLA and simple simulator integration
- MCTSAgent and TrafficAgent (no PGP parameters)
- Core planning and recognition systems

**soa-tests** - For keyboard-controlled driving experiments
- KeyboardAgent for manual WASD/arrow key control
- SharedAutonomyAgent with epistemic maneuver-level prediction
- SOA scenario configs (soa1.json, soa2.json, soa3.json)
- StatusWindow for failure zone monitoring
- `igp2/epistemic/` - Maneuver-level goal recognition module
- Run with: `python scripts/debug/soa_examples.py --carla -m soa1`

**pgp** - For neural network trajectory prediction work
- `igp2/pgp/` - PGP neural network models and training
- `igp2/vector_map/` - Lane graph representation
- Agents have `pgp_drive` and `pgp_control` parameters
- Run with: `python scripts/debug/carla_pgp_test.py`

## Directory Structure

```
IGP2/
├── igp2/                    # Main package
│   ├── agents/              # Agent implementations
│   │   ├── mcts_agent.py    # MCTS-based planning agent
│   │   ├── traffic_agent.py # Follows macro actions via A*
│   │   ├── keyboard_agent.py# Manual control (soa-tests only)
│   │   └── ...
│   ├── carlasim/            # CARLA simulator integration
│   │   ├── carla_client.py  # Main CarlaSim class
│   │   ├── traffic_manager.py
│   │   └── carla_agent_wrapper.py
│   ├── simplesim/           # Fast 2D simulator
│   ├── core/                # Core types (AgentState, Trajectory, Goal, etc.)
│   ├── planning/            # MCTS implementation
│   ├── planlibrary/         # Macro actions (Continue, Exit, LaneChange, etc.)
│   ├── recognition/         # Goal recognition and A* search
│   ├── epistemic/           # Maneuver-level goal recognition (soa-tests only)
│   │   ├── macro_guided_generator.py  # BFS maneuver sequence generation using macro-actions
│   │   ├── macro_guided_recognition.py # Inverse planning recognition (recommended)
│   │   ├── maneuver_astar.py       # A* search over maneuvers (legacy)
│   │   ├── maneuver_recognition.py # Direct maneuver recognition (legacy)
│   │   ├── maneuver_probabilities.py # Probability tracking for maneuvers
│   │   └── maneuver_factory.py     # Maneuver instantiation
│   ├── opendrive/           # OpenDRIVE map parsing
│   ├── pgp/                 # Neural network prediction (pgp branch only)
│   └── vector_map/          # Lane graphs (pgp branch only)
├── scenarios/
│   ├── configs/             # Scenario JSON configs
│   └── maps/                # OpenDRIVE .xodr map files
├── scripts/
│   ├── debug/               # Debug and test scripts
│   ├── experiments/         # Experiment scripts
│   └── run.py               # Main entry point
└── tests/                   # Unit tests
```

## Key Scripts

### On all branches:
- `scripts/debug/carla_test.py` - Basic CARLA simulation test
- `scripts/debug/debug_scenario.py` - Scenario-based testing
- `scripts/debug/debug_carla.py` - CARLA debugging
- `scripts/run.py` - Main simulation runner

### On soa-tests branch:
- `scripts/debug/soa_examples.py` - SOA experiment runner with keyboard control

### On pgp branch:
- `scripts/debug/debug_pgp_drive.py` - Test PGP-modified trajectories
- `scripts/debug/debug_pgp_control.py` - Test PGP control
- `scripts/debug/debug_pgp_prediction.py` - Test raw PGP predictions
- `scripts/debug/carla_pgp_test.py` - Full PGP+CARLA test

## Common Commands

```bash
# Install the package
pip install -e .

# Run CARLA simulation with a scenario
python scripts/run.py --carla -m Town01

# Run simple simulator
python scripts/run.py -m heckstrasse --plot 5

# Debug scripts (from repo root)
python scripts/debug/carla_test.py
python scripts/debug/debug_scenario.py
```

## Scenario Configuration

Scenarios are defined in JSON files under `scenarios/configs/`. Example structure:

```json
{
  "scenario": {
    "map_path": "scenarios/maps/Town01.xodr",
    "max_speed": 10.0,
    "fps": 20,
    "seed": 21
  },
  "agents": [
    {
      "id": 0,
      "type": "MCTSAgent|TrafficAgent|KeyboardAgent",
      "spawn": { "box": {...}, "velocity": [min, max] },
      "goal": { "box": {...} }
    }
  ],
  "failure_zones": [
    { "box": {...}, "frames": { "start": 0, "end": 275 } }
  ]
}
```

## Agent Types

| Agent | Description | Use Case |
|-------|-------------|----------|
| `MCTSAgent` | Plans using MCTS with goal recognition | Ego vehicle, complex decision making |
| `TrafficAgent` | Follows A* computed macro actions | Background traffic, simple navigation |
| `KeyboardAgent` | Manual WASD/arrow control | Testing, experiments (soa-tests only) |
| `SharedAutonomyAgent` | MCTS + epistemic maneuver prediction | SOA experiments (soa-tests only) |

## Key Classes

- `CarlaSim` (`igp2/carlasim/carla_client.py`) - CARLA simulator interface
- `Map` (`igp2/opendrive/map.py`) - OpenDRIVE map representation
- `MCTS` (`igp2/planning/mcts.py`) - Monte Carlo Tree Search planner
- `MacroAction` (`igp2/planlibrary/macro_action.py`) - High-level trajectory segments
- `GoalRecognition` (`igp2/recognition/goalrecognition.py`) - Infer agent goals
- `ManeuverRecognition` (`igp2/epistemic/maneuver_recognition.py`) - Maneuver-level goal recognition
- `SharedAutonomyAgent` (`igp2/agents/shared_autonomy_agent.py`) - SOA agent with epistemic prediction

## Epistemic Module (soa-tests branch)

The epistemic module (`igp2/epistemic/`) provides maneuver-level goal recognition, operating at a finer granularity than the macro-action level used by standard IGP2.

### Prediction Levels
- **Macro level** (`prediction_level="macro"`): Uses `GoalRecognition` with macro-actions (Continue, Exit, ChangeLane, Stop)
- **Maneuver level** (`prediction_level="maneuver"`): Uses `ManeuverRecognition` with maneuvers (FollowLane, Turn, GiveWay, Stop, SwitchLaneLeft/Right)

### Key Components
- `ManeuverAStar`: A* search over maneuver sequences to find paths to goals
- `ManeuverRecognition`: Compares observed trajectory against optimal maneuver trajectories
- `ManeuverProbabilities`: Tracks goal probabilities based on maneuver-level costs
- `ManeuverFactory`: Creates maneuver instances and determines applicable maneuvers

### SharedAutonomyAgent Configuration
```python
SharedAutonomyAgent(
    prediction_level="maneuver",  # or "macro"
    ego_goal_mode="true_goal",    # or "goal_recognition"
    predict_ego=True,
    human={"type": "KeyboardAgent"},  # or "TrafficAgent"
    # ... other params
)
```

### SharedAutonomyAgent Human Agent Architecture

The SharedAutonomyAgent uses a pluggable "human" agent architecture to separate the source of driver actions from the safety/prediction system:

```
┌─────────────────────────────────────────────────────────────┐
│                  SharedAutonomyAgent                         │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │   Human Agent   │───▶│  Intervention/Safety System  │   │
│  │  (configurable) │    │  - Goal Recognition (ego)    │   │
│  └─────────────────┘    │  - MCTS optimal planning     │   │
│          │              │  - Safety analysis           │   │
│          ▼              │  - Apply interventions       │   │
│    Base Actions         └──────────────────────────────┘   │
│          │                         │                        │
│          └─────────────────────────┘                        │
│                         │                                   │
│                   Final Actions                             │
└─────────────────────────────────────────────────────────────┘
```

**Supported Human Agent Types:**

| Type | Description | Use Case |
|------|-------------|----------|
| `KeyboardAgent` | Manual WASD/arrow control | Real human-in-the-loop testing |
| `TrafficAgent` | Follows A* path to goal | Simulated human (open-loop, no beliefs) |
| `EpistemicHumanAgent` | (FUTURE) Agent with beliefs | Simulated human with epistemic state |

**JSON Config Example:**
```json
{
  "id": 0,
  "type": "SharedAutonomyAgent",
  "human": {
    "type": "TrafficAgent",
    "config": {
      "open_loop": true
    }
  },
  "mcts": { ... }
}
```

The `human` key has two sub-keys:
- `type`: The agent class name (KeyboardAgent, TrafficAgent, MCTSAgent, etc.)
- `config`: All parameters to pass to the agent constructor (except agent_id, initial_state, goal, fps which are auto-provided)

**TrafficAgent open_loop option:**
- `open_loop: false` (default, recommended) - Closed-loop execution with feedback control. The agent uses maneuver-level control to follow lanes, reacting to deviations.
- `open_loop: true` - Open-loop execution (NOT fully supported yet). Would follow pre-computed trajectory without feedback control, but requires MacroAction modifications.

**Future: EpistemicHumanAgent**

The plan is to create an `EpistemicHumanAgent` that:
- Acts like a TrafficAgent (follows A* to goal) but with beliefs
- Has beliefs about other agents (e.g., assumed velocities, goals)
- Makes decisions based on those beliefs (which may be incorrect)
- Can be used to simulate human drivers with imperfect perception

Example future config:
```json
{
  "type": "SharedAutonomyAgent",
  "human": {
    "type": "EpistemicHumanAgent",
    "config": {
      "open_loop": true,
      "beliefs": {
        "agent_velocities": { "1": 5.0 },
        "agent_goals": { "1": "straight" }
      }
    }
  }
}
```

This architecture allows:
1. Testing with real human keyboard input
2. Running automated experiments with simulated humans
3. Studying how belief mismatches affect safety interventions

### Macro-Guided Recognition (Inverse Planning)

The `MacroGuidedRecognition` class uses inverse planning to determine which maneuver sequence the driver is following:

1. **Generate OPTIMAL trajectory** from initial position (benchmark)
2. **Generate CANDIDATE trajectories** from current position (possible futures)
3. **Prepend observed trajectory** to candidates (what driver DID + future)
4. **Compare using cost functions** - the candidate closest to optimal is selected
5. **Select best match** based on reward difference (not raw reward)

Key insight: Selection uses `reward_difference` (closeness to optimal) not raw `reward`, because we want to find which plan best explains the observed behavior.

## Development Notes

### Adding a new agent type:
1. Create class in `igp2/agents/` inheriting from `Agent` or `MacroAgent`
2. Implement `next_action(observation, prediction)` method
3. Add to `igp2/agents/__init__.py`
4. If CARLA-specific handling needed, update `carla_agent_wrapper.py`

### Adding a new scenario:
1. Create JSON config in `scenarios/configs/`
2. Ensure map .xodr file exists in `scenarios/maps/`
3. Run with `python scripts/run.py -m <scenario_name>`

### Coordinate systems:
- IGP2 uses (x, y) with y-positive pointing up
- CARLA uses (x, y) with y-positive pointing down
- Conversion: `igp2_y = -carla_y`

## Dependencies

- Python >= 3.8
- CARLA >= 0.9.13 (optional, for CARLA simulation)
- Key packages: casadi, shapely, scipy, matplotlib, lxml

## Testing

```bash
# Run tests
pytest tests/

# Syntax check
python -m py_compile igp2/**/*.py
```

## Troubleshooting

### CARLA connection issues
- Ensure CARLA server is running: `./CarlaUE4.sh` (Linux) or `CarlaUE4.exe` (Windows)
- Check port (default 2000): `--port 2000`

### Import errors
- Install package in editable mode: `pip install -e .`
- Check you're on the correct branch for the features you need

### AttributeError in CarlaSim.__del__
- This is handled gracefully - usually indicates __init__ failed earlier
- Check CARLA connection and map loading
