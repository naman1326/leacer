"""
scenario_config.py -- Single Source of Truth for the Active SUMO Scenario
==============================================================================
Every runner (leacer_runner, baseline_classical, baseline_dqn, baseline_marl,
baseline_gcn) imports SUMOCFG_PATH and NET_FILE_PATH from here instead of
hardcoding its own path. This is the direct fix for two confirmed bugs:

  1. leacer_runner.py's RoadGraph() was called with no arguments, so it
     always loaded the OLD 13-junction grid's net.xml even when the actual
     TraCI simulation (via SUMOEnv) was running MoST -- every vehicle's
     edge ID looked up against the wrong graph, returning None, so LEACER
     silently executed zero reroutes.

  2. baseline_classical.py, baseline_dqn.py, baseline_marl.py, and
     baseline_gcn.py all still hardcoded sumo_cfg/leacer.sumocfg directly,
     so a MoST comparison run would have evaluated LEACER on Monaco for
     7,200 steps while every baseline silently ran the old grid for 3,600 --
     not a valid comparison at all.

Every file MUST import from here rather than building its own path, so it
is now structurally impossible for runners to disagree about which network
is active. To switch scenarios, change ONLY the _ACTIVE line below.
"""

from pathlib import Path

SIM_DIR = Path(__file__).parent

# ---- ACTIVE SCENARIO -- the only line you should ever need to change ----
_ACTIVE = "most"   # "most" (Monaco rush-hour) or "grid" (original 13-junction network)

_SCENARIOS = {
    "most": dict(
        sumocfg=SIM_DIR / "most_scenario" / "most_rushhour.sumocfg",
        net_file=SIM_DIR / "most_scenario" / "in" / "most.net.xml",
    ),
    "grid": dict(
        sumocfg=SIM_DIR / "sumo_cfg" / "leacer.sumocfg",
        net_file=SIM_DIR / "sumo_cfg" / "leacer_network.net.xml",
    ),
}

if _ACTIVE not in _SCENARIOS:
    raise ValueError(f"Unknown scenario '{_ACTIVE}' -- choose one of {list(_SCENARIOS)}")

SUMOCFG_PATH  = _SCENARIOS[_ACTIVE]["sumocfg"]
NET_FILE_PATH = _SCENARIOS[_ACTIVE]["net_file"]

if not SUMOCFG_PATH.exists():
    raise FileNotFoundError(
        f"Active scenario is '{_ACTIVE}' but {SUMOCFG_PATH} does not exist. "
        f"Check the path or switch _ACTIVE in scenario_config.py."
    )
if not NET_FILE_PATH.exists():
    raise FileNotFoundError(
        f"Active scenario is '{_ACTIVE}' but {NET_FILE_PATH} does not exist."
    )

print(f"[scenario_config] Active scenario: '{_ACTIVE}'  "
      f"(cfg={SUMOCFG_PATH.name}, net={NET_FILE_PATH.name})")
