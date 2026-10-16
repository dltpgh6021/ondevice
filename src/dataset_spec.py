"""Single source of truth for the dataset version and its expected shape.

Generators, validators, tests and documentation read these values so the
scenario count can change in one place.
"""

from __future__ import annotations

DATASET_VERSION = "workflow-en-v4"
DOMAINS = ("shopping", "schedule", "travel", "place")
SCENARIOS_PER_DOMAIN = 15
EXECUTION_HORIZONS = (1, 3, 5)
RECOVERY_HORIZONS = (1, 3)

BASE_SCENARIOS = SCENARIOS_PER_DOMAIN * len(DOMAINS)
EXECUTION_TASKS = BASE_SCENARIOS * len(EXECUTION_HORIZONS)
RECOVERY_TASKS = BASE_SCENARIOS * len(RECOVERY_HORIZONS)
TOTAL_TASKS = EXECUTION_TASKS + RECOVERY_TASKS

# Error families injected at the same checkpoint of every recovery fixture.
# violating_candidate: a constraint-violating candidate saved with the requested option.
# wrong_option: the correct candidate saved with the other option kind.
ERROR_KINDS = ("violating_candidate", "wrong_option")
DEFAULT_ERROR_KIND = ERROR_KINDS[0]

# Conditions that run on one recovery fixture. Every one shares the checkpoint.
# normal_restart runs once per fixture; each error condition runs once per error kind.
RECOVERY_ERROR_CONDITIONS = ("error_restart", "signaled_error_restart")
RECOVERY_CONDITIONS = ("normal_restart",) + RECOVERY_ERROR_CONDITIONS
RUNS_PER_MODEL = EXECUTION_TASKS + RECOVERY_TASKS * (1 + len(RECOVERY_ERROR_CONDITIONS) * len(ERROR_KINDS))
