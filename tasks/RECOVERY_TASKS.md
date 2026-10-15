# Recovery tasks v2: 20 scenarios x horizons 1, 3, and 5

The 60 recovery fixtures retain the original question, target, and correct history of their matched execution-v2 tasks.
Both branches use the original requirement to leave exactly one saved result satisfying the request.
No extra user message reveals the error or the reference answer.

| Nominal horizon | Minimum normal calls | Minimum repair calls |
|---|---|---|
| 1 | 1 | 2 |
| 3 | 3 | 4 |
| 5 | 5 | 6 |

Horizon counts remaining normal workflow dependencies. The error branch additionally needs one cleanup call.
Record normal_remaining_calls and recovery_min_calls separately.

## Matched checkpoint branches

A stale draft for another request is a supplied artifact in the common starting fixture. It does not pre-create the target draft.
Replay the same initial_actions to reach the common checkpoint. The normal branch starts there. The error branch additionally confirms the stale draft and appends the real assistant call and tool result to the history.
The stale draft contains an ineligible candidate and an undesired option under a distinct opaque request_id.
Saving the correct target does not remove the stale result. The agent must discard_saved_result using the stale request_id. Cleanup before or after the normal workflow is valid.
Both branches share the checkpoint, correct history, tool schemas, original question, final goal, and response-turn budget. Only the error branch receives the injected saved result and its observable history.
One model therefore needs 120 recovery continuations: 60 normal_restart and 60 error_restart.

## Scenario list

Every base scenario has matched h1, h3, and h5 recovery fixtures.

| Base scenario | Domain | Common injected error |
|---|---|---|
| shopping_001 | shopping | Extra saved record violating the request |
| shopping_002 | shopping | Extra saved record violating the request |
| shopping_003 | shopping | Extra saved record violating the request |
| shopping_004 | shopping | Extra saved record violating the request |
| shopping_005 | shopping | Extra saved record violating the request |
| schedule_001 | schedule | Extra saved record violating the request |
| schedule_002 | schedule | Extra saved record violating the request |
| schedule_003 | schedule | Extra saved record violating the request |
| schedule_004 | schedule | Extra saved record violating the request |
| schedule_005 | schedule | Extra saved record violating the request |
| travel_001 | travel | Extra saved record violating the request |
| travel_002 | travel | Extra saved record violating the request |
| travel_003 | travel | Extra saved record violating the request |
| travel_004 | travel | Extra saved record violating the request |
| travel_005 | travel | Extra saved record violating the request |
| place_001 | place | Extra saved record violating the request |
| place_002 | place | Extra saved record violating the request |
| place_003 | place | Extra saved record violating the request |
| place_004 | place | Extra saved record violating the request |
| place_005 | place | Extra saved record violating the request |

## Interpretation and limits

This is one controlled error family: an extra saved result for a wrong request. Do not generalize to every error type or naturally generated agent failure.
The model must recognize and remove the erroneous record. Argument correction, exception handling, and other recovery mechanisms are outside this fixture family.
Compute recovery penalty only between normal_restart and error_restart of the same recovery fixture, not by subtracting recovery success from execution free success.
The horizon conditions are matched observations of 20 base scenarios, not 60 independent scenarios.
All supplied questions, request descriptions, candidate labels, titles, tool schemas, and injected observations use English. Metadata records language=en and dataset_version=workflow-en-v1.
