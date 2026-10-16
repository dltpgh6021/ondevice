# Recovery tasks v3: 20 scenarios x horizons 1 and 3

The 40 recovery fixtures retain the original question, target, and provided context of their matched h1 and h3 execution tasks.
Both branches use the original requirement to leave exactly one saved result satisfying the request.
No extra user message reveals the error or the reference answer.

| Horizon | Correct history before the checkpoint | Injected wrong continuation | Minimum normal calls | Minimum repair calls |
|---|---|---|---|---|
| 1 | Request, candidates, two details, and three prepared drafts (one correct), as in free h1 | Confirm the constraint-violating draft | 1 | 1 (confirm the correct draft) |
| 3 | Request and candidates, as in free h3 | Wrong details -> wrong draft -> confirm | 3 | 3 (correct details -> draft -> confirm) |

Horizon counts remaining normal workflow dependencies from the common checkpoint. Minimum repair calls equal the normal remaining calls, so recovery cost does not differ from normal work by construction.

There is no h5 recovery fixture. At h5 the checkpoint has no history, so normal_restart would equal free h5, and a fully injected wrong workflow would leave the same request and candidates as the h3 error branch. h5 is analyzed with free execution only.

## Matched checkpoint branches

Both branches replay the matched free task's provided context as the model's own earlier tool calls, so free and normal_restart hold the same information and differ only in delivery.
The error branch additionally continues from the checkpoint with the constraint-violating candidate (with the requested option kind) and confirms it for the same request.
This is the same mistake that free execution scores as saved_result_violates_request, so injected recovery and natural self-correction can be compared.
Confirming a correct draft for the same request replaces the wrong saved result. Discarding first and then confirming is also valid.
Both branches share the checkpoint, correct prefix, tool schemas, original question, final goal, and response-turn budget.
One model therefore needs 80 recovery continuations: 40 normal_restart and 40 error_restart.

## Scenario list

Every base scenario has matched h1 and h3 recovery fixtures.

| Base scenario | Domain | Common injected error |
|---|---|---|
| shopping_001 | shopping | Constraint-violating candidate saved for the same request |
| shopping_002 | shopping | Constraint-violating candidate saved for the same request |
| shopping_003 | shopping | Constraint-violating candidate saved for the same request |
| shopping_004 | shopping | Constraint-violating candidate saved for the same request |
| shopping_005 | shopping | Constraint-violating candidate saved for the same request |
| schedule_001 | schedule | Constraint-violating candidate saved for the same request |
| schedule_002 | schedule | Constraint-violating candidate saved for the same request |
| schedule_003 | schedule | Constraint-violating candidate saved for the same request |
| schedule_004 | schedule | Constraint-violating candidate saved for the same request |
| schedule_005 | schedule | Constraint-violating candidate saved for the same request |
| travel_001 | travel | Constraint-violating candidate saved for the same request |
| travel_002 | travel | Constraint-violating candidate saved for the same request |
| travel_003 | travel | Constraint-violating candidate saved for the same request |
| travel_004 | travel | Constraint-violating candidate saved for the same request |
| travel_005 | travel | Constraint-violating candidate saved for the same request |
| place_001 | place | Constraint-violating candidate saved for the same request |
| place_002 | place | Constraint-violating candidate saved for the same request |
| place_003 | place | Constraint-violating candidate saved for the same request |
| place_004 | place | Constraint-violating candidate saved for the same request |
| place_005 | place | Constraint-violating candidate saved for the same request |

## Interpretation and limits

This is one controlled error family: a request-violating candidate saved for the requested ID. Do not generalize to every error type.
In every scenario the violating candidate ranks first when constraints are ignored, so the injected error is the plausible mistake of skipping a constraint check.
The injected wrong continuation is supplied history, not a naturally generated session; its length differs by horizon (1 and 3 calls).
Compute recovery penalty only between normal_restart and error_restart of the same recovery fixture, not by subtracting recovery success from execution free success.
The horizon conditions are matched observations of 20 base scenarios, not 40 independent scenarios.
All supplied questions, request descriptions, candidate labels, titles, tool schemas, and injected observations use English. Metadata records language=en and dataset_version=workflow-en-v3.
