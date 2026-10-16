# Recovery tasks v5: 60 scenarios x horizons 1 and 3 x 2 error kinds

The 120 recovery fixtures retain the original question, target, and provided context of their matched h1 and h3 execution tasks.
Each fixture holds 2 error variants injected at the same checkpoint: violating_candidate, wrong_option.
Every branch uses the original requirement to leave exactly one saved result satisfying the request.
In normal_restart and error_restart no extra user message reveals the error or the reference answer.
signaled_error_restart is the signaled-error control: the same error history followed by one fixed, generic audit message that names no candidate, option, constraint, or answer:
> Automated audit notice: the result currently saved for this request does not satisfy the request. Continue working on the original request.

## Error kinds

| Error kind | Horizon | Correct history before the checkpoint | Injected wrong continuation | Minimum normal calls | Minimum repair calls |
|---|---|---|---|---|---|
| violating_candidate | 1 | Request, candidates, two details, three prepared drafts (one correct), as in free h1 | Confirm the constraint-violating draft | 1 | 1 (confirm the correct draft) |
| violating_candidate | 3 | Request and candidates, as in free h3 | Violating candidate details -> draft -> confirm | 3 | 3 (correct details -> draft -> confirm) |
| wrong_option | 1 | Same as above | Confirm the correct candidate's draft with the other option kind | 1 | 1 (confirm the correct draft) |
| wrong_option | 3 | Same as above | Correct candidate details -> draft with the other option kind -> confirm | 3 | 2 (draft with the requested option -> confirm) |

Horizon counts remaining normal workflow dependencies from the common checkpoint. For violating_candidate the minimum repair equals the normal remaining calls.
For wrong_option at h3 the correct detail is already in the history, so the minimum repair is 2 calls. Report recovery separately per error kind and do not pool the two kinds across horizons.

There is no h5 recovery fixture. At h5 the checkpoint has no history, so normal_restart would equal free h5, and a fully injected wrong workflow would leave the same request and candidates as the h3 error branch. h5 is analyzed with free execution only.

## Matched checkpoint branches

Every branch replays the matched free task's provided context as the model's own earlier tool calls, so free and normal_restart hold the same information and differ only in delivery.
The error branches additionally continue from the checkpoint with one wrong continuation and confirm it for the same request.
Both error kinds are mistakes that free execution scores as saved_result_violates_request, so injected recovery and natural self-correction can be compared.
Confirming a correct draft for the same request replaces the wrong saved result. Discarding first and then confirming is also valid.
All branches share the checkpoint, correct prefix, tool schemas, original question, final goal, and response-turn budget.
error_restart (silent error) and signaled_error_restart differ only by the final audit message, so their difference measures how much an explicit error signal helps.
One model therefore needs 600 recovery continuations: 120 normal_restart, plus 120 each of error_restart and signaled_error_restart for every error kind.

## Scenario list

Every base scenario has matched h1 and h3 recovery fixtures with both error kinds.

| Base scenario | Domain |
|---|---|
| shopping_001 | shopping |
| shopping_002 | shopping |
| shopping_003 | shopping |
| shopping_004 | shopping |
| shopping_005 | shopping |
| shopping_006 | shopping |
| shopping_007 | shopping |
| shopping_008 | shopping |
| shopping_009 | shopping |
| shopping_010 | shopping |
| shopping_011 | shopping |
| shopping_012 | shopping |
| shopping_013 | shopping |
| shopping_014 | shopping |
| shopping_015 | shopping |
| schedule_001 | schedule |
| schedule_002 | schedule |
| schedule_003 | schedule |
| schedule_004 | schedule |
| schedule_005 | schedule |
| schedule_006 | schedule |
| schedule_007 | schedule |
| schedule_008 | schedule |
| schedule_009 | schedule |
| schedule_010 | schedule |
| schedule_011 | schedule |
| schedule_012 | schedule |
| schedule_013 | schedule |
| schedule_014 | schedule |
| schedule_015 | schedule |
| travel_001 | travel |
| travel_002 | travel |
| travel_003 | travel |
| travel_004 | travel |
| travel_005 | travel |
| travel_006 | travel |
| travel_007 | travel |
| travel_008 | travel |
| travel_009 | travel |
| travel_010 | travel |
| travel_011 | travel |
| travel_012 | travel |
| travel_013 | travel |
| travel_014 | travel |
| travel_015 | travel |
| place_001 | place |
| place_002 | place |
| place_003 | place |
| place_004 | place |
| place_005 | place |
| place_006 | place |
| place_007 | place |
| place_008 | place |
| place_009 | place |
| place_010 | place |
| place_011 | place |
| place_012 | place |
| place_013 | place |
| place_014 | place |
| place_015 | place |

## Interpretation and limits

These are two controlled error families of one kind of mistake (a wrong result saved for the requested ID). Do not generalize to every error type.
In every scenario the violating candidate ranks first when constraints are ignored, so violating_candidate is the plausible mistake of skipping a constraint check.
wrong_option keeps the correct candidate and changes only the option kind, so it tests whether the option requirement is rechecked.
The injected wrong continuation is supplied history, not a naturally generated session; its length differs by horizon (1 and 3 calls).
Compute recovery penalty only between normal_restart and an error branch of the same recovery fixture, not by subtracting recovery success from execution free success.
The horizon conditions are matched observations of 60 base scenarios, not 120 independent scenarios.
All supplied questions, request descriptions, candidate labels, titles, tool schemas, and injected observations use English. Metadata records language=en and dataset_version=workflow-en-v4.
