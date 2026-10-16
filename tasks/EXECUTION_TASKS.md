# Execution tasks v2: 20 scenarios x horizons 1, 3, and 5

The 60 tasks are three starting-information conditions for the same 20 scenarios, rather than 60 independent questions.
Each scenario keeps the same original question. Shorter conditions seed earlier lookup results into the environment and show them in the user message.
They are never presented as the model's own earlier tool calls; that format is reserved for the matched recovery fixtures (normal_restart / error_restart).

| Horizon | Information provided in the user message | Required model calls |
|---|---|---|
| 1 | Request, candidates, two details, and three prepared drafts (one correct) | Choose and confirm the correct draft |
| 3 | Request and candidates | Read details -> prepare -> confirm |
| 5 | None | Read request -> find candidates -> read details -> prepare -> confirm |

At h1 the two wrong drafts are the best candidate with the other option kind and the constraint-violating candidate with the requested option kind, so a selection decision remains.
Confirmation requires an existing draft_id created by preparation. Lookup identifiers must likewise be issued by the preceding tools before those resources exist.
All horizons expose the same tool list. Calling inspect_state after completion does not change the final success judgment.

Every row below has three tasks with the suffixes `_h1`, `_h3`, and `_h5`. Actual questions, tool results, and reference answers are stored in tasks.json.

| Base ID | Request |
|---|---|
| shopping_001 | For the keyboard category, consider products with a base price of at most 100 and a rating of at least 4.0. Select the highest-rated product, breaking ties by the lowest price. Use the standard_delivery delivery option. |
| shopping_002 | For the mouse category, consider products with a base price of at most 60 and a rating of at least 4.2. Select the lowest-priced product, breaking ties by the highest rating. Use the express_delivery delivery option. |
| shopping_003 | For the headset category, consider products with a base price of at most 150 and a rating of at least 4.3. Select the highest-rated product, breaking ties by the lowest price. Use the standard_delivery delivery option. |
| shopping_004 | For the webcam category, consider products with a base price of at most 90 and a rating of at least 4.1. Select the highest-rated product, breaking ties by the lowest price. Use the express_delivery delivery option. |
| shopping_005 | For the usb_hub category, consider products with a base price of at most 70 and a rating of at least 4.0. Select the lowest-priced product, breaking ties by the highest rating. Use the standard_delivery delivery option. |
| schedule_001 | Schedule 'Team meeting' for one hour on 2026-10-20. Consider available slots with capacity of at least 4 and a start hour from 9 through 17, inclusive. Select the slot with the earliest start hour, breaking ties by the shortest walking time. Use the in_person meeting format. |
| schedule_002 | Schedule 'Paper reading group' for one hour on 2026-10-21. Consider available slots with capacity of at least 3 and a start hour from 10 through 18, inclusive. Select the slot with the shortest walking time, breaking ties by the earliest start hour. Use the hybrid meeting format. |
| schedule_003 | Schedule 'Experiment review meeting' for one hour on 2026-10-22. Consider available slots with capacity of at least 5 and a start hour from 9 through 16, inclusive. Select the slot with the earliest start hour, breaking ties by the shortest walking time. Use the in_person meeting format. |
| schedule_004 | Schedule 'Project presentation rehearsal' for one hour on 2026-10-23. Consider available slots with capacity of at least 5 and a start hour from 13 through 19, inclusive. Select the slot with the shortest walking time, breaking ties by the earliest start hour. Use the hybrid meeting format. |
| schedule_005 | Schedule 'Weekly retrospective' for one hour on 2026-10-24. Consider available slots with capacity of at least 4 and a start hour from 11 through 18, inclusive. Select the slot with the earliest start hour, breaking ties by the shortest walking time. Use the in_person meeting format. |
| travel_001 | Travel from Seoul to Busan on 2026-10-20. Consider options with a base fare of at most 100 and a departure hour from 8 through 12, inclusive. Select the option with the shortest travel duration, breaking ties by the lowest base fare. Use the refundable fare option. |
| travel_002 | Travel from Seoul to Daejeon on 2026-10-21. Consider options with a base fare of at most 60 and a departure hour from 9 through 13, inclusive. Select the option with the lowest base fare, breaking ties by the shortest travel duration. Use the nonrefundable fare option. |
| travel_003 | Travel from Busan to Daegu on 2026-10-22. Consider options with a base fare of at most 50 and a departure hour from 10 through 14, inclusive. Select the option with the shortest travel duration, breaking ties by the lowest base fare. Use the refundable fare option. |
| travel_004 | Travel from Daejeon to Seoul on 2026-10-23. Consider options with a base fare of at most 70 and a departure hour from 13 through 17, inclusive. Select the option with the shortest travel duration, breaking ties by the lowest base fare. Use the nonrefundable fare option. |
| travel_005 | Travel from Daegu to Seoul on 2026-10-24. Consider options with a base fare of at most 90 and a departure hour from 11 through 15, inclusive. Select the option with the lowest base fare, breaking ties by the shortest travel duration. Use the refundable fare option. |
| place_001 | Find a museum in Seoul. Consider candidates with a base admission fee of at most 30 and a walking time of at most 25 minutes. Select the highest-rated place, breaking ties by the shortest walking time. Choose the indoor visit area. |
| place_002 | Find a cafe in Seoul. Consider candidates with a base admission fee of at most 20 and a walking time of at most 20 minutes. Select the place with the shortest walking time, breaking ties by the highest rating. Choose the outdoor visit area. |
| place_003 | Find a restaurant in Busan. Consider candidates with a base admission fee of at most 40 and a walking time of at most 30 minutes. Select the highest-rated place, breaking ties by the shortest walking time. Choose the indoor visit area. |
| place_004 | Find a park in Seoul. Consider candidates with a base admission fee of at most 10 and a walking time of at most 25 minutes. Select the place with the shortest walking time, breaking ties by the highest rating. Choose the outdoor visit area. |
| place_005 | Find a gallery in Busan. Consider candidates with a base admission fee of at most 25 and a walking time of at most 20 minutes. Select the highest-rated place, breaking ties by the shortest walking time. Choose the indoor visit area. |

## Interpretation and limits

This design controls the number of remaining dependent calls in the same workflow. Starting information also differs, so it does not isolate only the length of a general real-world task.
Error accumulation and first-error position from a cold start are analyzed with h5. Treat the 20 base scenarios as the analysis units and compare horizon conditions as matched observations.
Recovery fixtures exist only for h1 and h3. An h5 normal_restart would start from the same empty state as free h5, so h5 is analyzed with free execution only.
In every scenario the constraint-violating candidate ranks first when the constraints are ignored, so saving it is the plausible mistake of skipping a constraint check.
All four domains currently use the same read-request -> search -> details -> prepare -> confirm workflow. Broader claims about agent behavior require more diverse tasks.
The only selection decisions (candidate and option) occur at h3 and h5 alike; the two extra h5 calls pass identifiers without a choice. Report h3-h5 differences as effects of longer dialogue and identifier passing, not of additional reasoning steps.
Free execution does not inject errors. Compare recovery only between normal_restart and error_restart within the matched fixture. Do not directly subtract recovery success from free-execution success.
The model-facing dataset language is English, recorded as language=en and dataset_version=workflow-en-v3.
