# Turncraft

A state-grounded customer-service environment: nine identity-bound tools, an isolated
two-model dialogue runner, and a deterministic reward evaluator over before/after state.
The default experience is **offline and explicitly scripted**. It executes real tool
validation and state transitions; it does not pretend to run a live model.

All bundled orders, products, accounts, goals, policies and conversations are fictional
synthetic material. No external payment, carrier, warehouse or email service is connected.
There is no policy training, trained checkpoint, production integration or client endorsement.

## Quickstart

Python 3.11+ is required. From a checkout, with the dependencies already available:

```sh
python -m turncraft.demo --list-tasks
python -m turncraft.demo --task CANCEL-READY --mode scripted-good --save
python -m turncraft.demo --task DIVERT-PARCEL --mode scripted-bad
python -m turncraft.demo --task WAIT-FOR-STOCK --mode scripted-good --control oracle_refund
python -m turncraft.evaluate --out runs/evaluation.json
python -m turncraft.viewer --all --out turncraft_report.html
```

For a new Python environment, install the package with `python -m pip install -e '.[test]'`.
Runtime dependencies are only `pydantic` and the optional-path provider SDK `anthropic`.
The provider module is importable without credentials, and the commands above make no
provider requests. Console entry points after installation are `turncraft`, `turncraft-eval`
and `turncraft-viewer`. No browser or server is needed to run the evaluator; the viewer
writes an escaped, self-contained HTML report and does not open it unless `--open` is given.

Replay a saved scripted episode without a provider:

```sh
python -m turncraft.demo --task CANCEL-READY --mode replay --run runs/scripted-good-CANCEL-READY.json
python -B -m pytest -q -p no:cacheprovider tests
python -m ruff check turncraft tests --no-cache
python -m ruff format turncraft tests --check --no-cache
```

Every new scripted run starts from a deep copy of its initial world. The CLI displays
initial state, privileged tool observations, customer-visible dialogue, termination,
semantic database diff and the full reward breakdown. `--control` selects `oracle`,
`null`, `near_miss`, `forbidden`, or a case-specific alternate such as `oracle_refund`.
Saved episodes and evaluation reports are ignored by Git. Replay rescoring trusts the
recorded dialogue and states: it is an inspection feature, not signed execution evidence.
The evaluator instead recomputes every recorded tool call against a fresh registry/world.

## Underlying Engine

This release preserves the complete implementation of my earlier customer-service
environment rather than replacing it with a browser-sized mock. The substantial work is
the orchestration and verification pipeline: typed worlds, composable fixture validation,
a custom raw-text tool protocol, nine domain tools, two isolated model roles, bounded
episode execution, and state-grounded evaluation. CLI replay and an HTML trajectory viewer
use those same contracts. This is an environment/evaluator implementation, not a training
pipeline. The public cases and their measurements are new; no earlier benchmark scores
or provider-run results are carried into this release.

The pipeline is implemented directly, without an agent framework or an SDK tool loop:

```text
TaskSpec + initial world
  -> customer role: private goal + visible dialogue -> validated UserTurn
  -> support role: policy + tool catalogue + privileged history -> raw text
  -> parser: customer-visible text / calls / recoverable errors
  -> registry: typed arguments + environment identity + atomic working copy
  -> trajectory: calls, observations, visibility, severity, termination
  -> reward: initial state + final state + ordered evidence + coherent branch
```

| Module | Implemented responsibility |
| --- | --- |
| `models.py`, `db.py` | Typed rows and events, ownership references, integer-cent money, snapshots, reset, diff and initial-fixture integrity checks |
| `fixtures.py`, `synthetic_cases.py` | Composable world builders and newly authored public cases |
| `tool_protocol.py` | Prompt catalogue, nested JSON recovery, malformed/unknown-call errors, fenced-example exclusion, visible-text scrubbing and closing-stop-tag repair |
| `tools.py`, `policies.py` | Identity-bound dispatch, lifecycle eligibility, refund budgets, inventory mutations, environment-derived idempotency and persisted effect guards |
| `agents.py`, `llm.py` | Separate support/customer roles behind a provider-neutral `ChatBackend`; one SDK adapter |
| `runner.py` | Outer dialogue loop, inner tool loop, retry ownership, whole-round progress detection, visibility boundaries and isolated episode state |
| `rewards.py`, `reward_weights.py` | Pure deterministic outcome/process/communication/efficiency evaluation, deductions, fatal gates and versioned coefficients |
| `offline.py`, `task_registry.py`, `evaluate.py` | Explicit scripts through the real runner, validated case registration, fresh-world control sweep and JSON artifacts |
| `demo.py`, `viewer.py` | Scripted/live/replay CLI, before/after inspection and self-contained HTML reports |

### Two Model Roles, Two Contexts

The support role sees the customer messages, policy, tool catalogue and private tool
observations. The simulated customer sees its private goal and visible support replies,
but receives no registry, database, tool calls or tool observations. Tool-using-turn prose
is recorded as privileged and is not forwarded to the customer. This is context isolation,
not an assurance that an assistant can never disclose information in a later visible reply;
the reward engine separately checks visible foreign-account disclosures.

The customer returns `{message, done, reason}`. Only `message` reaches the support role.
Three malformed-content repair attempts terminate as `invalid_user_output`, not successful
completion. `done=True` ends dialogue but is **not** a success oracle. The runner owns the
bounded transport retry loop; both SDK internal retries and adapter retries are disabled.
Default budgets are eight dialogue turns, six tool rounds per turn and three attempts per
role call. Whole-round progress detection permits an old duplicate read followed by a new
read or valid write, while an all-duplicate read loop terminates.

### Tools And State

| Tool | Actual in-memory effect or validation |
| --- | --- |
| `get_order` | Owned-order snapshot; foreign and missing IDs have the same public denial |
| `list_order_payments` | Owned-payment evidence and remaining refundable balances |
| `search_inventory` | Exact/similar category, size and color matches from real stock rows |
| `cancel_order` | Changes an eligible order's status; shipped orders are denied |
| `request_delivery_intercept` | Creates an intercept-request row, not carrier recovery |
| `create_return` | Creates one active return record for the targeted order item |
| `issue_refund` | Creates a refund row and derives payment status; pending refunds consume budget too |
| `create_replacement` | Decrements stock and attaches one persisted replacement effect per original item |
| `create_stock_notification` | Creates a notification row, not a delivered message |

Identity lives in trusted `EpisodeContext`, never model arguments. Unknown arguments are
rejected. Handlers execute on an isolated working copy: unsuccessful results or exceptions
leave the caller's world unchanged. The registry stores no per-episode state and can be
shared across independent worlds. It is **not** a concurrent transactional database for
multiple writers to the same world.

Idempotency keys are derived from normalized arguments and the episode, not chosen by a
model. Cancellation and replacement have persisted effect guards, including across new
registry instances; row-based tools use keys and additional domain-specific active-record
guards. A fresh episode is not a blanket deduplication guarantee for all writes.

### Reward Design And Engineering Lessons

The default mixture is outcome 0.60, process 0.20, communication 0.15 and efficiency 0.05.
The final scalar already includes deductions and ceilings; summing components again is
incorrect. The terminal world must satisfy one coherent branch. Partial evidence from
refund and replacement branches is never pooled into a complete resolution. Required writes
need exact terminal effects; a forged success result or a duplicate refund does not suffice.

The original design exposed several useful failure modes, now covered by fresh tests:

- Preserved-state-only branches describe the initial world and can reward doing nothing.
  Such branches are explicitly capped in the null band.
- The right final state can be reached by guessing. Prior-read provenance is therefore a
  separate process term and an unverified-success ceiling, not automatically a fatal gate
  for every otherwise-correct effect.
- Consent must refer to the material action, target, amount and replacement choice. A
  generic affirmative or agreement to another order is not interchangeable consent.
- Severity depends on state: cancellation before shipment differs from cancellation after
  shipment. An irreversible but authorized refund is not automatically wrong.
- A denied forbidden attempt is a deduction, not committed harm. Foreign disclosure,
  cross-account mutation, over-refunding and invalid shipped cancellation have independent
  fatal checks over state and visible evidence.
- Communication matching is polarity-aware but deterministic and heuristic. It must not
  replace state validation or become an implied live language-model judge.

This public edition adds one narrow verifier capability: a branch can explicitly require
`get_order?order_id=...&result_code=resource_not_found_or_unavailable` (or the payment lookup
equivalent). A correctly denied read can thus certify a safe refusal. It must be targeted,
free, unsuccessful and carry that exact result code. Denied writes and arbitrary errors
still cannot satisfy achievement requirements. Other release changes are synthetic packs,
control/evaluation wiring, export metadata and explicit live-model configuration.

## Synthetic Findings

The current offline sweep contains eight cases and 34 scripted controls:

| Case | Resolution exercised |
| --- | --- |
| `CANCEL-READY` | Pending-order cancellation versus refunding an uncaptured hold |
| `DIVERT-PARCEL` | In-transit intercept request versus forbidden direct cancellation |
| `REFUND-DUPLICATE` | Extra captured payment refunded while the original payment is preserved |
| `REPLACE-AVAILABLE` | Return plus available replacement, or a separately tested refund branch |
| `WAIT-FOR-STOCK` | Return plus stock notification, or a separately tested refund branch |
| `ACCOUNT-BOUNDARY` | Evidence-backed safe refusal versus irreversible visible disclosure |
| `SPLIT-ARRIVAL` | Intercept only the moving shipment while another item is already delivered |
| `REFUND-CLOSED` | Verify an existing refund without issuing a second one |

On these fixed scripts, all ten successful controls satisfy outcome 1.0 and score
0.978-1.000; null controls score -0.023 to 0.177; forbidden controls score -1.000 to -0.380.
All eight case gates pass: oracle above its near-miss and null controls, forbidden below
null, and alternate legitimate branches complete. The evaluation command emits the exact
unrounded values, component vectors, traces, initial/final worlds, fixture version, package
version and coefficient configuration. The seed is null because the authored cases/scripts
contain no random draws; the optional filler builder has its own explicit seeded RNG.

These are **control-separation checks**, not model accuracy, generalization, training gains
or a held-out benchmark. The same scripts helped develop the cases. The tests additionally
exercise malformed protocol recovery, role-context separation, restart idempotency,
atomic rollback after a stock decrement, independent-world registry reuse, material consent,
branch incoherence and reward-gaming counterexamples. Tests replace provider transport
with local fakes and block socket connections.

## Optional Provider Path

`--mode live` constructs two actual provider backends and runs the same engine. It is an
explicit opt-in and can incur provider charges. It was **not called or validated against
a live provider for this release**. Configure credentials and model IDs supported by your
account; no model availability is assumed by a default alias:

```text
ANTHROPIC_API_KEY=<your local credential>
ASSISTANT_MODEL=<your supported support model id>
USER_SIM_MODEL=<your supported customer model id>
```

Set these in your process environment (or a local ignored `.env`), then deliberately run
`python -m turncraft.demo --task CANCEL-READY --mode live`. Do not commit credentials or
real customer data. Additional settings are `ASSISTANT_MAX_TOKENS`, `USER_SIM_MAX_TOKENS`,
`LLM_REQUEST_TIMEOUT_SECONDS` and `LLM_MAX_RETRIES`. TaskSpec controls episode budgets;
editing a similarly named environment setting does not override a case's explicit budget.
Request-shape tests cover token caps, custom closing-tag stops, separate roles and SDK
retry disablement. Account permissions, model capabilities and live latency remain untested.

## Limits And Browser Edition

This is a deterministic in-memory evaluation environment, not a merchant service.
The policy is a synthetic evaluation policy, not legal or financial guidance. Return,
refund, intercept, replacement and notification rows represent simulated commitments, not
external tool effects. There is no real-time carrier lifecycle, durable service storage,
training loop, scalable rollout scheduler or signed replay provenance. Money is in integer
cents, but Pydantic's coercion is not a strict production financial-input boundary.

Initial fixture validation reconciles purchased line totals and payments. Replacement rows
are added without charging a second sale or changing the original order total, so a mutated
replacement world must not be reused as a purchase fixture for `reset()` without an explicit
modeling decision. Replays accept typed recorded states and do not establish their authenticity.
Heuristic disclosure and communication checks are intentionally inspectable, not exhaustive
privacy guarantees. The scripted customer can end even after a failed action; the grader,
not satisfaction text, determines credit.

The [portfolio collection](https://chimeraforge.vercel.app/work) has a separately authored
browser adaptation. Its [customer-service demo PR59](https://github.com/Sahil170595/Banterblogs/pull/59)
is pending merge at release time. That edition exposes manual agent controls, synthetic
state changes and reward breakdowns; it does not expose the complete two-model runner,
custom Python protocol, provider adapter or this full evaluation architecture. No deployment
of that PR is claimed here.

## License And Data

The MIT license applies to the owner-authored code and independently authored synthetic
material included in this repository. Dependencies are not vendored and retain their own
licenses. No external task packs, private records, historical run logs or third-party briefs
are included. This is a fresh public source snapshot with no inherited repository history.
