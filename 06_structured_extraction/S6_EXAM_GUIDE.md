# S6 — Structured Data Extraction · CCA-F Max-Score Guide

> Scope: Scenario **S6**, Domains **D4** (Prompt Engineering) + **D5** (Context Management).
> Tasks **4.2, 4.3, 4.4, 4.5, 5.5** in depth, plus the adjacent concepts that show up as distractors.
> Every code snippet is traceable to `schema.py` / `main.py` / `batch.py` / `prompts/extraction_prompt.txt` in this lab.

---

## 0. How this section is actually tested

Exam items in S6 are **scenario-framed multiple choice**: a paragraph describing a production extraction pipeline that is failing in a specific way, then four plausible fixes. Three of them are real techniques applied to the *wrong failure mode*. You max the score by classifying the **failure mode** first, then picking the technique that owns it.

| The failure you're shown | The one technique that owns it | Task |
|---|---|---|
| Output isn't valid JSON / drifts from the shape | `tool_use` + JSON schema + forced `tool_choice` | 4.3 |
| A field is **fabricated** because the doc doesn't have it | **Nullable** field (`["string","null"]`, not in `required`) | 4.3 |
| A field is **guessed** because the doc is ambiguous | **Enum with `"unclear"`** | 4.3 |
| A value **doesn't fit** the fixed taxonomy | **Enum with `"other"` + `detail` string** | 4.3 |
| Output shape is valid but **inconsistent run-to-run / across formats** | **Few-shot examples** (2–4, ambiguous cases) | 4.2 |
| Numbers don't add up / stated ≠ computed | **Self-correction field** (`calculated_total`, `conflict_detected`) + **deterministic validation in code** | 4.4 |
| Fixable errors ship to production unfixed | **Retry with error feedback** (`tool_result` + `is_error: True`) | 4.4 |
| Retry loop burns tokens and never converges | Classify **retryable vs non-retryable**; skip non-retryable; bound with `MAX_RETRIES` | 4.4 |
| Cost too high / volume too large / nobody is waiting | **Message Batches API** (50% off, 24h window) | 4.5 |
| Results come back mismatched to inputs | **`custom_id` correlation** (results return out of order) | 4.5 |
| Humans review everything, reviewer capacity wasted | **Confidence routing** (auto_approve / spot_check / human_review) | 5.5 |
| "97% accurate" but one doc type keeps failing | **Stratified sampling** — accuracy per document type AND per field | 5.5 |
| Thresholds were guessed | **Calibrate on a labeled validation set**, weighted by cost of error | 5.5 |

**Read the last sentence of the stem first.** It names the failure. The rest is set dressing.

---

## 1. The one mental model

> **A schema is a contract the model cannot break. So every legitimate answer must be expressible inside it — including "missing", "ambiguous", and "doesn't fit". If it isn't expressible, the model fabricates, because breaking the contract is not an option.**

Everything in S6 is a consequence of that sentence:

```
Unstructured doc
      │
      ▼
[ SCHEMA ]  ── forces shape ── guarantees syntax, NOT truth
      │        escape hatches: null · "unclear" · "other"+detail
      ▼
[ FEW-SHOT ] ── makes the escape hatches actually get used, consistently
      │
      ▼
[ SELF-CORRECTION ] ── model recomputes derived values, flags conflicts
      │
      ▼
[ VALIDATION ] ── deterministic Python, not the model
      │
      ├── retryable  → retry WITH the specific error fed back  ─┐
      └── non-retryable (info absent) → skip, do not retry      │
      │                                                        │
      ▼  (bounded by MAX_RETRIES) ◄────────────────────────────┘
[ CONFIDENCE ROUTING ] ── auto_approve / spot_check / human_review
      │
      ▼
[ CALIBRATION ] ── labeled set, stratified by doc type AND field
```

Layers are **complementary, not alternatives**. Any answer option that says "instead of X, do Y" where X and Y sit at different layers is almost always the wrong option.

---

## 2. Task 4.3 — Enforce structured output using tool use and JSON schemas

### 2.1 Exam concept

`tool_use` with a JSON schema is **the most reliable method for schema-compliant structured output**. You define a tool that has **no executable function** — it exists only as a shape. The model "calls" it, and the extracted data arrives as `block.input`, already parsed into a dict. The API enforces the schema; you never parse text.

### Best design approach

Define the tool as a companion `_schema` dict, pass it in `tools=[...]`, and **force** it with `tool_choice`:

```python
# schema.py
extract_invoice_schema = {
    "name": "extract_invoice",
    "description": (
        "Extract structured data from an invoice document. "
        "For fields where the document provides no information, you MUST "
        "pass null rather than fabricating a value."
    ),
    "input_schema": {"type": "object", "properties": {...}, "required": [...]},
}

# main.py
response = client.messages.create(
    model=MODEL,
    max_tokens=4096,
    system=system_prompt,
    tools=[extract_invoice_schema],
    tool_choice={"type": "tool", "name": "extract_invoice"},   # ← forced
    messages=messages,
)

for block in response.content:
    if block.type == "tool_use":
        return block.input      # already a dict — no json.loads, no regex
```

### The three `tool_choice` modes — memorize this table

| Mode | Syntax | Guarantees | Use when |
|---|---|---|---|
| **Forced** | `{"type": "tool", "name": "extract_invoice"}` | **This specific tool** is called, every time | One known schema. Extraction. **This lab.** |
| **any** | `{"type": "any"}` | *Some* tool is called — model picks which | Document type is unknown and you registered several schemas (invoice vs receipt vs PO) |
| **auto** | `{"type": "auto"}` (default) | Nothing — model may return plain text | Conversational agents where a text reply is a valid outcome |
| **none** | `{"type": "none"}` | No tool is called | Force a text-only turn even though tools are registered |

> Trap: an option that says "use `tool_choice: auto` and instruct the model in the prompt to always call the tool" is **wrong for extraction** — `auto` permits a text response, which is exactly the failure you were asked to eliminate.
> Trap: forced `tool_choice` **disables extended thinking** on the same request, and the model can't emit a preamble. If a question wants reasoning *and* strict shape, the answer is two passes (reason → then extract), not one forced call.

### 2.2 The three escape hatches

These are the highest-yield facts in the whole section.

**(a) Nullable — for data that is ABSENT**

```python
"vendor_phone": {
    "type": ["string", "null"],          # union type
    "description": "Vendor phone number. Null if not present in the document.",
},
# and: "vendor_phone" is NOT in the "required" list
```

Returns real JSON `null` → Python `None` → SQL `NULL`. `if x is None` works.
Without it the model returns the **four-character string `"null"`**, and downstream code treats it as a real phone number.

**(b) Enum with `"unclear"` — for data that is PRESENT but AMBIGUOUS**

```python
"payment_terms": {
    "type": "string",
    "enum": ["net_15", "net_30", "net_45", "net_60", "due_on_receipt", "unclear"],
    "description": "Payment terms. Use 'unclear' when terms are ambiguous "
                   "or reference external agreements.",
},
```
Invoice 04 says *"Per existing retainer agreement — see section 4.2 … or Net 45 if retainer balance is insufficient."* A plain string forces a commit to `"net_45"`, and AP schedules the wrong payment. `"unclear"` lets the model decline **inside** the contract.

**(c) Enum + `"other"` + `detail` — for data that DOESN'T FIT the taxonomy**

```python
"category": {
    "type": "object",
    "properties": {
        "value": {"type": "string",
                  "enum": ["consulting", "office_supplies", "technology",
                           "maintenance", "travel", "utilities", "other"]},
        "detail": {"type": ["string", "null"],
                   "description": "Explanation when value is 'other'. Null otherwise."},
    },
    "required": ["value"],
},
```
Invoice 05 is catering. Without `"other"`, it gets filed as `"consulting"` and enters the wrong approval workflow. `{"value": "other", "detail": "catering and event services"}` keeps the taxonomy closed **and** captures the long tail — that's the extensibility argument.

| Symptom | Hatch |
|---|---|
| Info is **not in the document** | nullable |
| Info **is** in the document but can't be resolved | enum + `"unclear"` |
| Info is clear but **no listed option fits** | enum + `"other"` + `detail` |

### 2.3 `required` semantics

- A field in `required` **must** appear — if it's also nullable it can still be `null`, but the key must be present.
- A field **not** in `required` may be omitted entirely — so always `.get()`, never `[...]`, in validation code.
- In this lab: `invoice_number, vendor_name, vendor_address, customer_name, invoice_date, payment_terms, currency, line_items, stated_total, category` are required; `vendor_phone, due_date, purchase_order, subtotal, tax_rate, tax_amount` are nullable-and-optional. Steps 5 and 9 add `calculated_total`, `conflict_detected`, `confidence` to `required` — **a self-correction or confidence field that isn't required is a field the model can silently skip.**

### 2.4 Syntax vs semantics — the sentence they love to test

> **A strict schema eliminates JSON *syntax* errors. It does not eliminate *semantic* errors.**

Guaranteed by the schema: valid JSON, correct types, keys present, enum membership.
**Not** guaranteed: the value is *true*, totals add up, dates were read correctly, the vendor name isn't a customer name.

That gap is exactly why 4.4 (validation/retry) and 5.5 (confidence/human review) exist. If a question says "we added a strict schema and still get bad data," the answer is never "make the schema stricter."

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| "Return JSON" in the prompt + `json.loads()` | No enforcement. Prose preamble, markdown fences, trailing commas, dropped keys. You're writing a parser and a repair loop instead of using the one the API gives you. |
| Assistant **prefill** with `{` | Nudges shape, enforces nothing — no key/type/enum guarantees, and it fights `tool_choice`. Fine for tone/format steering, not for contracts. |
| Regex / string parsing of a text response | Brittle to any format change, silently wrong, unmaintainable across 7 invoice formats let alone hundreds. |
| Post-hoc `jsonschema` validation of free text | Detects failure *after* paying for a bad generation, and gives you nothing to repair with. Use `jsonschema` **in addition to** tool_use, never instead of it. |
| Fine-tuning a model to emit the shape | Wrong altitude, wrong cost, wrong iteration speed — and the exam's stack is prompt/schema engineering, not training. |
| Make every field `required: string` for "consistency" | Directly causes fabrication. This is the anti-pattern the whole lab is built to kill. |

---

## 3. Task 4.2 — Few-shot prompting for consistency and quality

### 3.1 Exam concept

Instructions alone produce **inconsistent** results at scale: null handling varies run-to-run, edge cases resolve differently each time. Few-shot examples are the most effective lever for **consistency**, because they demonstrate ambiguous-case handling *concretely* instead of describing it. **Target 2–4 examples covering the most common ambiguous scenarios.**

### Best design approach

Put examples in the **system prompt** via a template variable, wrap each in **XML tags**, and choose examples by **the decision each one teaches** — not by how typical they are.

```
# prompts/extraction_prompt.txt  (loaded, then .format()-ed — never an f-string)
## Examples

<examples>
{few_shot_examples}
</examples>
```

```python
# main.py
def format_few_shot_examples(examples):
    formatted = []
    for ex in examples:
        extraction_json = json.dumps(ex["extraction"], indent=2)
        block = (
            f"<example>\n"
            f"<invoice>\n{ex['document']}\n</invoice>\n"
            f"<correct_extraction>\n{extraction_json}\n</correct_extraction>\n"
            f"</example>"
        )
        formatted.append(block)
    result = "\n\n".join(formatted)
    return result

system_prompt = system_template.format(few_shot_examples=few_shot_text)
```

### The three examples in `data.py` and what each one buys

| # | Example | Teaches |
|---|---|---|
| 1 | Acme Corp — clean, all fields present | Baseline shape and the normalized formats (`2025-01-10`, `7020.00`, `net_30`) |
| 2 | River Stone Landscaping — sparse | **`null` is the correct answer**, plus `"unclear"` payment terms. This is the anchor example. |
| 3 | Dupont & Fils — French, `€9.500,00`, `28/02/2025`, total mismatch | Format normalization **and** self-correction: `conflict_detected: true` + a `medium` confidence flag |

> **The single most quotable line:** *without an example showing `"vendor_phone": null`, the model's prior is to fill every field.* One null-handling example shifts that prior significantly.

### Generalization

The model doesn't memorize French. It sees comma-decimals + DD/MM/YYYY handled once and applies the same logic to invoice_06's German `€13.145,00` / `15/03/2025`. **Examples teach the decision procedure, not the locale** — so pick examples that span *decision types*, not *countries*.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| More explicit prose instructions | Describes the rule; doesn't demonstrate the judgment call. Consistency plateaus. Instructions and examples are complementary — the prompt keeps the normalization rules *and* the examples. |
| 20–50 examples | Diminishing returns after ~2–4, inflates every request's input tokens, raises latency and cost, and can over-anchor the model on example-specific quirks. |
| Only clean/typical examples | Teaches nothing. Examples earn their tokens by covering the cases where the model would otherwise guess. |
| Examples in the `user` turn per request | Works, but bloats each message and is harder to cache. System prompt + one cache breakpoint is the efficient pattern. |
| Fine-tune on 10k invoices | Massively slower iteration for a problem few-shot solves in three examples. Only argue for training when the exam explicitly gives you a fixed high-volume, latency- and cost-critical, stable-schema scenario. |
| Chain-of-thought instead | Helps reasoning, not shape consistency — and forced `tool_choice` disables extended thinking anyway. |

---

## 4. Task 4.4 — Validation, retry, and feedback loops

### 4.1 Exam concept — three separate mechanisms

1. **Self-correction fields** — make the model independently derive a value and compare it to the stated one.
2. **Deterministic validation** — plain Python that checks the extraction. Not a second LLM call.
3. **Retry with error feedback** — re-prompt with the original doc + the failed extraction + **the specific errors**, only for errors that are actually fixable.

### 4.2 Self-correction (schema-side)

```python
"calculated_total": {
    "type": "number",
    "description": "Sum of line item amounts plus tax. Computed by you "
                   "independently of the stated total.",
},
"conflict_detected": {
    "type": "boolean",
    "description": "True when calculated_total differs from stated_total.",
},
# both added to "required"
```

Invoice 03: stated `$5,037.58`, line items sum to `$4,676.82`, +7.5% tax = `$5,027.58`. The $10 gap surfaces as `conflict_detected: true` instead of a silent bad payment. **The boolean is the point** — it's the machine-readable flag downstream routing consumes. A free-text "note any discrepancies" field can't be branched on.

### 4.3 Deterministic validation (code-side)

```python
def validate_extraction(extraction):
    errors = []

    calculated = extraction.get("calculated_total")
    stated = extraction.get("stated_total")
    if calculated is not None and stated is not None:
        diff = abs(calculated - stated)
        if diff > 0.01:                      # float tolerance, not ==
            errors.append(
                f"Total mismatch: stated_total={stated}, "
                f"calculated_total={calculated}, difference={diff:.2f}"
            )

    for field in ["invoice_number", "vendor_name", "invoice_date"]:
        if extraction.get(field) is None:
            errors.append(f"Required field '{field}' is null — info absent from document")

    date_val = extraction.get("invoice_date", "")
    if date_val and not _is_valid_date(date_val):
        errors.append(f"Invalid date format: {date_val} (expected YYYY-MM-DD)")

    if not extraction.get("line_items", []):
        errors.append("No line items extracted")

    return errors
```

Three things the exam checks here: **tolerance-based float comparison** (`> 0.01`, never `!=`), **error strings carry the actual values** (they become the retry feedback), and **validation is deterministic code** — an LLM validator is a second thing that can hallucinate, and it costs a round trip.

### 4.4 Retry with error feedback — the exact message shape

You reconstruct a conversation in which the model already called the tool, and you return a **failed tool result**:

```python
messages = [
    {"role": "user", "content": f"Extract all fields from this invoice:\n\n<invoice>\n{invoice_text}\n</invoice>"},
    {"role": "assistant", "content": [
        {"type": "tool_use", "id": "retry_call", "name": "extract_invoice", "input": extraction}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result",
         "tool_use_id": "retry_call",                 # MUST match the tool_use id
         "content": (f"Validation failed. Fix these errors and re-extract:\n{error_list}\n\n"
                     f"Previous extraction:\n{failed_json}"),
         "is_error": True}                            # marks it as a failure
    ]},
]
```

Three non-negotiables: `tool_use_id` must match the `id` of the `tool_use` block; `is_error: True` signals failure rather than data; and the retry carries **the original document, the failed extraction, and the specific errors** — all three. A bare "try again" retry is the classic wrong answer: it re-rolls the dice instead of correcting.

### 4.5 Retryable vs non-retryable — the highest-value distinction in 4.4

```python
retryable     = [e for e in errors if "absent" not in e.lower()]
non_retryable = [e for e in errors if "absent" in e.lower()]

if retryable:
    for attempt in range(1, MAX_RETRIES + 1):        # bounded
        extraction = retry_with_feedback(client, invoice_text, extraction, retryable, system_prompt)
        errors = validate_extraction(extraction)
        retryable = [e for e in errors if "absent" not in e.lower()]
        if not retryable:
            break
```

| Error | Class | Why |
|---|---|---|
| Total mismatch (invoice_03) | **Retryable** | Model may have miscounted line items or misread the total |
| Date not `YYYY-MM-DD` | **Retryable** | Pure format failure, re-reading fixes it |
| Missing line items | **Retryable** | Likely a parsing miss |
| `invoice_number` is null (invoice_07, Mike's Plumbing) | **NOT retryable** | The informal invoice **has no invoice number**. No amount of re-reading conjures one. |

Retrying a non-retryable error burns tokens, adds latency, and can push the model to **fabricate** just to satisfy you — actively harmful. Route it to human review instead.

> Note on invoice_03: after the retry the model may return the *same* totals, because the document genuinely disagrees with itself (there's a volume-discount note). **That is a valid outcome.** The mechanism engaged; the conflict is real. Retry is not a guarantee of resolution.

### 4.6 Dismissal pattern analysis (know the term)

Track a `detected_pattern` field naming the construct that triggered the failure (e.g. `"European comma-decimal format"`). Over time you learn which flags are *always* dismissed by reviewers — e.g. "missing PO" on informal invoices is never a real issue — and you **suppress that check for that document type**. `confidence.flags` serves the same role. This is how you shrink human review load *without* lowering the confidence bar.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Retry with the same prompt, no feedback | Resampling, not correcting. The model has no idea what was wrong. |
| Unbounded retry until valid | Infinite loop + runaway cost on non-retryable errors. Always cap (`MAX_RETRIES = 2`). |
| Retry every error indiscriminately | Wastes calls on absent data and pressures the model to fabricate. |
| A second LLM call to validate | Non-deterministic, costs a round trip, can hallucinate approval. Deterministic checks belong in code; save model calls for judgment that code can't do. |
| Lower the temperature / add "be accurate" | Doesn't detect anything. No feedback loop, no flag, no route. |
| Tighten the schema further | Schema fixes syntax; these are semantic errors. |
| Have the model fix it in a follow-up *text* turn | Loses the schema contract on the corrected output. Retry must go back through the same forced tool call. |

---

## 5. Task 4.5 — Efficient batch processing strategies

### 5.1 Exam concept — the three numbers

**50% cost reduction · 24-hour processing window · no latency SLA.**
(Most batches finish far sooner; you may not *rely* on that. Results are retrievable for ~29 days.)

### Best design approach

Same params as the sync call, wrapped per-request with a **`custom_id`**:

```python
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

params = MessageCreateParamsNonStreaming(
    model=MODEL,
    max_tokens=4096,
    system=system_prompt,                                  # same few-shot system prompt
    tools=[extract_invoice_schema],
    tool_choice={"type": "tool", "name": "extract_invoice"},   # still forced
    messages=[{"role": "user", "content": user_content}],
)
request = Request(custom_id=filename.replace(".txt", ""), params=params)

batch = client.messages.batches.create(requests=requests)
batch.id                        # save it
batch.processing_status         # in_progress → ended
batch.request_counts            # processing / succeeded / errored / expired / canceled
```

Retrieval — note the per-result **result type switch**, which is where batch differs most from sync:

```python
for result in client.messages.batches.results(batch_id):
    cid = result.custom_id                       # ← the ONLY reliable correlation key
    if result.result.type == "succeeded":
        for block in result.result.message.content:
            if block.type == "tool_use":
                extraction = block.input
    elif result.result.type == "errored":
        ...   # resubmit by custom_id
    elif result.result.type == "expired":
        ...   # hit the 24h window
    elif result.result.type == "canceled":
        ...
```

### The batch decision table

| Use batch | Do **not** use batch |
|---|---|
| Overnight bulk extraction of the day's invoices | Pre-merge code review (blocking a developer) |
| Monthly reprocessing of a backlog | Real-time customer support |
| **Prompt refinement on a sample set** — batch 50 diverse invoices, review, refine the prompt, then batch the full volume | Anything where a **user is waiting** for the result |
| Any offline, non-blocking, cost-sensitive volume | Agentic loops needing tool call → result → next call |

### SLA arithmetic (they *will* make you do this)

> AP SLA = invoices processed within **30 hours** of receipt. Batch window = **24 hours**. Buffer = **6 hours** → batch every 4 hours and you stay inside the SLA.
> Change the SLA to **20 hours** and batch **cannot** meet it — the 24h window alone exceeds the budget. Use the sync API.

Method: `SLA − 24h = buffer`. Buffer ≤ 0 → sync. Buffer > 0 → submit at an interval ≤ buffer.

### Four batch facts that show up as distractors

1. **Results return in a different order than submitted.** `custom_id` is the only reliable way to match a result to its source document. Not index, not order, not filename-in-content.
2. **Each request is a single turn — no multi-turn tool calling inside a batch request.** One forced tool call is fine (that's this lab). An agentic loop is not; that needs the sync API.
3. **Requests are independent** — no shared state, no ordering guarantees, no cross-request context.
4. **Failures are resubmitted by `custom_id`**, optionally with modifications — e.g. chunk an oversized document and resubmit as `invoice_42_part1`, `invoice_42_part2`.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Parallel sync calls with a thread pool | No 50% discount, hits rate limits, you hand-build retry/backoff — for work nobody is waiting on. |
| Batch for an interactive user flow | No latency SLA. A user cannot wait up to 24h. |
| Batch for an agentic multi-turn loop | Single-turn per request; the loop can't run. |
| Correlate results by submission order | Order is not preserved. This is the trap answer. |
| Streaming | Reduces *time-to-first-token* for one interactive request. Orthogonal to bulk cost. Never the answer to "reduce cost on 10,000 documents." |
| Resubmit the whole batch on partial failure | Pay twice for everything that already succeeded. Resubmit only failed `custom_id`s. |

---

## 6. Task 5.5 — Human review workflows and confidence calibration

### 6.1 Exam concept — two halves, both tested

**(a) Confidence routing.** The model **self-reports** confidence in the schema; the pipeline routes on it. Purpose: **prioritize limited reviewer capacity** — humans spend time on extractions that need attention, not on clean invoices the model got right.

**(b) Accuracy validation.** Thresholds must be **calibrated against a labeled validation set**, not guessed. Aggregate accuracy **masks** per-type and per-field weakness, so you measure with **stratified sampling**.

### Best design approach — schema

```python
"confidence": {
    "type": "object",
    "properties": {
        "overall": {"type": "string", "enum": ["high", "medium", "low"]},
        "flags": {"type": "array", "items": {"type": "string"},
                  "description": "List of fields or issues with reduced confidence."},
    },
    "required": ["overall", "flags"],
},
# "confidence" added to the top-level required list
```

Backed by explicit rubric text in the system prompt — **a confidence field without a rubric is noise**:

```
- "high":   all key fields are clearly stated and unambiguous
- "medium": some fields required interpretation, format conversion, or minor inference
- "low":    significant information is missing, ambiguous, or contradictory
```

### Best design approach — routing

```python
def classify_review_need(extraction):
    confidence = extraction.get("confidence", {})
    overall = confidence.get("overall", "low")     # fail-safe default: low
    flags = confidence.get("flags", [])

    if overall == "high" and not flags:
        return "auto_approve"
    elif overall == "low" or len(flags) >= 3:
        return "human_review"
    else:
        return "spot_check"
```

Note the design: **three tiers, not two** (the middle tier is sampled, not fully reviewed — that's the capacity win), **flags override confidence** (`high` + any flag ≠ auto-approve), and the **default is the safe direction** (missing confidence → `low` → human review).

Expected routing across the lab's corpus: `invoice_01` clean → `auto_approve`; `invoice_05` catering/`other` category → `spot_check`; `invoice_06` German/EUR format conversion → `spot_check`; `invoice_07` informal, no invoice number → `human_review`.

### 6.2 Calibration — thresholds come from data and from cost

```python
LABELED_VALIDATIONS = {           # data.py — ground truth per document TYPE
    "invoice_01.txt": {...},      # clean corporate
    "invoice_02.txt": {...},      # technology, missing fields
    "invoice_06.txt": {...},      # European format
    "invoice_07.txt": {...},      # informal handwritten
}
```

Those four are chosen deliberately — one per document type in the corpus.

> **The masking argument, stated the way the exam states it:** 92% overall sounds fine. Broken out: 100% corporate, 95% technology, 90% European, **75% informal**. Automate on the aggregate and **every fourth informal invoice ships bad data to payment.** Stratified random sampling — accuracy **per document type AND per field** — is what exposes it.

Two rules that follow:
- **Only automate the (document type × field) cells where accuracy is validated.** Auto-approve corporate invoices; keep informal ones in human review. Partial automation by stratum beats an all-or-nothing threshold.
- **Thresholds depend on the cost of an error.** A $50 office-supplies invoice and a $50,000 consulting engagement do not deserve the same bar. Higher error cost → higher threshold → more human review. Expect an item where the right answer is "raise the threshold for high-value invoices," not "raise it globally."

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Route on token log-probabilities | Not exposed by the API. Self-reported confidence in the schema is the mechanism. |
| A 0–100 numeric confidence score | False precision — models are poorly calibrated on fine-grained self-scores, and "78 vs 82" isn't actionable. A 3-level enum plus a flags array is both calibratable and routable. |
| Two tiers (auto / human) | Throws away the capacity win. The middle tier exists to be *sampled*. |
| Human-review everything | Safe and useless — it's the cost problem you were asked to solve. |
| Auto-approve everything above a guessed threshold | Uncalibrated. "High confidence on everything" is a real model failure mode; you can't know without labels. |
| Report a single aggregate accuracy number | Masks per-type failure. The whole point of 5.5(b). |
| Add more few-shot examples to fix low accuracy on one doc type | Might help — but you can't know *which* type is failing until you measure stratified. Measure first. |
| Retry low-confidence extractions automatically | Low confidence usually means the *source* is ambiguous or incomplete → non-retryable. That's a human-review case, not a retry case. |

---

## 7. Adjacent concepts that appear as distractors

These aren't S6 task statements, but they show up in S6 answer options. Know when each is right and — more often — when it's the plausible-but-wrong pick.

### Prompt caching
The system prompt here is large and **identical across every invoice** (rules + 3 full few-shot examples). Cache it: mark the last system block with `"cache_control": {"type": "ephemeral"}`. Cache **reads** cost roughly 0.1× base input; **writes** ~1.25× (5-minute TTL) or ~2× (1-hour TTL). There's a minimum cacheable prefix (~1024 tokens; higher for Haiku-class models) and a small number of breakpoints (4). Order matters: **static content first** (tools → system → examples), variable content (the invoice) last, or you invalidate the prefix every request.
**Right answer when:** "reduce cost/latency on a high-volume sync pipeline with a large fixed prompt."
**Wrong answer when:** the ask is bulk offline throughput (→ batch) or output shape (→ schema). Caching and batching **combine**; they aren't rivals.

### PDF and image input
Real invoices are PDFs and scans. Send them as `document` blocks (base64 / URL / Files API) or `image` blocks — the model reads page text *and* page layout. There are per-request page and size limits, and page images are token-expensive, so budget accordingly.
**Right answer when:** "our invoices are scanned PDFs, how do we extract?" → same forced-tool-call pipeline, different input block. **The schema, validation, retry, and routing layers do not change.**
**Wrong answer when:** offered as a fix for fabrication or inconsistency — OCR quality is an input problem, not a schema problem. (Note: `extract_images.py` in this lab decodes the RVL-CDIP invoice image corpus — that's the on-ramp to this exact scenario.)

### Citations
For provenance ("which part of the document did this value come from?"), citations tie output spans back to source document blocks. Relevant to audit trails in AP.
**Wrong answer when:** the question is about shape enforcement or missing fields.

### Chunking / long documents
A 200-page invoice packet exceeds sane per-request limits. Chunk by document boundary, extract per chunk, correlate by `custom_id` (`doc_42_part1`, …), then merge. The lab explicitly mentions chunking oversized documents before **resubmission** after a batch failure.

### Model selection
Cheaper/faster models for high-volume mechanical extraction; larger models for ambiguous, high-value documents. A defensible answer is **tiered**: small model first, escalate low-confidence extractions to a larger model *before* escalating to a human. Cost per document is the deciding axis — but only after accuracy is validated **per stratum**.

### Task 4.1 — explicit criteria to reduce false positives (from Lab 05)
Vague criteria → noisy output. In extraction this is the `description` on every schema field, plus the normalization rules in the system prompt (`"March 15, 2025"` → `2025-03-15`; `"€9.500,00"` → `9500.00`). **Field descriptions are prompt engineering** — they're read by the model, not just by you.

### Task 4.6 — multi-pass / multi-instance review (from Lab 05)
Independent passes over the same input, or per-item then cross-item passes. In extraction the analogue is: pass 1 extracts, pass 2 verifies against the source. Legitimate for high-value documents — but in S6 the cheaper, expected answer is **self-correction field + deterministic validation**, not a second model pass. Reach for multi-pass only when the stem stresses very high error cost.

### Task 5.2 — escalation and ambiguity resolution (from Lab 01)
Same shape as confidence routing: define the ambiguity, expose it in the contract (`"unclear"`, `flags`), route it to a human rather than guessing. If an S6 item is phrased as "escalation," it's still 5.5 machinery.

---

## 8. One-page decision tables

### Which layer fixes this?

| Symptom in the stem | Layer | Concrete fix |
|---|---|---|
| "returns prose sometimes" | Schema | forced `tool_choice` |
| "invented a phone number" | Schema | nullable `["string","null"]`, drop from `required` |
| "guessed the payment terms" | Schema | enum + `"unclear"` |
| "filed catering as consulting" | Schema | enum + `"other"` + `detail` |
| "returns the string `"null"`" | Schema | nullable union type (the field is a plain `string` today) |
| "inconsistent across runs / new formats" | Prompt | 2–4 few-shot examples of ambiguous cases |
| "European invoices come out wrong" | Prompt | normalization rules **+** a European example |
| "totals silently wrong" | Schema + code | `calculated_total` + `conflict_detected` + tolerance check |
| "bad data reaches AP" | Code | deterministic `validate_extraction` |
| "fixable errors ship unfixed" | Code | retry with `tool_result` + `is_error: True` |
| "retry loop never ends" | Code | retryable/non-retryable split + `MAX_RETRIES` |
| "10,000 docs, too expensive" | API | Message Batches (50% / 24h) |
| "results mismatched to inputs" | API | `custom_id` |
| "reviewers overloaded" | Schema + code | `confidence` + 3-tier routing |
| "97% but one type keeps failing" | Process | stratified accuracy per type × field |
| "how do we pick the threshold?" | Process | labeled validation set + cost of error |
| "same big prompt every request, cost too high, sync" | API | prompt caching |

### Sync vs Batch

| | Sync | Batch |
|---|---|---|
| Cost | 1× | **0.5×** |
| Latency | seconds | **up to 24h, no SLA** |
| Multi-turn tool loops | yes | **no — single turn** |
| Result ordering | N/A | **arbitrary → `custom_id`** |
| User waiting? | yes | **never** |
| Partial failure | per-call | per-`custom_id`, resubmit selectively |

---

## 9. How to pick when all four options look right

1. **Name the failure mode in one word** — fabrication / ambiguity / taxonomy / consistency / arithmetic / cost / correlation / capacity / calibration. Then use the table in §0.
2. **Match the layer.** Schema fixes shape; prompt fixes judgment; code fixes verification; API fixes economics; process fixes trust. An option operating at the wrong layer is wrong no matter how good it sounds.
3. **Prefer the mechanism that makes the failure *expressible* over the one that makes it *less likely*.** "Add `"unclear"` to the enum" beats "instruct the model to be careful."
4. **Prefer deterministic over model-based** for anything checkable in code.
5. **Prefer measure-then-act.** When the stem contains an aggregate metric, the answer is usually "break it down by stratum," not "apply fix X."
6. **Cheapest sufficient mechanism wins.** Nullable field > retry loop > second model pass > human review > fine-tuning. Pick the leftmost that actually solves it.
7. **Watch for "instead of."** Layers are complementary. An option that removes a working layer to add another is usually the trap.
8. **Absent data is never fixed by re-prompting.** Any option that retries, re-reads, or "prompts harder" for information that isn't in the document is wrong.

---

## 10. Practice exam — 18 scenario items

*Answers and distractor analysis in §11. Cover it and take these cold.*

**Q1.** Your extraction pipeline returns valid JSON 100% of the time, but 8% of extractions contain a `vendor_phone` for invoices that have no phone number on them. What fixes this?
A. Add "do not guess" to the system prompt
B. Change `vendor_phone` to `{"type": ["string","null"]}` and remove it from `required`
C. Add a post-extraction regex check that the phone matches a valid format
D. Lower temperature to 0

**Q2.** Which `tool_choice` setting fits a pipeline that receives invoices, receipts, and purchase orders, with a separate schema registered for each and the document type unknown at request time?
A. `{"type": "tool", "name": "extract_invoice"}`
B. `{"type": "any"}`
C. `{"type": "auto"}`
D. `{"type": "none"}`

**Q3.** An invoice reads: *"Terms per master services agreement, section 7."* Your `payment_terms` field is a plain string and the model returns `"net_30"`. What's the right schema change?
A. Make `payment_terms` nullable
B. Make `payment_terms` an enum including `"unclear"`
C. Make `payment_terms` an object with `value` and `detail`
D. Move `payment_terms` out of `required`

**Q4.** Your team added a strict JSON schema with forced `tool_choice`. Extractions are always schema-valid, yet AP still reports incorrect totals reaching payment. What's the correct read?
A. The schema isn't strict enough; add `additionalProperties: false`
B. Schemas guarantee syntax, not semantics; add a self-correction field plus deterministic validation
C. Switch to a larger model
D. The API is dropping fields; add retries

**Q5.** Which set of few-shot examples best improves consistency for an invoice extractor?
A. 12 clean invoices from your 12 largest vendors
B. 3 examples: one clean, one sparse with `null`s and `"unclear"`, one European-format with a total mismatch
C. 1 example showing the full JSON shape
D. 25 examples spanning every country you operate in

**Q6.** `validate_extraction` reports `"Required field 'invoice_number' is null — info absent from document"` for an informal handwritten invoice. What should the pipeline do?
A. Retry with the error fed back, up to `MAX_RETRIES`
B. Retry with a larger model
C. Skip the retry and route to human review
D. Re-extract with `tool_choice: any`

**Q7.** Which retry message correctly feeds validation errors back to the model?
A. A new `user` message: "That was wrong, try again."
B. `user` (original doc) → `assistant` (`tool_use` with the failed input) → `user` (`tool_result`, matching `tool_use_id`, `is_error: True`, listing the specific errors)
C. Append the errors to the system prompt and resend the original request
D. A `user` message containing only the failed JSON

**Q8.** You must extract 40,000 archived invoices for a year-end audit. Nobody is waiting on individual results. Cost is the binding constraint. What do you use?
A. Sync API with 20 parallel workers
B. Message Batches API
C. Sync API with streaming enabled
D. Sync API with prompt caching only

**Q9.** Batch results come back and your database has vendor names attached to the wrong invoices. Most likely cause?
A. The batch expired
B. You correlated results by submission order instead of `custom_id`
C. `tool_choice` was set to `auto`
D. `max_tokens` was too low

**Q10.** Your AP SLA requires invoices processed within 20 hours of receipt. Batch offers a 24-hour window at 50% cost. What's the correct decision?
A. Batch every 4 hours — the buffer covers it
B. Batch hourly to reduce the risk
C. Batch cannot meet a 20-hour SLA; use the sync API
D. Batch and accept occasional SLA misses since most batches finish in under an hour

**Q11.** Your accuracy check reports 92% overall across 400 invoices, and leadership wants to automate everything. What do you say?
A. 92% is above the 90% bar; automate
B. Break accuracy down per document type and per field before automating anything — the aggregate can hide a failing stratum
C. Add more few-shot examples until it reaches 98%, then automate
D. Automate and rely on confidence routing to catch failures

**Q12.** An extraction returns `confidence.overall = "high"` with `flags: ["tax_rate inferred from total"]`. How should `classify_review_need` route it?
A. `auto_approve` — overall confidence is high
B. `spot_check` — high confidence but a flag is present
C. `human_review` — any flag means human review
D. Retry to resolve the flag

**Q13.** Which is the best justification for a three-tier routing scheme instead of auto-approve/human-review only?
A. Three tiers are easier to log
B. The middle tier is sampled rather than fully reviewed, so reviewer capacity goes to the extractions that need it
C. The model produces three confidence levels, so routing must have three tiers
D. It reduces API cost

**Q14.** You want the model to catch its own arithmetic errors on invoice totals. Best design?
A. Add a `notes` string field asking it to mention any discrepancies
B. Add `calculated_total` (number) and `conflict_detected` (boolean) to `required`, and compare with a tolerance in code
C. Ask the model in the prompt to double-check the math
D. Run a second extraction and diff the two results

**Q15.** Your sync pipeline sends the same 6,000-token system prompt (rules + 3 few-shot examples) with every one of 50,000 daily invoices. Which change cuts cost the most without changing behavior?
A. Remove two of the three few-shot examples
B. Enable prompt caching on the system prompt, keeping the invoice text last
C. Switch to batch and accept the 24-hour window
D. Enable streaming

**Q16.** An option in a question reads: *"Replace deterministic validation with a second Claude call that reviews the extraction."* Why is this usually wrong?
A. The API doesn't allow chained calls
B. It's non-deterministic, costs an extra round trip, and can hallucinate approval for checks that plain code can verify exactly
C. It would exceed the context window
D. Tool use can't be used twice on the same document

**Q17.** Invoice 03 states $5,037.58; line items plus tax compute to $5,027.58. Retry with feedback runs twice and the model returns the same numbers both times. What's the correct interpretation?
A. The retry mechanism is broken
B. The document genuinely disagrees with itself; the mechanism worked and the conflict should be flagged for review
C. `MAX_RETRIES` should be raised
D. The tolerance threshold should be widened to 10.00 so it passes

**Q18.** Your invoices arrive as scanned PDFs instead of text files. What has to change in the pipeline?
A. The schema — add OCR confidence fields
B. Everything — PDFs require a separate extraction service
C. Only the input: send the PDF as a document block. Schema, few-shot, validation, retry, and routing are unchanged
D. Switch to batch, since PDFs take longer

---

## 11. Answer key with distractor analysis

**Q1 — B.** Fabrication of *absent* data is a schema-expressibility problem. A plain `string` in `required` leaves the model no legal way to say "not present."
A: instructions help marginally but the contract still forbids the honest answer. C: detects a *well-formed* fabricated number — validation can't tell invented from real. D: temperature doesn't create an option that isn't in the schema.

**Q2 — B.** `any` guarantees *some* tool runs while letting the model choose which schema fits.
A: forces the invoice schema onto receipts and POs. C: `auto` permits a plain text reply. D: forbids tool use entirely.

**Q3 — B.** The data is present but unresolvable → enum + `"unclear"`.
A: nullable is for *absent* data; the doc *does* state terms, just by reference. C: `other`+`detail` is the taxonomy-fit hatch (categories), not the ambiguity hatch. D: makes the field optional — the model still fabricates when it does emit it.

**Q4 — B.** The signature sentence: strict schemas eliminate JSON *syntax* errors, not *semantic* ones.
A: `additionalProperties` blocks extra keys, not wrong values. C: may raise accuracy, doesn't detect anything. D: nothing was dropped; the value was wrong.

**Q5 — B.** 2–4 examples chosen for the *decisions* they teach: baseline, null/unclear handling, normalization + self-correction.
A: teaches only the easy case. C: one clean example anchors "fill every field." D: over budget, over-anchors, and country coverage isn't the axis — decision coverage is.

**Q6 — C.** "Info absent from document" is the canonical **non-retryable** error. Retrying wastes tokens and pressures fabrication.
A/B/D: all re-read a document that does not contain an invoice number.

**Q7 — B.** The exact lab shape: original doc + failed `tool_use` (`input` = failed extraction) + `tool_result` with matching `tool_use_id`, `is_error: True`, and the specific errors.
A: resampling, not correction. C: system prompts are for standing rules; this loses the failed extraction and per-doc specificity. D: no error context, no source document.

**Q8 — B.** Offline + high volume + cost-bound = Message Batches (50% / 24h / no SLA).
A: no discount, rate limits, hand-rolled backoff. C: streaming changes time-to-first-token for one request. D: caching helps but batch's 50% dominates here (and the two can combine).

**Q9 — B.** Batch results return in arbitrary order; `custom_id` is the only reliable correlation key.
A: expiry yields `expired` results, not mismatches. C: `auto` would cause missing extractions, not swapped ones. D: truncation, not misalignment.

**Q10 — C.** `20h − 24h` = negative buffer. The window alone blows the SLA.
A: that's the *30-hour* variant. B: submission frequency can't shrink a 24-hour processing window. D: "most batches finish sooner" is not a guarantee you may design against — there is no latency SLA.

**Q11 — B.** Aggregate masks stratum failure — the 100/95/90/**75** breakdown. Automate only validated (type × field) cells.
A: assumes uniform accuracy. C: you don't know which type is failing yet. D: confidence routing must itself be calibrated against labeled data first.

**Q12 — B.** Flags override a high rating: `auto_approve` requires `overall == "high"` **and** an empty flags array. One flag isn't the `>= 3` human-review trigger, so it lands in `spot_check`.
A: ignores flags. C: over-escalates and wastes the middle tier. D: an inferred tax rate is a source-ambiguity issue, not a retryable one.

**Q13 — B.** The middle tier exists to be sampled — that's the reviewer-capacity argument.
A: irrelevant. C: reverses cause and effect; the tiers are a business design, and you could route three levels into two lanes. D: routing happens after the API call; it doesn't change API cost.

**Q14 — B.** Independent recomputation plus a machine-readable boolean, verified in code with a float tolerance (`> 0.01`).
A: free text can't be branched on. C: no output artifact, nothing to route. D: doubles cost and gives two guesses with no ground truth.

**Q15 — B.** A large, byte-identical prefix on every request is the textbook caching case; keep static content first and the invoice last so the prefix stays valid. Cache reads run ~0.1× base input.
A: degrades the accuracy the examples buy. C: changes behavior — this is a sync pipeline; 24h latency may be unacceptable (and if it's acceptable, batch *and* caching together is the real answer). D: no cost effect.

**Q16 — B.** Deterministic checks belong in code; model calls are for judgment code can't do.
A/C/D: all false — chained calls are fine, context isn't the issue, and tool use can be reused freely.

**Q17 — B.** Retry is a mechanism, not a guarantee. Invoice 03 carries a volume-discount note; the source is genuinely inconsistent. `conflict_detected: true` plus a flag is the correct terminal state.
A: it engaged and produced consistent output. C: more retries won't fix the document. D: widening tolerance to hide a real $10 discrepancy defeats the check.

**Q18 — C.** Only the input block changes. The forced tool call, schema hatches, few-shot, validation, retry, and routing layers all still apply — that's the layered architecture paying off.
A: OCR confidence isn't a field you can get this way; use `confidence.flags`. B: PDFs are native document input. D: input format doesn't determine sync vs batch — latency tolerance does.

---

## 12. 60-second recall sheet

- **tool_use + JSON schema** = most reliable structured output. `block.input` is already a dict.
- **forced** = this tool · **any** = some tool · **auto** = maybe text · **none** = no tools.
- **null** = absent · **`"unclear"`** = ambiguous · **`"other"` + detail** = doesn't fit.
- Nullable fields use `["type", "null"]` **and** are excluded from `required`.
- Strict schema kills **syntax** errors, never **semantic** ones.
- Few-shot: **2–4 examples**, chosen for **ambiguous-case handling**, in the system prompt, XML-tagged. Without a `null` example, the model's prior is to **fill every field**.
- Self-correction = `calculated_total` + `conflict_detected`, both `required`.
- Validate in **deterministic code**, compare floats with a **tolerance** (`> 0.01`).
- Retry = original doc + failed `tool_use` + `tool_result(is_error=True)` with **specific errors**, bounded by `MAX_RETRIES`.
- **Retryable** = format/calculation. **Non-retryable** = info absent from source → human review.
- `detected_pattern` / `confidence.flags` → **dismissal pattern analysis** → suppress checks that are always dismissed for a doc type.
- Batch = **50% cost, 24h window, no latency SLA**, single-turn only, results **out of order**, correlate by **`custom_id`**, resubmit failures by `custom_id`.
- SLA math: `SLA − 24h = buffer`. Buffer ≤ 0 → sync.
- Confidence: enum `high/medium/low` + `flags[]`, `required`. `high` **and** no flags → auto_approve; `low` **or** ≥3 flags → human_review; else spot_check. Default missing → `low`.
- Calibrate on a **labeled validation set**; thresholds scale with the **cost of an error**.
- **Aggregate accuracy masks per-type failure** → **stratified sampling** per document type **and** per field. Automate only validated strata.
- Prompt caching: static first, variable last, ~0.1× read cost — for large fixed prompts on sync volume. Combines with batch.

---

*Built from `06_structured_extraction/` — README steps 3–9, `schema.py`, `main.py`, `batch.py`, `data.py`, `prompts/extraction_prompt.txt`. Tasks 4.2 · 4.3 · 4.4 · 4.5 · 5.5 (D4 + D5).*
