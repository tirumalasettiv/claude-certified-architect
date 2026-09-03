# S1 — Customer Support Resolution Agent · CCA-F Max-Score Guide

> Scope: Scenario **S1**, Domains **D1** (Agent Architecture) + **D2** (Tool Design) + **D5** (Context Management).
> Tasks **1.1, 1.2, 1.4, 1.5, 2.1, 2.2, 2.3, 5.1, 5.2** in depth, plus the adjacent concepts that show up as distractors.
> Every code snippet is traceable to `main.py` / `tools.py` / `system_prompt.txt` in this lab.

---

## 0. How this section is actually tested

Exam items in S1 are **scenario-framed multiple choice**: a paragraph describing a support-agent pipeline failing in a specific way, then four plausible fixes. You max the score by classifying the **failure mode** first, then picking the mechanism that owns it.

| The failure you're shown | The one technique that owns it | Task |
|---|---|---|
| Loop never stops / stops too early / relies on text parsing | Check **`stop_reason`** (`tool_use` → continue, `end_turn` → stop) | 1.1 |
| Iteration cap is the *only* thing preventing runaway loops | `stop_reason` is primary; `MAX_LOOP_ITERATIONS` is a safety net, never the driver | 1.1 |
| One role needs isolated context, different tools, or parallel work | **Coordinator–subagent** pattern (multi-agent), not one bloated agent | 1.2 |
| A financial/compliance step *might* be skipped by the model | **Programmatic prerequisite gate** in code — prompt instructions are not enough | 1.4 |
| Customer asks two things in one message | **Decompose into distinct items**, investigate each with shared context, synthesize one reply | 1.4 |
| Policy must be enforced with 100% certainty (e.g., refund cap) | **PostToolUse hook** intercepts the result before the model sees it | 1.5 |
| Tool output formats vary across sources (timestamps, locales) | PostToolUse hook for **data normalization** | 1.5 |
| Model picks the wrong tool / confuses two tools | Rewrite **tool descriptions** — differentiate purpose, inputs, when-to-use | 2.1 |
| Agent has 15+ tools, selection gets unreliable | **Fewer, focused tools** per agent — distribute across agents instead | 2.3 |
| Tool fails and the agent doesn't know whether to retry | **Structured error**: `errorCategory` + `isRetryable` | 2.2 |
| A search legitimately found nothing | **Valid empty result**, not an error — don't conflate with an access failure | 2.2 |
| Need a specific tool called first, or guaranteed *some* tool call | **`tool_choice`**: forced `{"type":"tool",...}` or `{"type":"any"}` | 2.3 |
| Long conversation, exact values drift or get lost in summarization | **`case_facts`** — persistent structured fact extraction outside the summarized history | 5.1 |
| Important detail is buried in the middle of a long context | **"Lost in the middle"** — put key facts at the start/end, use section headers | 5.1 |
| Tool results bloat the context with irrelevant fields | **Trim tool outputs** to relevant fields before appending to history | 5.1 |
| Escalate based on customer tone | **Wrong** — sentiment is not a reliable proxy for complexity; use explicit criteria | 5.2 |
| Ambiguous multi-match lookup (e.g., name matches 2 customers) | **Ask for a disambiguating identifier**, don't guess | 5.2 |

**Read the last sentence of the stem first.** It names the failure. The rest is set dressing.

---

## 1. The one mental model

> **The model drives the loop; the code enforces what the model cannot be trusted to enforce.** Anything the model decides (which tool, when to stop, when to escalate) should be driven by a real signal (`stop_reason`, explicit criteria) — never by parsing prose. Anything that must be guaranteed (identity verification before a refund, a dollar cap) belongs in code that runs whether or not the model complies.

```
User message
      │
      ▼
[ AGENTIC LOOP ]  ── stop_reason=="tool_use" → execute tool → append → continue
      │              stop_reason=="end_turn"  → stop, return final text
      │              MAX_LOOP_ITERATIONS is a safety net, not the driver
      ▼
[ PREREQUISITE GATE ] ── code-level check BEFORE a tool executes
      │                  e.g. block process_refund unless get_customer succeeded
      ▼
[ TOOL EXECUTION ] ── clear, differentiated descriptions drive correct selection
      │
      ▼
[ POSTTOOLUSE HOOK ] ── intercepts the RESULT after execution, before the model sees it
      │                  compliance (refund cap) + normalization (dates/locales)
      ▼
[ STRUCTURED ERROR ] ── errorCategory + isRetryable tell the model how to recover
      │
      ▼
[ CASE FACTS ] ── verified facts persisted outside conversation history
      │            survives summarization, mitigates "lost in the middle"
      ▼
[ ESCALATION ] ── explicit criteria, never sentiment; ambiguity → ask, don't guess
```

Layers are **complementary, not alternatives**. An answer option that says "instead of the gate, tell the model in the prompt to always verify first" is almost always the wrong option — prompts fail; gates don't.

---

## 2. Task 1.1 — Design and implement agentic loops

### 2.1 Exam concept

The agentic loop is driven by **`stop_reason`**, not by parsing what the model said. Two values matter here:

- `"tool_use"` → the model wants to call one or more tools. Execute them, append results, loop again.
- `"end_turn"` → the model is done. Extract the final text and stop.

### Best design approach

```python
for iteration in range(MAX_LOOP_ITERATIONS):
    response = client.messages.create(
        model=MODEL, max_tokens=1024, system=system_prompt,
        tools=ALL_TOOLS, messages=messages,
    )

    if response.stop_reason == "end_turn":
        # final answer — stop
        break

    if response.stop_reason == "tool_use":
        # execute each tool_use block, append assistant turn + tool_result turn, continue
        ...
else:
    # MAX_LOOP_ITERATIONS exhausted — safety net triggered, not the normal path
    ...
```

The `for ... else` shape matters: the safety net (`MAX_LOOP_ITERATIONS`) only fires if the loop *never* hit a `break` — i.e., `stop_reason` never resolved to `end_turn`. That's the tell that something is wrong (a stuck loop), not the expected exit.

### The three anti-patterns the exam tests

| Anti-pattern | Why it's wrong |
|---|---|
| Iteration cap as the **primary** stopping mechanism | Stops correct multi-step work early, or masks a genuinely broken loop as "normal." The cap is a safety net only. |
| Parsing natural language for a completion signal (e.g., checking if the agent said "I'm done") | Brittle — wording varies, and the model may say "done" mid-reasoning. `stop_reason` is the actual API contract. |
| Checking for assistant **text content** as a completion indicator | The model can emit text *and* a tool call in the same response (interleaved thinking) — text presence alone doesn't mean the turn is over. |

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Fixed number of tool calls, then stop | Under-serves multi-step cases, over-serves simple ones. Not adaptive. |
| Regex-match the response for "I'll now call X" | Fragile, model-wording-dependent, breaks across model versions. |
| Stop when no tool call appears in the raw content list, ignoring `stop_reason` | Usually correlates with `end_turn`, but `stop_reason` is the documented, stable signal — don't reinvent it. |

---

## 3. Task 1.2 — Orchestrate multi-agent systems with coordinator-subagent patterns

### 3.1 Exam concept

This lab is **single-agent** — one agent, four tools, one role. The exam tests whether you know *when that stops being sufficient*.

### Decision table

| Single agent (this lab) | Coordinator–subagent (Lab 03 pattern) |
|---|---|
| One role handles the whole task | Distinct roles need **isolated context** from each other |
| Small, focused tool set (≤ ~5 tools) | Subagents need **different tool sets** per role |
| Sequential reasoning is fine | Work can be **parallelized** across subagents |
| Simpler to build, trace, and debug | Coordinator spawns subagents (e.g., via a `Task`-style tool), collects results, synthesizes |

> Trap: an option that says "just give the single agent more tools to cover the new responsibility" is the wrong answer when the real signal is *context isolation* or *parallelism* — that's an architecture problem, not a tool-count problem.

---

## 4. Task 1.4 — Multi-step workflows with enforcement and handoff patterns

### 4.1 Exam concept — two distinct mechanisms

1. **Programmatic prerequisite gates** — code-level checks that block a tool call unless an earlier step already succeeded.
2. **Multi-concern decomposition** — when one user message bundles multiple requests, investigate each and synthesize a single coherent reply, without dropping either concern.

### 4.2 The prerequisite gate

```python
def check_prerequisite(tool_name, tool_history):
    """Block process_refund unless get_customer has already been called successfully."""
    if tool_name == "process_refund" and "get_customer" not in tool_history:
        return {
            "error": True,
            "errorCategory": "validation",
            "isRetryable": True,
            "message": "Customer identity must be verified via get_customer before processing a refund.",
        }
    return None
```

`tool_history` is a `set()` of tool names that completed **without an error** — populated in `execute_tool` only when `not result.get("error")`. The gate runs *before* the tool executes, so a missing prerequisite never reaches business logic at all.

**Why this can't live in the prompt alone:** "always verify the customer first" is a instruction with a non-zero failure rate. A gate is a guarantee — the refund tool physically cannot run without the prerequisite, regardless of what the model decides to do that turn.

### 4.3 Multi-concern requests

Query: *"My headphones from ORD-5501 arrived damaged and I need a refund. Also, can you check if my other order ORD-5502 has been delivered?"*

Correct behavior: one `get_customer` call establishes shared identity context, then both concerns are investigated (`lookup_order` × 2, `process_refund` × 1) and the final reply addresses both — the agent does not abandon one issue to chase the other.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Trust the system prompt instruction alone ("always verify first") | Non-zero failure rate on a financial operation is unacceptable; use a code-level gate. |
| Re-verify the customer before every single tool call | Correct in spirit but wasteful — the gate checks `tool_history`, not "verify every time." One successful `get_customer` satisfies the prerequisite for the rest of the conversation. |
| Handle only the first concern in a multi-concern message, ask the customer to repeat the second | Poor UX and fails the decomposition requirement the exam tests. |
| A second LLM call to "double check" the prerequisite was met | Non-deterministic and unnecessary — a `set()` membership check is deterministic and free. |

---

## 5. Task 1.5 — Apply Agent SDK hooks for tool call interception and data normalization

### 5.1 Exam concept — two use cases for a PostToolUse hook

1. **Compliance enforcement** — deterministically block or replace a result that violates policy, regardless of what the model would have done with the raw result.
2. **Data normalization** — convert inconsistent tool output formats (timestamps, locale strings) into one consistent shape before the model reasons over them.

### Best design approach — compliance

```python
def post_tool_use_hook(tool_name, tool_input, tool_result):
    """Intercept tool results to enforce policy rules."""
    if tool_name == "process_refund" and tool_input.get("amount", 0) > MAX_REFUND_AMOUNT:
        return {
            "error": True,
            "errorCategory": "policy_violation",
            "isRetryable": False,
            "message": f"Refund amount ${tool_input['amount']:.2f} exceeds the ${MAX_REFUND_AMOUNT} policy limit. This refund must be escalated to a human agent.",
            "action": "escalate_to_human",
        }
    return tool_result
```

The hook runs **after** `process_refund` executes (the refund logic itself has no dollar-cap awareness) and **before** the model ever sees the raw success result. The replacement carries `isRetryable: False` (retrying won't help — the amount is still over the cap) and an `action` hint that steers the model toward the correct recovery path (`escalate_to_human`) without hardcoding the escalation call itself.

### Best design approach — normalization (the exam tests this beyond what the lab implements)

> Tools from different MCP servers may return dates as Unix timestamps, ISO 8601, or locale-formatted strings. A PostToolUse hook converts all of them to one consistent format before the model processes the result — so the model never has to reconcile three date formats in the same conversation.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Tell the model in the prompt "never process refunds over $500" | Same failure mode as §4.3 — a compliance rule with financial consequences needs a guarantee, not a request. |
| Validate the amount inside `process_refund` itself and raise | Conflates business logic with policy enforcement; harder to test/change the cap independently, and doesn't generalize to normalization use cases. The hook pattern separates "did the tool work" from "is the result compliant/well-formed." |
| Block the tool call **before** execution instead of intercepting the result | Wrong stage for this case — you need the actual refund amount from the tool input, which you already have; but more importantly, some policy checks (e.g., normalization) inherently require the *result*, not just the input. Know PostToolUse (after) vs a prerequisite gate (before) are different stages solving different problems. |
| A second LLM call to review the result for policy compliance | Non-deterministic, costs a round trip, can be argued around by the model. A hook is a plain conditional in code. |

---

## 6. Task 2.1 — Design effective tool interfaces with clear descriptions and boundaries

### 6.1 Exam concept

Tool **descriptions** are the primary mechanism the model uses to select the right tool. A description must state: what the tool does, its inputs, its outputs, and **when to use it versus alternatives**.

### Best design approach

```python
get_customer_schema = {
    "name": "get_customer",
    "description": (
        "Look up a customer record by customer ID or email address. Returns customer "
        "profile including name, email, account status, and tier. Use this tool FIRST "
        "before any account action — customer identity must be verified before processing "
        "refunds or changes. Accepts either customer_id (e.g., 'CUST-1001') or email "
        "(e.g., 'maria.santos@example.com'), not both."
    ),
    ...
}
```

Note what this description does: names the ordering constraint ("use this FIRST"), disambiguates the two valid input shapes ("either ... not both"), and gives concrete examples. Every tool in `tools.py` follows the same pattern — `lookup_order` says when to use it ("before processing a refund"), `process_refund` repeats the verification precondition, `escalate_to_human` enumerates the four trigger conditions directly in the description.

### The keyword-sensitivity trap

> System prompt wording can **override** what a good tool description says. If the system prompt says "always look up customer orders first," the model may bias toward `lookup_order` even when `get_customer` is the correct first step. Reviewing the system prompt for keyword-sensitive phrasing that creates unintended tool associations is itself an exam-tested skill — a good description doesn't protect you from a conflicting system prompt.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Minimal description ("looks up a customer") | Underspecifies inputs, ordering, and boundaries — the model has to guess, and guesses inconsistently across runs. |
| Put the "when to use" guidance only in the system prompt, keep tool descriptions generic | Splits the selection signal across two places and risks the keyword-override problem above; the description is read at the point of the decision — put the constraint there. |
| Rely on parameter names alone to convey meaning | Parameter names help typing, not selection. The `description` field is what the model reasons over when picking *which* tool. |

---

## 7. Task 2.2 — Implement structured error responses for MCP tools

### 7.1 Exam concept

Every failure path returns a **structured** error, not a bare exception or a plain string, so the model has enough information to decide how to recover.

### Best design approach

```python
{
    "error": True,
    "errorCategory": "validation",   # or "business", "policy_violation", ...
    "isRetryable": True,             # or False
    "message": "human/model-readable explanation",
}
```

- `errorCategory` classifies *why* it failed (`validation` — bad/missing input, e.g. unknown customer ID; `business` — a business rule blocked it, e.g. order not yet delivered; `policy_violation` — a hard compliance cap, e.g. refund over $500).
- `isRetryable` tells the model whether trying again (with corrected input) could succeed, or whether this is terminal for this path (e.g., refund over the cap will never succeed by retrying — it needs escalation, hence `False`).

### The distinction the exam pushes hardest: access failure vs valid empty result

| | Access failure | Valid empty result |
|---|---|---|
| Example | Timeout, service down, auth error | Search ran fine, found 0 matches |
| Is it an error? | Yes — `{"error": True, ...}` | **No** — `{"results": []}` is a success |
| Recovery | Retry / backoff / escalate depending on `isRetryable` | Just tell the user nothing was found — no retry needed |

Conflating the two means either retrying a query that will always return zero results (waste), or treating a real outage as "nothing found" (silently wrong answer to the customer).

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Raise a raw Python exception and let it bubble up | The model never sees it structured — it either crashes the loop or the exception string gets stuffed into content with no `errorCategory`/`isRetryable` to act on. |
| Return `None` or an empty string on failure | Ambiguous with a legitimately empty/absent value — the model can't tell "tool failed" from "tool found nothing." |
| A single generic `"error": "something went wrong"` for every failure | No `errorCategory` to differentiate business rule vs validation vs policy; no `isRetryable` to decide next action. The model is left guessing. |
| Treat every empty result as an error | Triggers needless retries or escalations for a perfectly successful "no matches" outcome. |

---

## 8. Task 2.3 — Distribute tools across agents and configure tool choice

### 8.1 Exam concept — two separate levers

1. **Tool count / distribution**: this agent has exactly **4** tools — a focused set for one role. Giving a single agent too many tools (the exam's example: 18) degrades selection reliability; the fix is to distribute tools across specialized agents (§3), not to keep piling onto one.
2. **`tool_choice`**: controls *whether/which* tool must be called on a given turn.

### The `tool_choice` modes — memorize this table

| Mode | Syntax | Guarantees | Use when |
|---|---|---|---|
| **auto** (default, used in this lab) | `{"type": "auto"}` | Nothing forced — model may reply with text or call a tool | Conversational agents where a plain text reply is a valid outcome. This lab's `run_agent` omits `tool_choice`, i.e. uses `auto`. |
| **any** | `{"type": "any"}` | *Some* tool is called — model picks which | You need structured output/action, not conversational text, but don't care which registered tool handles it |
| **Forced** | `{"type": "tool", "name": "extract_metadata"}` | **This specific tool** is called | A specific tool must run first, e.g. before an enrichment step. (Lab 06 exercises this for guaranteed structured extraction.) |
| **none** | `{"type": "none"}` | No tool is called | Force a text-only turn even though tools are registered |

> Trap: this lab is correctly `auto` — a support agent needs to be able to just *answer* a question in plain text sometimes. An option that forces `tool_choice: any` on every turn of a conversational agent is wrong; it removes the model's ability to simply reply.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| One agent with 15+ tools instead of distributing across specialized agents | Selection reliability degrades as the tool list grows; the fix is architectural (§3), not prompt tuning. |
| Force `tool_choice: any` on a conversational support agent | Breaks the "just answer in text" path that this domain legitimately needs. |
| Force a specific tool on every turn "to be safe" | Removes the model's ability to route correctly turn-by-turn; forced choice is for a known, single required step, not the steady state of a multi-turn conversation. |

---

## 9. Task 5.1 — Manage conversation context across long interactions

### 9.1 Exam concept — three related mechanisms

1. **`case_facts`** — persistent, structured extraction of verified transactional facts (customer_id, order_id, refund_amount, etc.) that survives progressive summarization.
2. **"Lost in the middle"** — models reliably process the beginning and end of long inputs but may miss content buried in the middle.
3. **Trimming tool outputs** — reduce token accumulation by keeping only the fields that matter before appending to history.

### 9.2 Case facts — the problem it solves

In a long conversation, older turns get summarized to save tokens. Progressive summarization loses *exact* values: "$249.99 refund" becomes "a refund," "order ORD-5501" becomes "the customer's order." When the agent later needs the exact number, it either hallucinates one or answers vaguely.

### Best design approach

```
## Case facts
<case_facts>
{case_facts}
</case_facts>
```

```python
system_prompt = system_template.format(case_facts=case_facts)
```

- Lives in an **XML-tagged section** with a **template variable** — never built with ad-hoc string concatenation.
- Updated after **each successful (non-error) tool result** with only the fields that matter (`customer_id`, `customer_name`, `order_id`, `order_total`, `refund_amount`, `refund_status`) — not the full raw tool response.
- Re-injected into the system prompt on **every loop iteration**, so it survives regardless of what happens to the message history.

### 9.3 "Lost in the middle"

> Place key facts at the **beginning** (or end) of aggregated input, and use section headers to make important content easy to locate. The `<case_facts>` block sitting near the top of the system prompt — rather than buried in a 20-turn message history — is a direct application of this mitigation.

### 9.4 Trimming tool outputs

> Full tool results accumulate in conversation history disproportionately to their relevance. A 40-field order lookup should be trimmed to the fields that matter for the task at hand (e.g., return-relevant fields only) *before* it's appended — not left for the model to filter out mentally on every subsequent turn.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Rely on the raw (summarized) conversation history for exact values | Summarization is lossy by design — exact numbers and IDs are exactly what gets rounded away. |
| Re-call the tool every time an exact value is needed | Wasteful and doesn't scale; `case_facts` is a cache of what's already been verified. |
| Put case facts in a `user` message instead of the system prompt | Works but is harder to keep current across iterations and loses the "always visible, top-of-context" placement that mitigates lost-in-the-middle. |
| Never trim tool outputs — keep full results "just in case" | Guarantees token growth is dominated by irrelevant fields, accelerating the point where summarization (and its losses) kicks in. |
| Summarize *case_facts* itself when the context gets long | Defeats the purpose — case_facts exists specifically to be the thing that does NOT get summarized. |

---

## 10. Task 5.2 — Design escalation and ambiguity resolution patterns

### 10.1 Exam concept

Escalation decisions must be driven by **explicit criteria**, not by sentiment. A frustrated customer with a simple, resolvable issue does not need a human; a calm customer hitting a genuine policy gap does.

### The four escalation criteria (this lab)

```
Escalate to a human agent when ANY of these conditions is true:
- The customer explicitly requests a human agent or manager
- The request involves a policy gap — the policy does not clearly cover the situation
- You cannot make progress after two tool calls on the same issue
- The case involves suspected fraud indicators

Do NOT escalate based on customer tone or sentiment alone.
```

### Few-shot examples reinforce the criteria — including a negative example

| # | Customer message | Correct action | Why |
|---|---|---|---|
| 1 | "I want to talk to a real person." | Escalate immediately, `customer_request` | Explicit request — no further investigation needed first |
| 2 | Gift bought under one account, recipient wants to return it | Escalate, `policy_gap` | Standard refund policy doesn't cover a different-recipient return |
| 3 | "This is ridiculous! ... still hasn't arrived!" | **Do NOT escalate** — look up order status, give a concrete update | Frustration alone is not a trigger; the issue is resolvable |

Example 3 is the highest-yield one on the exam: it's the negative case that proves sentiment is explicitly excluded as a trigger.

### Ambiguity resolution — the multi-match case

> When a customer lookup (e.g., by name) returns **multiple matches**, the agent should ask for an additional identifier (email, phone) to disambiguate — never pick one heuristically (e.g., "most recent," "first alphabetically"). This lab's data has unique customers per identifier, but the exam scenario introduces genuine ambiguity to test this.

### Why the alternatives lose

| Alternative | Why it loses |
|---|---|
| Escalate whenever the customer sounds upset/uses exclamation points | Sentiment is an unreliable proxy for complexity — directly contradicted by Example 3. |
| Never escalate, always attempt resolution | Ignores the explicit criteria (customer request, policy gap, fraud) that exist precisely because some cases genuinely need a human. |
| On a multi-match lookup, pick the most plausible candidate and proceed | Risks acting on the wrong customer's account — always disambiguate explicitly instead of guessing. |
| Escalate after any single failed tool call | Too aggressive — the criterion is "no progress after **two** tool calls on the same issue," not one. |

---

## 11. Adjacent concepts that appear as distractors

These aren't S1 task statements, but they show up in S1 answer options.

### Single-agent vs multi-agent (ties to Task 1.2, contrasts with Lab 03)
This lab is single-agent by design — one role, one tool set. Lab 03's coordinator spawns subagents via a Task-style tool for isolated context / parallel work. An exam item describing this lab's exact shape (4 tools, one conversational role, sequential reasoning) and asking "should this be split into subagents?" — the answer is **no**, it doesn't need context isolation or parallelism.

### Hooks beyond PostToolUse (ties to Task 1.5)
The exam may reference other hook points (PreToolUse, on-stop) from the broader Agent SDK hook system. This lab exercises PostToolUse specifically (intercept the *result*). Know that a prerequisite gate (§4.2, checked *before* execution) and a PostToolUse hook (§5, checked *after* execution) are different stages — don't conflate "block the call" with "rewrite the result."

### `tool_choice: any` for guaranteed structured output (ties to Task 2.3, previewed from Lab 06)
This lab stays on `auto` because conversational text is a valid outcome. Lab 06's extraction pipeline forces a specific tool because a text reply is *never* valid there. If a stem describes a scenario where a text-only reply would be a failure, the answer moves toward forced/`any` `tool_choice` — that's a Lab 06-flavored distractor showing up in an S1-shaped question.

### Task 4.2 — few-shot prompting for consistency (from Lab 06)
The escalation few-shot examples in `system_prompt.txt` are the same mechanism Lab 06 uses for extraction consistency: instructions alone under-specify judgment calls; 2–4 concrete examples (including a negative one) fix that. If a stem frames this lab's escalation behavior as inconsistent across runs, the fix is still "add/refine few-shot examples," not "add more prose criteria."

---

## 12. One-page decision table

| Symptom in the stem | Layer | Concrete fix |
|---|---|---|
| "loop never stops / stops too early" | Loop | check `stop_reason`, not text or iteration count |
| "the cap always gets hit" | Loop | `MAX_LOOP_ITERATIONS` is a safety net — the real bug is `stop_reason` never resolving to `end_turn` |
| "one role needs separate context/tools/parallelism" | Architecture | coordinator–subagent pattern |
| "refund processed on an unverified customer" | Code | prerequisite gate before the tool executes |
| "customer asked two things, agent only answered one" | Code/Prompt | decompose, investigate both, synthesize one reply |
| "refund over the policy cap went through" | Code | PostToolUse hook intercepting the result |
| "dates/timestamps inconsistent across tool sources" | Code | PostToolUse normalization hook |
| "wrong tool selected" | Prompt | rewrite tool descriptions — purpose, inputs, when-to-use |
| "system prompt biases tool selection unexpectedly" | Prompt | check for keyword-sensitive phrasing overriding descriptions |
| "18 tools, selection degrades" | Architecture | distribute tools across agents, don't just prune |
| "tool failed, agent doesn't know whether to retry" | Contract | `errorCategory` + `isRetryable` |
| "empty search result treated as a failure" | Contract | valid empty result ≠ access failure |
| "need a specific tool to run, or a guaranteed tool call" | API | forced `tool_choice` / `{"type":"any"}` |
| "exact $ amount / order ID drifted after several turns" | Context | `case_facts` extraction |
| "key fact buried in a long context gets missed" | Context | lost-in-the-middle — place at start/end, use headers |
| "context grows fast from verbose tool results" | Context | trim tool outputs to relevant fields before appending |
| "escalates because customer sounds angry" | Process | sentiment ≠ escalation trigger; use explicit criteria |
| "lookup returns 2+ matches" | Process | ask for a disambiguating identifier, don't guess |

---

## 13. How to pick when all four options look right

1. **Name the failure mode in one word** — loop-control / architecture / enforcement / normalization / selection / error-contract / context-loss / escalation-trust. Then use the table in §0.
2. **Match the layer.** Loop control is a signal (`stop_reason`); enforcement is code (gates, hooks); selection is prompt engineering (descriptions); context management is structural (case_facts, trimming); escalation is criteria-driven, not sentiment-driven.
3. **Prefer the mechanism that guarantees over the one that requests.** A code-level gate beats a system prompt instruction every time compliance must be certain.
4. **Prefer the documented API signal over parsed text.** `stop_reason` beats "does the response contain the word done."
5. **Watch for "instead of."** Layers are complementary — a gate and a hook and good tool descriptions all coexist. An option that removes a working layer to add another is usually the trap.
6. **Sentiment is never the answer** to an escalation question in this scenario — if an option says "escalate when the customer seems upset," it's wrong regardless of what else it says.
