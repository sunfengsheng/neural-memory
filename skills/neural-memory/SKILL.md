---
name: neural-memory
description: Use at the start of every conversation to recall relevant context, when the user asks to remember/recall/forget something, when starting any non-trivial task, and at the end of a session to persist important learnings. Operates the neural-memory MCP server (6 tools).
---

# Neural Memory Skill

Operate the `neural-memory` MCP server as a persistent long-term memory layer. The server simulates human memory: neurons decay (Ebbinghaus curve), strengthen on recall, form synapses, and consolidate into schemas.

## The 6 MCP Tools

| Tool | Purpose | Key params |
|---|---|---|
| `recall` | Hybrid search (semantic + FTS + spreading activation) | `query`, `top_k=10`, optional `neuron_type`, `layer`, `tags` |
| `remember` | Store new memory + auto-embed + auto-link | `content`, `neuron_type`, `importance`, `tags`, optional `file_content`+`file_type`+`file_extension` |
| `reflect` | Run maintenance (decay, prune, promote, consolidate) | none |
| `forget` | Delete by ID or query | `neuron_id` OR `query`+`confirm=true` |
| `associate` | Manually link two neurons | `neuron_id_a`, `neuron_id_b`, `synapse_type`, `weight` |
| `memory_status` | System stats | none |

---

## Auto-Trigger: Session Start (EVERY conversation)

When this skill is loaded at the start of a conversation, **immediately and silently**:

1. Extract 1-3 keywords from the user's first message (project name, topic, error, person name, etc.)
2. Call `recall(query="<keywords>", top_k=5)`
3. If any result has score > 0.5, weave it into your response naturally ("Last time we worked on...", "Based on our previous session...")
4. If no relevant results, proceed silently — **never** say "I checked memory and found nothing"

**Examples:**
- User: "继续优化那个API" → `recall(query="API optimization", top_k=5)`
- User: "帮我看看这个bug" → `recall(query="bug fix debugging", top_k=5)`
- User: "我想给项目加个新功能" → `recall(query="project new feature", top_k=5)`

---

## Auto-Trigger: Session End

When any of these signals appear, **automatically persist** the session:

**End signals:**
- User says: "谢谢", "好了", "就这样", "thanks", "done", "that's all"
- Task is clearly completed (feature implemented, bug fixed, question answered thoroughly)
- User invokes `/remember` skill

**What to do (ALL steps mandatory):**

### Step 1: Session Summary (ALWAYS do this)

Write a concise summary of the entire conversation and store it as ONE episodic memory:

```
remember(
  content="[Session Summary] <date> — <1-2 sentence topic>. Details: <what was discussed, what was done, key decisions, outcomes, unresolved items>",
  neuron_type="episodic",
  importance=0.6,
  tags='["session-summary","<main-topic>","<date>"]'
)
```

The summary should capture:
- What the user asked for / wanted to accomplish
- What was actually done (files changed, commands run, decisions made)
- Final outcome (succeeded, partial, blocked)
- Any unresolved items or next steps

**Example:**
```
remember(
  content="[Session Summary] 2026-04-27 — Reinstalled neural-memory plugin from scratch. User had MCP -32001 timeout errors. Cleaned all plugin files (cache, marketplace, installed_plugins.json, known_marketplaces.json), re-cloned from GitHub, manually registered in installed_plugins.json. Fixed successfully, all 6 MCP tools responding. Then updated SKILL.md to add full session summary memory on every conversation end.",
  neuron_type="episodic",
  importance=0.6,
  tags='["session-summary","neural-memory","plugin-install","2026-04-27"]'
)
```

### Step 2: High-Value Extractions (if any)

Additionally, extract and store separately any:
- User preferences stated ("I prefer X", "always use Y") → `importance=0.8`
- Architectural decisions or project conventions → `importance=0.7`
- Non-trivial bugs solved (root cause + fix) → `importance=0.6`
- Milestones reached → `importance=0.6`

These get their own `remember()` calls with appropriate `neuron_type` and `tags`.

### Step 3: Reflect if needed

If session was long (>30 min) or >10 memories were created, call `reflect()`.

---

## Manual Patterns

### Recall on demand
User references past conversations ("last time", "previously", "earlier we"):
```
recall(query="<topic from user message>", top_k=5)
```

### Capturing a preference
User says "I always use pytest, never unittest":
```
remember(
  content="User prefers pytest over unittest for all Python testing",
  neuron_type="semantic",
  importance=0.8,
  tags='["preference","python","testing"]'
)
```

### Capturing a fix
After solving a bug:
```
remember(
  content="Fix for <error>: root cause was X, solution was Y. Affects <component>.",
  neuron_type="procedural",
  importance=0.6,
  tags='["bugfix","<lang>","<component>"]'
)
```

### Capturing reusable code
```
remember(
  content="Utility: <one-line description>",
  neuron_type="procedural",
  importance=0.6,
  tags='["snippet","<lang>"]',
  file_content="<the actual code>",
  file_type="code/python",
  file_extension=".py"
)
```

### Forgetting (always preview first)
```
# Step 1: preview
forget(query="<what to delete>", top_k=5, confirm=false)
# Show the user the matches, get confirmation
# Step 2: execute
forget(query="<same query>", confirm=true)
```

### Associating related memories
When two recalled memories are clearly related but not yet linked:
```
associate(neuron_id_a, neuron_id_b, synapse_type="semantic")
```

### End-of-session consolidation
```
reflect()
```

---

## Importance Scale

| Value | Meaning |
|---|---|
| 0.9-1.0 | Critical: identity, core preferences, security-relevant |
| 0.7-0.8 | Strong: project conventions, stated preferences, key decisions |
| 0.5-0.6 | Normal: bug fixes, useful snippets, meaningful events |
| 0.3-0.4 | Weak: passing observations |
| < 0.3 | Don't bother saving |

## neuron_type Guide

- `episodic` — "On 2026-04-23 we debugged X" (time-bound events)
- `semantic` — "FastAPI uses Pydantic for validation" (timeless facts)
- `procedural` — "To deploy, run `make ship`" (how-to)
- `schema` — auto-generated by `reflect`, do not create manually

## Anti-patterns

- ❌ Narrating every memory op ("Let me save that to memory...")
- ❌ Storing chat fluff ("user said hi")
- ❌ Calling `forget` without preview
- ❌ Forgetting to pass `tags` (kills future recall precision)
- ❌ Using importance=1.0 by default (devalues real high-importance memories)
- ❌ Announcing "no memories found" when recall returns empty
