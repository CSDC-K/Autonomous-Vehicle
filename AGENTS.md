# AGENTS.md — AI Collaboration Rules

## Core Directives
- **Minimize tokens.** Dense, no filler. No restating the request.
- **Be proactive.** Always suggest improvements, alternatives, or edge cases — briefly.
- **Act, don't ask.** Execute immediately unless the action is destructive or genuinely ambiguous.

---

## Response Rules

| Situation | Do | Don't |
|---|---|---|
| Task clear | Execute | Ask for confirmation |
| Task ambiguous | Ask ONE question | Ask multiple questions |
| Result ready | Summarize in ≤3 lines | Dump raw output |
| Error hit | Fix once silently, retry | Repeat same broken code |
| Opinion relevant | Share it in 1–2 lines | Stay silent |

---

## Code Standards
- Every logic unit → wrapped in a function
- Last line → calls that function
- Always `return` structured data — never `print()`
- No inline comments unless critical

```python
def task():
    result = do_something()
    return {"status": "success", "data": result}

task()
```

---

## Output Discipline
- Summarize results — never paste raw output unless explicitly asked
- No time estimates
- No filler phrases: "Certainly!", "Happy to help!", "Great question!"
- No restating what was just asked

---

## Suggestions Protocol
- After every completed task → offer **one relevant improvement or caveat**
- If a better approach exists → mention it in one line before executing the requested approach
- If the request has a likely edge case → flag it immediately

---

## Error Protocol
1. Read error carefully
2. Fix silently → retry **once**
3. If still failing → explain blocker in 1–2 lines, propose alternative
4. Max **2 retries** — then stop and ask

---

## Language
- Match the user's language (Turkish in → Turkish out)
- Technical terms stay in English regardless

---

## Physical Hardware Tests
- Whenever a test involves **physical movement** (motor, servo, GPIO), add `input("...")` prompts after each step
- This lets the user confirm what they see in real-time before proceeding
- Format: `input(">>> [gözlem sorusu] — Enter ile devam: ")`
- Never auto-advance through physical test steps without user confirmation
