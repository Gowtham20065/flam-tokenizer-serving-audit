# AI Usage Disclosure

## 1. Transparency Statement
I used Claude Sonnet 4.6 and Gemini 3.8 Flash via the Antigravity developer environment to draft scripts, parse dataset schemas, and sanity-check arithmetic while building this submission. I separately used Claude (chat interface) as a second-pass auditor to review the finished submission before the defense — it read through the code and deliverables, flagged issues, and proposed fixes, which I then verified myself.

As required by the assignment's Ground Rules, every claim, code snippet, and mathematical derivation was independently verified against first principles before it went into the submission — including a final independent re-run of Part A outside Antigravity, in a plain VS Code + venv setup, to make sure nothing depended on the AI tool's environment specifically.

---

## 2. Where AI Misled Me (and How I Caught It)

1. **Flagged `random.seed(1337)` as a bug when it isn't one.**
   Antigravity's first pass flagged the seed line as "unnecessary state" / a likely bug. I checked: `random` is never called anywhere in the script. An AST trace and a direct diff (with/without the line) confirmed delta = 0.000. Flagging it without that check would have cost -5 points under the rubric, so I classified it as a harmless red herring instead.

2. **Got the direction of the double-space bug backwards.**
   The AI claimed `line.split(" ")` "inflates" Hindi fertility because empty strings add extra tokens. That's wrong — the empty string lands in the *denominator* (words), not the numerator (tokens), so it actually *deflates* the ratio. I isolated the affected line (`किताबें  अलमारी में रखी हैं।`) and measured it directly: fixing the split raised Hindi fertility from 7.45 to 7.60, not down.

3. **Suggested a dataset loader that doesn't work.**
   The AI generated `load_dataset("facebook/flores", "hin_Deva", split="devtest")`. That dataset is gated on the Hub and needs an auth token, and a fallback (`Muennighoff/flores200`) relies on a custom loading script that's deprecated in current `datasets` versions. I had to diagnose both failures myself and switch to the open `openlanguagedata/flores_plus` mirror.

4. **Left a stale, contradictory print statement in `corrected_analysis.py`.**
   An earlier AI-assisted draft of the closing "which denominator should drive routing" summary hardcoded placeholder numbers (~0.17/~0.50 tok/byte, "~3x"/"~1.2x") written before the real 1,012-sentence corpus existed. I didn't catch this until a pre-defense read-through — the placeholder text directly contradicted the script's own saved CSV and my memo (2.90x for GPT-2, 0.49x for XLM-R).

5. **Second-pass audit caught the bug above — but couldn't verify the fix end-to-end itself.**
   I had Claude (chat) review the finished repo as a fresh pair of eyes before the defense. It caught the stale print block in point 4 by reading the code and comparing it against the saved CSV, and it wrote the replacement code. It could check the fix's *arithmetic* against my existing CSV, but it could not actually execute `tiktoken`/`transformers` itself (no access to the OpenAI vocab endpoint or the HF Hub from its own environment) — so it could not independently confirm the fix by running the pipeline. I ran the fixed script myself, twice, in two separate environments (Antigravity and a clean VS Code venv), and confirmed both runs now print the corrected ratios (2.90x, 0.49x) matching the memo. The catch was AI-assisted; the verification was mine.

---

## 3. Where AI Accelerated My Workflow
- Boilerplate for the HuggingFace tokenizer wrapper and CSV writer in `corrected_analysis.py`.
- Formatting the large comparison tables across 7 languages and 4 denominators.
- Slicing and cross-checking rows in `bench_log.csv` across prompt lengths.

---

## 4. Preparedness for the Live Defense
Every calculation in `partB/calculations.md` was verified by hand, and Part A was re-run end to end in two independent environments (Antigravity, and a separate VS Code + plain venv setup) to confirm the fix in point 4/5 above holds regardless of tooling. If asked in the defense to change a parameter and recalculate live, modify `audit_fertility.py` on the spot, or explain why `tok/byte` beats `tok/grapheme` for this decision, I can derive each result from first principles without help — including explaining exactly what the second-pass AI review did and did not verify.