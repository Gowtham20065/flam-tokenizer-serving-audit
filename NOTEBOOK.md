# Antigravity Lab Notebook

## Project: Tokenizer Audit, Multi-Lingual Evaluation & Serving Capacity Reconciliation
**Candidate:** Gowtham Boda
**Date Range:** September 2026
**Environment:** Built in Antigravity (Claude Sonnet 4.6, Gemini 3.8 Flash); verified with a plain Python venv in VS Code
**Status:** Complete

---

### Entry 01: Initial Inspection & Threat Assessment
- **Objective:** Audit `REPORT_v0.md`, `fertility.py`, `corpus_sample/`, and `bench/` before leadership commits to infrastructure sizing and traffic routing.
- **Initial Hypothesis:** The previous intern's headline numbers (Hindi fertility is 5.89x worse than English; budget 6x serving cost; longer prompts increase GPU throughput; batch 48 delivers ~3200 tok/s) are either mathematically unsound or fundamentally misleading.
- **Observations:**
  1. `REPORT_v0.md` asserts Hindi fertility is 5.89x worse based solely on `gpt2` on 10 cherry-picked parallel lines in `corpus_sample/`.
  2. Claims "tok/char agrees: 1.579 vs 0.226 = 7.0x worse per character, which confirms the per-word number." This is circular reasoning: both metrics share the same numerator (tokens).
  3. Recommends assuming ~1600 tok/s per L4 and linearly scaling with batch size to batch 48 (~3200 tok/s). But `bench_log.csv` shows batch 48 was already benchmarked (row 14) and only yielded 1298.5 tok/s with 23 preemptions.

---

### Entry 02: Auditing `fertility.py` & Finding Flaws
- **Experiment:** Went through `fertility.py` line by line against the Evidence Rule.
- **Findings & Bug Isolation:**
  - **Flaw 1 (Code Bug):** `words = line.split(" ")` splits strictly on single spaces. On line 10 of `hin_sample.txt` (`किताबें  अलमारी में रखी हैं।`), consecutive spaces produce an empty string as a phantom word.
    - *Dead end I hit first:* I assumed a phantom word would inflate fertility. It doesn't — `words` is in the denominator, so an extra phantom word actually **deflates** the ratio. Fixing `split(" ")` to `split()` raised Hindi fertility from 7.45 to 7.60 (delta = +0.15).
  - **Flaw 2 (Code Bug):** `line = line.lower()` applied uniformly. Devanagari/Dravidian scripts are unicase, so lowercasing is a no-op there. For English it breaks acronyms like "NASA" into multi-token subwords under GPT-2 BPE, artificially inflating English's token count and shrinking the apparent English/Hindi gap.
  - **Flaw 3 (Statistical Flaw):** Macro-averaging (`sum(per_line_fertility)/n`) averages ratios across lines of very different lengths, which skews the aggregate. Micro-averaging (sum(tokens)/sum(words)) is the correct approach.
  - **Flaw 4 (The Core Conceptual Flaw):** `tok/word` assumes "word" is a linguistically invariant unit across languages. It isn't — Dravidian (Kannada, Tamil, Telugu) and Indo-Aryan (Hindi, Marathi) languages are agglutinative/inflected, so one whitespace-delimited word can carry the meaning of 2-3 English words. This is the actual root cause of most of the apparent fertility gap.
  - **The Red Herring:** `random.seed(1337)` looked like leftover/incomplete code at first glance, but `random` is never called anywhere in the script. Verified with an AST trace — removing the line changes nothing (delta = 0.000). Classified as harmless dead code, not a bug.

---

### Entry 03: Constructing the Eval Corpus (FLORES-200) & Dead Ends
- **Dead End 1:** Tried `facebook/flores` via HF `datasets` — gated dataset, needs an auth token; also hit a `trust_remote_code` deprecation error.
- **Dead End 2:** Tried `Muennighoff/flores200` — uses a custom `.py` loading script, which modern `datasets` versions refuse to run (`RuntimeError: Dataset scripts are no longer supported`).
- **Dead End 3 (Windows console encoding):** Printing a "→" arrow in status messages threw `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'` in the PowerShell terminal. Fixed with `$env:PYTHONIOENCODING="utf-8"`.
- **Resolution:** Switched to `openlanguagedata/flores_plus` (ungated, Parquet format). Downloaded 1,012 parallel sentences per language for English, Hindi, Kannada, Tamil, Telugu, Bengali, Marathi.

---

### Entry 03a: Pre-Defense Self-Review — Caught a Stale Console Print
- **What happened:** Before the defense, I had Claude (chat) do a second-pass audit of the finished submission. It flagged that the closing `print()` block in `corrected_analysis.py`'s `main()` still had hardcoded placeholder numbers from an early Antigravity draft — written before the real 1,012-sentence corpus existed. It claimed GPT-2 tok/byte was ~0.17 (eng) / ~0.50 (hin), and that XLM-R still made Hindi worse than English (~1.2x). That directly contradicted the script's own saved `results/corrected_fertility.csv` (eng=0.2047/hin=0.5947, 2.90x for GPT-2; eng=0.2321/hin=0.1133, 0.49x for XLM-R — Hindi *cheaper*, not worse) and my own `memo.md`.
- **Why it mattered:** This is exactly the kind of self-contradiction the Evidence Rule is meant to catch. If the interviewer had run the script live during the defense, the terminal would have printed numbers that flatly disagreed with my memo.
- **Fix:** Rewrote the closing block to compute the ratios from the `all_results` dict populated during the actual run instead of a fixed string.
- **Independent verification:** Claude could propose the fix and check its arithmetic against my existing CSV, but it couldn't execute `tiktoken`/`transformers` itself (no access to the OpenAI vocab endpoint or the HF Hub from its environment). I ran the fixed script myself, twice, in two separate environments — once inside Antigravity, once in a clean `venv` in VS Code — and confirmed both runs print:
  - GPT-2: eng 0.2047, hin 0.5947 tok/byte -> 2.90x
  - XLM-R: eng 0.2321, hin 0.1133 tok/byte -> 0.49x
  Matches `memo.md` exactly. Also re-checked `results/corrected_fertility.csv` — all `tok_per_byte` values identical across both runs. (Grapheme totals shifted by a handful of counts, e.g. Kannada 90,371 vs 90,899 — almost certainly a `grapheme` package version difference between the two environments; doesn't affect the `tok/byte` headline number the memo relies on.)
- **Lesson:** The evidence rule applies to console output too, not just the written memo. A script that prints one thing while the report says another is a fabrication risk even if no individual number was invented. Also: an AI reviewer can catch an inconsistency and propose a fix, but "the AI says it's fixed" and "I ran it and confirmed it's fixed" are different claims — only the second one belongs in a defense.

---

### Entry 04: Serving Capacity & Bench Log Forensics (Part B)
- **Arithmetic Verification:**
  - Model: 4.2B params, 28 layers, 8 KV heads, head_dim=128, fp16 (2 bytes).
  - KV bytes/token: 2 x 2 x 8 x 128 x 28 = 114,688 bytes = 112 KiB/token.
  - Usable VRAM on 24GB L4: 24GB x 0.92 - 1.6GB = 20.48GB.
  - Model weights: 4.2B x 2 bytes = 8.4GB.
  - KV pool: 20.48 - 8.4 = 12.08GB.
  - Block granularity: PagedAttention uses 16 tokens/block (1.75MiB/block). Total blocks = floor(12.08GB / 1.75MiB) = 6,582 blocks = 105,312 tokens.
  - Concurrent 4096-token sequences: 105,312 / 4096 = 25.7 -> 25-26 sequences.
- **Anomaly Detection (The Preemption Cliff):**
  - Prompt 3584 + gen 512 = 4096 tokens/sequence.
  - Batch 24: 24 x 4096 = 98,304 tokens, kv_cache_util = 0.93, 0 preemptions.
  - Batch 32: 32 x 4096 = 131,072 tokens (exceeds the ~105k pool). kv_cache_util caps at 0.97, 7 sequences preempted, throughput drops from 1607.4 to 1384.0 tok/s.
  - Batch 48: 23 preemptions, throughput drops further to 1298.5 tok/s.
- **The "Goodput" Deception:**
  - `reported_tok_s` includes prefill tokens (3584/4096 = 87.5% of total at this prompt length).
  - Real generation goodput at batch 24, two independent methods:
    - Wall clock: (24 x 512) / 61.16s = 200.9 tok/s.
    - Proportional: 1607.4 x (512/4096) = 200.9 tok/s.
  - The report claimed long prompts give better throughput (1311 vs 883 tok/s at batch 16). Actual decode-only goodput at batch 16: 164 tok/s for long prompts vs 294 tok/s for short prompts — the opposite conclusion.

---

### Entry 05: Decision Memo Formulation (Part C)
- **Constraints:** 1x A100-80GB for 2 weeks, 1 native reviewer (Hindi + Kannada only, 10h/week = 20h total), 3-week launch window, $0 external API budget.
- **Reviewer Capacity Reality Check:**
  - Reviewer covers 20h x ~30 samples/h = ~600 reviewed samples.
  - Path (a) SFT needs ~30,000 pairs across 6 languages. Reviewer can only check 600/30,000 = 2% of it. 98% of Tamil/Telugu/Bengali/Marathi pairs would ship unvetted.
  - Path (b) 1B rewriter has the same data-review bottleneck plus adds 150-300ms latency per request.
  - Path (c) prompt engineering gets results Day 1, costs $0, and uses 100% of the reviewer's time on meaningful validation instead of bulk data review.
- **Kill switch:** If Day 5 reviewer score is below 3.0/5.0, pivot to a narrow Hindi+Kannada-only SFT (3,000 pairs, fully reviewable in 20h) and ship the other 4 languages in beta with a monitoring commitment.