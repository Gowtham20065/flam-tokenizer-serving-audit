# Live Defense Cheat Sheet & Counterfactual Playbook

> **Timebox:** 30 minutes, screen shared, terminal and repo open.  
> **Rule:** Fabricated evidence is an automatic fail. Be ready to re-derive numbers live, modify code on the fly, paste inputs, and answer counterfactuals.

---

## 1. Quick Terminal Commands (Keep These Ready to Copy-Paste)

### Run the Bug Audit (Evidence Rule & Reproduction)
```powershell
$env:PYTHONIOENCODING="utf-8"
python partA/audit_fertility.py
```

### Run the Multi-Lingual FLORES Benchmark (7 Languages)
```powershell
$env:PYTHONIOENCODING="utf-8"
python partA/corrected_analysis.py
```

---

## 2. Live Defense "Re-Derive This Number" Cheat Sheet

### Q1: "Re-derive the 114,688 bytes (112 KiB) KV cache per token."
**Your live blackboard derivation:**
```
bytes_per_token = 2 (Key + Value)
                × 2 (fp16 precision = 2 bytes)
                × 8 (GQA KV heads)
                × 128 (head_dim)
                × 28 (layers)

Arithmetic:
= 4 × 8 × 128 × 28
= 32 × 128 × 28
= 4,096 × 28
= 114,688 bytes

In KiB:
114,688 / 1024 = 112 KiB exactly.
```
*Interviewer counterfactual:* **"What if the model used standard MHA instead of GQA?"**
- *Answer:* "In MHA, KV heads equal Q heads (24). The multiplier becomes $24 / 8 = 3\times$ larger: $114,688 \times 3 = 344,064\text{ bytes} \approx 336\text{ KiB/token}$. Our sequence capacity would drop from 26 down to ~8 concurrent sequences."

---

### Q2: "Re-derive the ~26 concurrent 4096-token sequences capacity."
**Your live blackboard derivation:**
```
1. Usable VRAM: 24 GB × 0.92 = 22.08 GB
2. Model weights (fp16): 4.2B params × 2 bytes = 8.40 GB
3. Non-KV overhead: 1.60 GB (activations, CUDA graphs)
4. Available for KV cache: 22.08 - 8.40 - 1.60 = 12.08 GB
5. Tokens in pool: 12.08 × 10⁹ bytes / 114,688 bytes/token ≈ 105,329 tokens
6. Sequences at 4096 tokens: 105,329 / 4096 = 25.7 ≈ 25–26 sequences.
```
*Interviewer check against log:* **"Where does this show up in bench_log.csv?"**
- *Answer:* "At batch 24 with prompt 3584 + gen 512 = 4096 tokens, `kv_cache_util` is **0.93** (98,304 tokens / 0.93 = ~105,700 token pool). At batch 32, demand is 131,072 tokens, which exceeds the ~105,700 capacity, immediately triggering **7 preempted sequences** in column `preempted_seqs`."

---

### Q3: "Re-derive the honest goodput of 200.9 tok/s for batch 24 long-prompt."
**Your live blackboard derivation (show both independent methods):**

- **Method 1 (Wall-clock generation):**
  $$\text{Goodput} = \frac{\text{num\_requests} \times \text{gen\_len}}{\text{wall\_clock\_s}} = \frac{24 \times 512}{61.16\text{ s}} = \frac{12,288}{61.16} = \mathbf{200.92\text{ tok/s}}$$

- **Method 2 (Proportional decomposition of `reported_tok_s`):**
  $$\text{Goodput} = \text{reported\_tok\_s} \times \frac{\text{gen\_len}}{\text{prompt\_len} + \text{gen\_len}} = 1607.4 \times \frac{512}{3584 + 512} = 1607.4 \times \frac{512}{4096} = 1607.4 \times 0.125 = \mathbf{200.925\text{ tok/s}}$$

*Interviewer question:* **"Why did the report claim 1607 tok/s?"**
- *Answer:* "Because `reported_tok_s` counted prefill prompt tokens (3584 tokens/req = 87.5% of the total). Prefill is parallel matrix multiplication; decode is sequential memory-bandwidth bound. Conflating them confuses prefill speed with generated output."

---

## 3. Live Code Modifications & "Run With This Input"

### If they ask: "Add an arbitrary input string live and print its tokens across tokenizers"
Open python interactively or create a 3-line scratch script:
```python
import tiktoken
from transformers import AutoTokenizer

text = "आप कैसे हैं?"  # Or whatever they paste
gpt2 = tiktoken.get_encoding("gpt2")
xlmr = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

g_toks = gpt2.encode(text)
x_toks = xlmr.encode(text, add_special_tokens=False)

print(f"Text bytes: {len(text.encode('utf-8'))}")
print(f"GPT-2 tokens: {len(g_toks)} -> {len(g_toks)/len(text.encode('utf-8')):.3f} tok/byte")
print(f"XLM-R tokens: {len(x_toks)} -> {len(x_toks)/len(text.encode('utf-8')):.3f} tok/byte")
```

---

### If they ask: "Why did fixing `line.split(' ')` INCREASE Hindi fertility from 7.45 to 7.60?"
- **The exact explanation:**
  "In line 10 of `hin_sample.txt`:
  `किताबें  अलमारी में रखी हैं।`
  There is a double space between `किताबें` and `अलमारी`.  
  `line.split(' ')` returned `['किताबें', '', 'अलमारी', 'में', 'रखी', 'हैं।']` (length 6 instead of 5).  
  Because `words` is in the denominator ($\text{tokens} / \text{words}$), adding an empty string increased the denominator, which **deflated** that line's fertility ratio from 4.80 down to 4.00.  
  When we fixed it to `line.split()` (which collapses whitespace), the denominator dropped from 6 to 5, so the fertility ratio for that line went back up to 4.80, pulling the Hindi average up from **7.45 to 7.60**."

---

### If they ask: "Why didn't you flag `random.seed(1337)` as a bug?"
- **The exact defense:**
  "Under the Evidence Rule, every bug must produce a measurable distortion. `random.seed(1337)` is defined, but `random` is never called anywhere in `fertility.py`. It is dead code, but it produces $\Delta = 0.0000$. Flagging harmless code as a bug without evidence costs $-5$ points under the rubric."

---

### If they ask: "Why tok/byte instead of tok/word?"
- **The exact defense:**
  "A 'word' is linguistically non-invariant. Kannada and Tamil are agglutinative languages: case markers, tense inflections, and postpositions fuse onto a single word. In our parallel FLORES corpus, the exact same meaning takes 21,901 words in English but only 16,100 words in Kannada. Denominating by word punishes agglutination. UTF-8 bytes are hardware-neutral: network requests, memory footprints, and compute costs scale directly with byte payload."

---

## 4. Part C Defense: "Why not SFT?"

### If they ask: "Why shouldn't we do a LoRA SFT pass for all 6 languages?"
- **Your response:**
  "Because of the reviewer constraint. We have **one native speaker who only knows Hindi + Kannada for 10 hours/week for 2 weeks = 20 hours total**. At 30 samples/hour, they can review at most **600 samples**.
  An SFT pass across 6 languages needs at least $5,000 \times 6 = 30,000\text{ pairs}$. We would only inspect $600 / 30,000 = \mathbf{2\%}$ of the dataset.
  Furthermore, with \$0 external API budget, the model would generate its own synthetic data. We would ship completely unverified synthetic models in Tamil, Telugu, Bengali, and Marathi with zero native-speaker sign-off.
  Prompt engineering gives us Day 1 iteration, costs \$0, and allows 100% of the reviewer's 20 hours to validate high-leverage few-shot exemplars and negative constraints. And we have a Day 5 kill criterion to fall back to a narrow 2-language SFT if human scores $< 3.0/5.0$."
