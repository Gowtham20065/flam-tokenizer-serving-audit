# Flam AI Team Intern Assignment — The Audit & Decision Memo

This repository contains the complete audit, reproduction experiments, multi-lingual evaluation, capacity reconciliation calculations, and decision memo for the **Flam AI Team Intern Assignment**.

---

## 📂 Repository Structure

```
.
├── README.md                      # Setup and execution guide
├── NOTEBOOK.md                    # Chronological lab notebook (hypotheses, experiments, dead ends)
├── AI_USAGE.md                    # Transparent disclosure of AI assistance and human verification
├── partA/                         # Tokenizer audit, eval corpus, corrected benchmarks, & memo
│   ├── prepare_corpus.py          # Downloads FLORES-200 devtest for eng, hin, kan, tam
│   ├── audit_fertility.py         # Step-by-step bug isolation & delta measurement (Evidence Rule)
│   ├── corrected_analysis.py      # Multi-tokenizer & multi-denominator benchmark engine
│   ├── memo.md                    # ≤1 page executive recommendation memo
│   ├── corpus/                    # 1,012 parallel sentences per language (eng, hin, kan, tam)
│   │   ├── eng.txt
│   │   ├── hin.txt
│   │   ├── kan.txt
│   │   ├── tam.txt
│   │   └── meta.json
│   └── results/
│       └── corrected_fertility.csv # Benchmark numbers across all tokenizers & denominators
├── partB/                         # Serving capacity reconciliation
│   └── calculations.md            # B1 arithmetic, B2 preemption cliff, B3 goodput derivation, B4 metrics
├── partC/                         # Strategic decision memo
│   └── memo.md                    # ≤1 page decision memo for casual-tone generation in 6 Indic languages
└── starter_kit/                   # Original artifacts from previous intern (REPORT_v0, fertility.py, bench/)
```

---

## 🚀 Quickstart & Reproduction in VS Code

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Set Up Virtual Environment
Open the repository folder in VS Code or your terminal:

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Or on Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Running the Code

### Part A: Tokenizer Bug Isolation & Evidence Rule (`partA/audit_fertility.py`)
Reproduces the original `REPORT_v0.md` numbers and isolates each bug one by one to measure the exact direction and magnitude of the distortion:
```bash
python partA/audit_fertility.py
```

### Part A: Multi-Lingual Evaluation on FLORES-200 (`partA/corrected_analysis.py`)
Runs the full evaluation across GPT-2 and XLM-RoBERTa on 4 languages (English, Hindi, Kannada, Tamil) across 4 denominators (`tok/word`, `tok/char`, `tok/grapheme`, `tok/UTF-8 byte`):
```bash
python partA/corrected_analysis.py
```
*Results are automatically printed to the terminal and saved to `partA/results/corrected_fertility.csv`.*

### Re-downloading the Evaluation Corpus (Optional)
If you ever want to re-download the 1,012 FLORES-200 parallel sentences:
```bash
python partA/prepare_corpus.py
```

---

## 📊 Summary of Key Findings

1. **Part A — Tokenizer Audit:**
   - The reported 5.89× Hindi fertility was an artifact of using an English-centric BPE tokenizer (`gpt2`) on `tok/word`.
   - On the invariant metric **`tok/UTF-8 byte`**, GPT-2's overhead drops to **2.90×**.
   - Under an Indic-aware multilingual tokenizer (`XLM-RoBERTa`), Indic languages actually consume **fewer tokens per byte (~0.10–0.11 tok/byte) than English (0.23 tok/byte)** because of the high information density of Indic scripts.

2. **Part B — Serving Capacity Forensics:**
   - **KV-Cache:** $114,688\text{ bytes} \approx 112\text{ KiB/token}$.
   - **Max 4096-token sequences:** Physical ceiling is **26 sequences** due to VRAM and PagedAttention block limits (matches the 0.93 utilization at batch 24 in `bench_log.csv`).
   - **Goodput:** The intern reported 1607 tok/s by mistakenly including prefill tokens. The true generation goodput is **$200.9\text{ tok/s}$** (8× lower).

3. **Part C — Decision Memo:**
   - Recommended **Prompt Engineering** with few-shot exemplars and strict negative constraints. Reviewer capacity (20 hours = 600 pairs) cannot validate a 30k SFT dataset across 6 languages. Fast kill switch on Day 5 if human evaluation score $< 3.0/5.0$.

---

---

## Rubric Coverage (self-assessment, not a claimed score)

| Component | File / Artifact | What's there |
|---|---|---|
| A1 Corpus construction & caveats | `partA/A1_corpus_documentation.md`, `partA/prepare_corpus.py` | 7-language corpus (1,012 sents each). Documented Wikipedia/news domain, NFC normalization, and explicit caveats on colloquial Hinglish vs formal prose. |
| A2 Script/metric audit (Evidence Rule) | `partA/A2_script_audit.md`, `partA/audit_fertility.py` | Isolated 2 code bugs + 1 statistical flaw with exact delta, the conceptual `tok/word` flaw, and the `random.seed(1337)` red herring defended with delta=0. |
| A3 Corrected analysis & denominator | `partA/A3_corrected_analysis.md`, `partA/corrected_analysis.py` | 7 languages x 2 tokenizers (GPT-2, XLM-R) x 4 denominators (`tok/word`, `tok/char`, `tok/grapheme`, `tok/byte`). |
| A4 Recommendation memo | `partA/memo.md` | <=1 page: corrected headline table, routing recommendation, biggest caveat, production monitoring counter. |
| B1-B4 Capacity reconciliation | `partB/calculations.md` | KV bytes/token, sequence ceiling validated against the log, preemption-cliff mechanism, goodput derived two independent ways, monitoring metric. |
| C Decision memo | `partC/memo.md` | <=1 page: assumptions, arithmetic, numeric success threshold, dated kill criterion, Day-1 experiment. |

**Known limitation:** Part A's token counts depend on `tiktoken`/`transformers` fetching `gpt2` and `xlm-roberta-base` at run time (needs network access to OpenAI's blob store and the HF Hub). If run offline, only Part B and the pre-computed `results/corrected_fertility.csv` will be available.



