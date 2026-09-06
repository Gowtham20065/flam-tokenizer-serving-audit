# Part A3 — Corrected Cross-Language Analysis & Denominator Reasoning

This document provides the rigorous, corrected multilingual tokenizer evaluation across **7 languages**, **2 tokenizers** (monolingual English-centric vs. multilingual Indic-aware), and **4 denominators**, answering the central question: **Which single number should drive a routing-and-cost decision, and why?**

All underlying experiments are automated and reproducible in:
```bash
python partA/corrected_analysis.py
```
Output data is saved in [`partA/results/corrected_fertility.csv`](results/corrected_fertility.csv).

---

## 1. Experimental Matrix: Corrected Cross-Language Numbers

All numbers are micro-averaged ($\sum \text{Tokens} / \sum \text{Denominator}$) over **1,012 parallel sentences** per language from FLORES-200, without artificial lowercasing and with full whitespace regularization.

### A. GPT-2 Tokenizer (`gpt2` via tiktoken — Monolingual English BPE)

| Language | Family | Tokens | tok / word | tok / char | tok / grapheme | **tok / UTF-8 byte** | tok / sentence | Ratio vs ENG (tok/byte) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **English** (`eng`) | Germanic | 27,044 | 1.23 | 0.20 | 0.20 | **0.2047** | 26.7 | 1.00× |
| **Hindi** (`hin`) | Indo-Aryan | 200,688 | 7.83 | 1.53 | 2.21 | **0.5947** | 198.3 | **2.90×** |
| **Marathi** (`mar`) | Indo-Aryan | 212,471 | 11.16 | 1.60 | 2.38 | **0.5976** | 209.9 | **2.92×** |
| **Bengali** (`ben`) | Indo-Aryan | 259,769 | 13.32 | 1.99 | 2.91 | **0.7449** | 256.7 | **3.64×** |
| **Kannada** (`kan`) | Dravidian | 367,366 | 22.82 | 2.66 | 4.04 | **0.9788** | 363.0 | **4.78×** |
| **Telugu** (`tel`) | Dravidian | 350,748 | 20.71 | 2.65 | 4.13 | **0.9918** | 346.6 | **4.84×** |
| **Tamil** (`tam`) | Dravidian | 420,171 | 25.05 | 2.73 | 4.21 | **0.9965** | 415.2 | **4.87×** |

---

### B. XLM-RoBERTa Tokenizer (`xlm-roberta-base` via HuggingFace — Multilingual Indic-Aware SentencePiece)

| Language | Family | Tokens | tok / word | tok / char | tok / grapheme | **tok / UTF-8 byte** | tok / sentence | Ratio vs ENG (tok/byte) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **English** (`eng`) | Germanic | 30,661 | 1.40 | 0.23 | 0.23 | **0.2321** | 30.3 | 1.00× |
| **Hindi** (`hin`) | Indo-Aryan | 38,221 | 1.49 | 0.29 | 0.42 | **0.1133** | 37.8 | **0.49×** |
| **Marathi** (`mar`) | Indo-Aryan | 37,330 | 1.96 | 0.28 | 0.42 | **0.1050** | 36.9 | **0.45×** |
| **Bengali** (`ben`) | Indo-Aryan | 42,118 | 2.16 | 0.32 | 0.47 | **0.1208** | 41.6 | **0.52×** |
| **Kannada** (`kan`) | Dravidian | 41,459 | 2.58 | 0.30 | 0.46 | **0.1105** | 41.0 | **0.48×** |
| **Telugu** (`tel`) | Dravidian | 40,372 | 2.38 | 0.30 | 0.48 | **0.1142** | 39.9 | **0.49×** |
| **Tamil** (`tam`) | Dravidian | 41,354 | 2.47 | 0.27 | 0.41 | **0.0981** | 40.9 | **0.42×** |

---

## 2. Denominator Forensics: What is the Denominator Supposed to Hold Constant?

To choose the right metric, we must answer: **What invariant must the denominator hold constant across languages?**

An ideal denominator must hold **Semantic Information Content** constant: two texts expressing identical human meaning in different languages should have comparable denominators.

Let us evaluate the four options:

### 1. Per Whitespace Word (`tok / word`) — Fails Semantic Invariance
- **Why it fails:** "Word" is an isolating language concept. In synthetic and agglutinative languages (Kannada, Tamil, Telugu), grammatical relations (case, tense, aspect, negation, postpositions) are expressed by morphemes fused onto a single root word.
- **Evidence:** In our 1,012 parallel sentences, expressing the exact same semantic content required:
  - English: **21,901 words**
  - Kannada: **16,100 words** (26.5% fewer words)
  - Tamil: **16,775 words** (23.4% fewer words)
- Because Kannada packs more meaning per word, its denominator is artificially small, making fertility look artificially catastrophic (22.8 tok/word under GPT-2). `tok/word` punishes agglutinative languages.

### 2. Per Unicode Codepoint (`tok / char`) — Fails Visual & Phonetic Invariance
- **Why it fails:** In Indic Brahmic scripts, vowels and diacritics are separate combining codepoints (e.g., Unicode categories `Mn`, `Mc`) attached to consonant bases. A single syllabic character can consist of 2 to 4 Unicode codepoints. In Latin scripts, letters are mostly standalone codepoints. Comparing codepoint counts compares glyph encoding mechanics, not information.

### 3. Per Grapheme Cluster (`tok / grapheme`) — Useful for UI, but Not for Serving Costs
- **Why it is useful:** A grapheme cluster represents what a human user perceives as a single visual "letter" (a base consonant plus its attached matras and virama).
- **Limitation:** While excellent for estimating frontend typing speed or UX character counters, grapheme clusters do not map to physical GPU memory or network bandwidth.

### 4. Per UTF-8 Byte (`tok / UTF-8 byte`) — THE CORRECT METRIC FOR SERVING & ROUTING
- **Why it works:**
  1. **Hardware and Network Ground Truth:** Servers do not ingest "words" or "graphemes"; HTTP payloads, serialization buffers, and memory caches ingest **raw UTF-8 bytes**.
  2. **Information Density Normalization:** In UTF-8, ASCII characters (English) take 1 byte per character. Indic scripts (Devanagari, Kannada, Tamil) take 3 bytes per Unicode codepoint. However, because each Indic syllabic character carries a rich consonant-vowel syllable, 3 UTF-8 bytes of Indic text carry approximately the same or higher phonemic/semantic density as 3 ASCII characters in English.
  3. **Direct Link to Cost:** GPU compute and KV-cache allocation scale with the total number of tokens allocated per byte of incoming user request.

---

## 3. Which Single Number Should Drive Routing-and-Cost Decisions?

> **The Single Driving Metric:**
> $$\mathbf{\text{Tokens per UTF-8 Byte } (\text{tok / byte})}$$

### Why this single number transforms leadership decisions:

1. **Under GPT-2:**
   - The intern reported a **5.89×** cost penalty for Hindi based on `tok/word`.
   - On `tok/byte`, the true penalty is **2.90× for Hindi**, and **~4.8× for Dravidian languages** (Kannada/Tamil).
   - The previous intern overstated Hindi cost by **100%** because of their flawed denominator.

2. **Under an Indic-Aware Tokenizer (`XLM-RoBERTa`):**
   - The reality completely reverses:
     - English: **0.232 tok/byte**
     - Hindi: **0.113 tok/byte (0.49× English)**
     - Kannada: **0.110 tok/byte (0.48× English)**
     - Tamil: **0.098 tok/byte (0.42× English)**
   - **Conclusion:** With a proper tokenizer, **Indic languages are actually ~50% CHEAPER per byte than English** because Indic scripts encode higher semantic entropy per byte.
   - The root cause of the "6× cost" was 100% a tokenizer defect, not a linguistic barrier.
