# Part A4 — Recommendation Memo: Tokenizer Selection & Routing Strategy

**To:** Engineering Leadership & Product Architecture  
**From:** Software Engineering Intern (Auditing Team)  
**Date:** September 2026  
**Subject:** Corrected Tokenizer Fertility Audit, Multi-Lingual Evaluation, and Cost Routing Decisions  

---

## 1. Corrected Headline Numbers

The previous report (`REPORT_v0.md`) claimed Hindi has **5.89× worse** fertility than English under GPT-2 and asserted that leadership must budget **6× serving cost** because "Hindi simply has more Unicode characters per word."

Our audit re-evaluated this on **FLORES-200** (`openlanguagedata/flores_plus` devtest: 1,012 parallel sentences across English, Hindi, Kannada, and Tamil) with bug fixes applied (no artificial lowercasing, whitespace regex splitting, micro-averaging) across two tokenizers and four denominators.

### Experimental Results Summary (from `results/corrected_fertility.csv`)

| Tokenizer | Language | Family | tok / word | tok / char | tok / grapheme | **tok / UTF-8 byte** | Ratio vs ENG (tok/byte) |
|---|---|---|---:|---:|---:|---:|---:|
| **GPT-2** (tiktoken) | English (`eng`) | Germanic | 1.23 | 0.20 | 0.20 | **0.20** | 1.00× |
| | Hindi (`hin`) | Indo-Aryan | 7.83 | 1.53 | 2.21 | **0.59** | **2.90×** |
| | Marathi (`mar`) | Indo-Aryan | 11.16 | 1.60 | 2.38 | **0.60** | **2.92×** |
| | Bengali (`ben`) | Indo-Aryan | 13.32 | 1.99 | 2.91 | **0.74** | **3.64×** |
| | Kannada (`kan`) | Dravidian | 22.82 | 2.66 | 4.04 | **0.98** | **4.78×** |
| | Telugu (`tel`) | Dravidian | 20.71 | 2.65 | 4.13 | **0.99** | **4.84×** |
| | Tamil (`tam`) | Dravidian | 25.05 | 2.73 | 4.21 | **1.00** | **4.87×** |
| **XLM-RoBERTa** (`xlm-roberta-base`) | English (`eng`) | Germanic | 1.40 | 0.23 | 0.23 | **0.23** | 1.00× |
| | Hindi (`hin`) | Indo-Aryan | 1.49 | 0.29 | 0.42 | **0.11** | **0.49×** |
| | Marathi (`mar`) | Indo-Aryan | 1.96 | 0.28 | 0.42 | **0.10** | **0.45×** |
| | Bengali (`ben`) | Indo-Aryan | 2.16 | 0.32 | 0.47 | **0.12** | **0.52×** |
| | Kannada (`kan`) | Dravidian | 2.58 | 0.30 | 0.46 | **0.11** | **0.48×** |
| | Telugu (`tel`) | Dravidian | 2.38 | 0.30 | 0.48 | **0.11** | **0.49×** |
| | Tamil (`tam`) | Dravidian | 2.47 | 0.27 | 0.41 | **0.10** | **0.42×** |

---

## 2. Which Single Metric Should Drive Routing and Cost Decisions?

**The single metric is `tokens per UTF-8 byte` (tok/byte).**

- **Why `tok/word` fails:** "Word" is fundamentally linguistically non-invariant across languages. In agglutinative languages like Kannada and Tamil, bound morphemes, case markers, and postpositions combine into a single compound token (e.g., Kannada has 16,100 whitespace words vs English's 21,901 words for the exact same semantic meaning). A word-level denominator punishes agglutination and rewards isolating morphology.
- **Why `tok/char` fails:** A Devanagari or Dravidian base glyph with matras and viramas spans multiple Unicode codepoints, distorting raw character counts.
- **Why `tok/UTF-8 byte` wins:** All network requests, KV cache consumption, and cloud network ingress/egress are bounded by bytes. UTF-8 byte representation is an invariant, hardware-neutral ground truth for the raw information density transmitted over the wire.
- **Key takeaway:** Under `tok/byte`, GPT-2 Hindi overhead drops from 6× down to **2.90×**. Under a multilingual tokenizer (`XLM-RoBERTa`), Hindi, Kannada, and Tamil actually consume **fewer tokens per byte (~0.10 - 0.11 tok/byte) than English (0.23 tok/byte)** because Indic scripts pack high semantic density per 3-byte UTF-8 sequence.

---

## 3. Routing and Architecture Recommendation

1. **Do NOT budget 6× serving cost for Hindi.** The 6× factor was a measurement artifact of using an English-only BPE vocabulary on whitespace words.
2. **Deploy an Indic-Aware Tokenizer/Model for Production:**
   - Migrate serving to an open multilingual tokenizer (e.g., XLM-R, Llama-3/Gemma multilingual vocabs, or IndicBERT). As demonstrated above, an Indic-aware vocabulary completely resolves the token explosion for Dravidian and Indo-Aryan languages (reducing Kannada tokens from 367k down to 41k — an **8.8× reduction in compute & KV allocation**).
3. **If Stuck on GPT-2-Class Vocabs Temporarily:**
   - Budget a **2.9× multiplier for Hindi** and a **4.8× multiplier for Kannada/Tamil** per input byte, NOT 6× across the board.

---

## 4. The Biggest Caveat

FLORES-200 represents formal, edited prose (Wikipedia and news domain). Production consumer traffic at Flam consists of **casual, colloquial, and code-mixed (Hinglish, Tanglish, Kanglish) queries**. In code-mixed queries, Latin-transliterated Indic words behave differently than native script: Latin transliteration avoids the byte-fallback penalty of GPT-2, but causes subword fragmentation on multilingual models. Real production fertility will sit between the native script benchmark and the Latin script benchmark.

---

## 5. Production Counter to Catch This Analysis Being Wrong

**Monitor:** `vllm:avg_prompt_throughput_tok_per_byte` segmented by `request_language_header`.

Specifically, compute the real-time ratio:
$$\text{Fertility}_{\text{prod}} = \frac{\sum \text{Prompt Tokens Allocated}}{\sum \text{Request Ingress Payload (UTF-8 Bytes)}}$$

If the production ratio for Indic traffic diverges by $>15\%$ from our calibrated $0.11\text{ tok/byte}$ (on multilingual) or $0.59\text{ tok/byte}$ (on GPT-2 fallback) over a rolling 1-hour P95 window, trigger an automated routing alarm to recalibrate KV-cache autoscaling headroom.
