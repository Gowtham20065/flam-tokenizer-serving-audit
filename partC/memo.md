# Part C — Decision Memo: Casual-Tone Generation in 6 Indic Languages

**To:** Product Team / Engineering Leadership
**From:** Gowtham
**Date:** September 2026
**Re:** Recommendation for casual-tone multilingual generation (Hindi, Kannada, Tamil, Telugu, Bengali, Marathi)

---

## Recommendation: Path (c) Prompt Engineering first, with a fast kill into a focused SFT pass

**Recommended path: Prompt engineering for week 1–2, narrow SFT pass for week 3 if needed.**

---

## Assumptions

1. "Casual" is a register shift, not a capability gap — the model already knows Hindi/Kannada etc.; it was fine-tuned on formal instruction data, so the prior can be steered without retraining.
2. Our one native speaker covers Hindi + Kannada only. We cannot measure quality in Tamil, Telugu, Bengali, or Marathi — any path that requires human eval on those languages is blocked.
3. The A100-80GB fits a 7–13B model comfortably in fp16 for inference and in LoRA-SFT mode for training.
4. 3-week wall clock; launch review is a hard deadline.
5. No external API budget means no GPT-4 for synthetic data generation via API — we use the hosted model to generate its own casual rewrites.

---

## Back-of-envelope arithmetic

### Path (a) — SFT pass

| Item | Estimate |
|---|---|
| Synthetic data needed | ~5,000 pairs/language × 6 = 30,000 pairs |
| Generation rate (7B model, batch 32, A100) | ~800 tok/s; avg 200 tok/pair → ~30,000 × 200 / 800 = **7,500 s ≈ 2.1 h** |
| LoRA SFT (r=16, 7B, 30k pairs, 3 epochs) | ~6–8 h on one A100 |
| Reviewer throughput (Hindi+Kannada only) | 10 h/week × 2 weeks = 20 h; ~30 samples/h → **600 reviewed samples** |
| Coverage | 600 / 30,000 = **2% reviewed** — unacceptably low |
| Time left after data gen + training | <1 week for eval + deployment |

**Problem**: 30k pairs is too large to adequately review in 20 reviewer-hours. We'd ship unvalidated casual Hindi/Kannada + entirely unreviewed Tamil/Telugu/Bengali/Marathi. **Risk is unacceptable for a launch.**

### Path (b) — Small rewriter model (≤1B)

| Item | Estimate |
|---|---|
| Train a 1B rewriter | Need ~5k reviewed casual pairs to avoid hallucination; same reviewer bottleneck |
| Serving overhead | Doubles inference cost (2 forward passes per user request) |
| Latency | +150–300 ms per reply; likely unacceptable for conversational UX |
| Benefit over SFT | Swappable without retraining base — but we don't have the data to train it well |

**Problem**: The reviewer bottleneck bites equally here, and we add latency. Eliminates itself.

### Path (c) — Prompt engineering

| Item | Estimate |
|---|---|
| Iteration time | Write → test → revise in <1 h per language |
| Reviewer time needed | 10 samples/language × 6 = 60 samples total; ~2 h reviewer time |
| Coverage | 100% of reviewer budget used on meaningful eval |
| Cost | \$0 inference budget (uses the same model already running) |
| Time to first result | **Day 1** |
| Deployment risk | Zero — rollback is deleting 3 lines from the system prompt |

---

## Chosen path: Prompt engineering

### Success metric (with numeric threshold)

**Human eval score ≥ 3.5 / 5.0** on a casual-register rubric (1 = very formal, 5 = native-casual) averaged over 30 Hindi + 30 Kannada responses reviewed by the native speaker. Secondary: automated style classifier (trained on 200 labelled examples) scoring ≥ 65% "casual" on all 6 languages.

### Kill criterion

**By end of Day 5**: If the native speaker scores Hindi + Kannada below **3.0/5.0** across 3 independent prompt variants (different few-shot examples, different persona framing, different instruction phrasing), prompt engineering cannot deliver the register shift — the formality is baked into the RLHF layer and cannot be overridden by instruction. At that point, pivot to a narrow **Hindi + Kannada-only SFT** (2 languages, ~3,000 pairs, fully reviewed by the native speaker within 20 h at 30 samples/h) and ship the remaining 4 languages with the best prompt-engineered version, accepting lower quality with a monitoring commitment.

### First experiment — Day 1

Run 3 prompt variants on 20 fixed test prompts in Hindi:
1. **Persona framing**: `"You are a friendly college student in Mumbai. Reply in casual, colloquial Hindi. Use common Hindi slang and contractions."`
2. **Few-shot exemplars**: 3 formal→casual example pairs in the system prompt (hand-authored by the reviewer in 30 min).
3. **Negative instruction**: `"Do not use formal or textbook language. Never use passive voice. Prefer short sentences."`

### Days 2–5: the other 4 languages

Tamil, Telugu, Bengali, and Marathi are not part of the kill-criterion decision — there is no reviewer to evaluate them regardless of how Hindi/Kannada perform. From Day 1 onward they ship under the same prompt-engineering approach (personas adapted from the Hindi/Kannada winners), gated only by the automated classifier, under an explicit "beta" label. This does not change based on the Day 5 outcome: even if Hindi/Kannada pivot to SFT, the other 4 stay on prompt engineering, since Path (a) was already ruled out for them by reviewer bandwidth. Their only feedback loop until a native reviewer becomes available is the classifier score and in-product user reports.

Score all three with the reviewer (1 h of their 10 h/week). Pick the best. Extend to Kannada. Adapt persona for other 4 languages without reviewer sign-off but with automated classifier gating.

---

## Biggest caveat

We have **zero native-speaker coverage for 4 of 6 languages**. The automated classifier is a weak proxy. If Tamil/Telugu/Bengali/Marathi outputs are casual in surface form but culturally inappropriate or grammatically degraded, we will not catch it before launch. This is the dominant risk of all three paths — not a prompt engineering-specific failure. The mitigation is to agree with the product team that the 4 un-reviewed languages ship as "beta" with an explicit in-product label and a fast-feedback loop.
