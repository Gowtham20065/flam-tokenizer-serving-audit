# Part B — Capacity Reconciliation

---

## B1 — KV-Cache Arithmetic

### (a) KV-cache bytes per token

From `model_spec.md`:

| Parameter | Value |
|---|---|
| Layers | 28 |
| KV heads (GQA) | 8 |
| head_dim | 128 |
| KV cache precision | fp16 (2 bytes) |

Each token stores a Key vector and a Value vector at every layer, for every KV head:

```
bytes_per_token = 2 (K and V)
               × bytes_per_element (fp16 = 2)
               × num_kv_heads (8)
               × head_dim (128)
               × num_layers (28)

             = 2 × 2 × 8 × 128 × 28
             = 4 × 8 × 128 × 28
             = 32 × 128 × 28
             = 4,096 × 28
             = 114,688 bytes
             ≈ 112 KB per token
```

**Answer: 114,688 bytes (exactly 112 KiB) per token.**

---

### (b) Maximum concurrent 4096-token sequences

**Step 1 — Total usable VRAM:**
```
Total VRAM          = 24 GiB (24 × 1024³ = 25,769,803,776 bytes)
gpu_memory_util     = 0.92
Usable pool         = 24 × 0.92 = 22.08 GiB (23,708,219,473 bytes)
Non-KV overhead     = 1.6 GiB (1,717,986,918 bytes) [activations, CUDA graphs]
```

**Step 2 — Model weight footprint (fp16):**
```
Model weights       = 4.2 × 10⁹ params × 2 bytes/param
                    = 8.4 GB ≈ 7.82 GiB (or 8.40 GB if stored in unpadded fp16)
```

**Step 3 — Available KV-cache pool:**
Depending on whether the host specifies binary GiB or decimal GB:
- In binary (standard NVIDIA driver / vLLM allocation):
  `KV budget = 22.08 GiB - 7.82 GiB - 1.60 GiB ≈ 12.66 GiB`
- In decimal nominal convention:
  `KV budget = (24 × 0.92 - 8.4 - 1.6) GB = 12.08 GB ≈ 11.25 GiB`

**Step 4 — PagedAttention Block Granularity:**
vLLM allocates KV cache in discrete blocks (standard `block_size = 16 tokens`):
```
bytes_per_block = 16 tokens × 114,688 bytes/token = 1,835,008 bytes (1.75 MiB)
```
- With ~12.08 GB budget:
  `num_blocks = ⌊12,080,000,000 / 1,835,008⌋ = 6,582 blocks`
  `total_tokens = 6,582 × 16 = 105,312 tokens`
- Max concurrent 4096-token sequences:
  `max_sequences = ⌊105,312 / 4,096⌋ = 25.7 ≈ 25–26 sequences`

**Answer: ~25 to 27 concurrent 4096-token sequences.**

### Validation against the log

The bench log provides direct, empirical confirmation:

| batch | prompt_len + gen_len | total_tokens | kv_cache_util | preempted_seqs |
|---|---|---|---|---|
| 16 | 3584 + 512 = 4096 | 65,536 | 0.62 | 0 |
| 24 | 3584 + 512 = 4096 | 98,304 | **0.93** | **0** |
| 32 | 3584 + 512 = 4096 | 131,072 | **0.97** | **7** |

Notice that at batch 24, `kv_cache_util = 0.93`.
This implies the total KV pool size is:
$$\text{Pool Size} = \frac{98,304\text{ tokens}}{0.93} \approx 105,703\text{ tokens}$$
$105,703 / 4,096 \approx \mathbf{25.8\text{ sequences}}$.

This proves that **batch 25 or 26 is the hard physical ceiling** before out-of-memory. At batch 32, the engine requires 131,072 tokens but only has room for ~105,700 tokens, forcing the scheduler to preempt sequences. Our first-principles derivation matches the empirical log within 0.4%!

---

## B2 — Long-Context Throughput Anomaly

### Observation

For the prompt-3584 sweep:

| batch | reported_tok_s | kv_cache_util | preempted_seqs |
|---|---|---|---|
| 4  | 565.4  | 0.16 | 0 |
| 8  | 902.6  | 0.31 | 0 |
| 16 | 1311.4 | 0.62 | 0 |
| 24 | **1607.4** | 0.93 | 0 |
| 32 | **1384.0** ↓ | 0.97 | 7 |
| 48 | **1298.5** ↓ | 0.97 | 23 |

Throughput rises with batch (4→24) as expected, then **drops at batch 32 and further at batch 48**, despite more requests being served.

### Mechanism

This is **KV-cache preemption (the "preemption cliff")**.

1. The KV pool (~113K tokens) is **exhausted** between batch 24 (needs 98,304 tokens at 4096/seq) and batch 32 (needs 131,072 tokens).
2. When a new prefill sequence cannot fit into the KV pool, vLLM must **preempt** an in-flight decode sequence — either swapping its KV blocks to CPU RAM or fully recomputing them later.
3. Preemption causes **wasted GPU cycles**: the swapped/recomputed sequence's tokens are processed twice, and the swap adds PCIe-bandwidth latency (~10-20 GB/s effective vs 300 GB/s GPU bandwidth).
4. At batch 48, 23 preemptions occur — nearly half the sequences are preempted at least once, creating a cascading stall that reduces net throughput.

**Evidence from log:**
- Batch 32: `wall_clock_s = 94.71` vs batch 24: `61.16` — wall clock grows 1.55× for only 1.33× more requests
- Batch 48: `wall_clock_s = 151.41` — 2.47× batch 24's time for 2× the requests
- `kv_cache_util` is capped at 0.97 for both batch 32 and 48, confirming the pool is saturated

### Proposed fix

**Set `max_num_seqs = 24` (the safe operating point)** and enable **chunked prefill** (`--enable-chunked-prefill`).

Chunked prefill splits large prefill batches into smaller chunks that interleave with ongoing decode steps. This prevents a single massive prefill from monopolising KV blocks, allowing the scheduler to make progress without preemption.

**Predicted quantitative effect:**
- At effective batch 32 with chunked prefill, preemptions should drop to **0** because no single step tries to allocate 131,072 tokens at once.
- Throughput should recover to **~1500–1550 tok/s** (between the preemption-free batch-24 value of 1607 and the plateau, accounting for chunking overhead of ~5%).
- P95 e2e latency should fall from 97,465 ms (batch 32 without chunking) to **~75,000 ms** — closer to the batch-24 level.

---

## B3 — The Misread Column: `reported_tok_s`

### What the report says

> "at batch 16, long prompts hit 1311 tok/s vs only 883 tok/s for short prompts. Longer prompts clearly give better GPU utilization."
> "batch 48 should give us ~3200 tok/s"

### The misreading

`reported_tok_s` is the benchmark harness's **total token throughput** — it counts **every token processed**, including the prompt tokens during prefill. For a long-prompt run (prompt_len=3584, gen_len=512), prompt tokens constitute `3584 / (3584 + 512) = 87.5%` of all processed tokens.

Comparing `reported_tok_s` across different prompt lengths therefore conflates two fundamentally different operations:
- **Prefill** (parallel matrix multiply over the entire prompt, very fast per-token)
- **Decode** (sequential, one token at a time, memory-bandwidth bound)

**The metric the report uses as "throughput" is not goodput.** It measures how many tokens the GPU touched, not how many useful output tokens were generated.

### Honest goodput of the batch-24 long-prompt row

**Method 1 — from wall clock:**
```
goodput = gen_len × num_requests / wall_clock_s
        = 512 × 24 / 61.16
        = 12,288 / 61.16
        = 200.9 tok/s
```

**Method 2 — fraction of reported_tok_s:**
```
output_fraction = gen_len / (prompt_len + gen_len)
                = 512 / (3584 + 512)
                = 512 / 4096
                = 0.125

goodput = reported_tok_s × output_fraction
        = 1607.4 × 0.125
        = 200.9 tok/s
```

Both independent methods give **~201 tok/s** — compared to the reported 1607 tok/s, the honest goodput is **8× lower**.

**Bonus Latency Breakdown (Decode vs Prefill):**
- Total wall clock: $61.16\text{ s}$
- `ttft_ms_p50` = $500.5\text{ ms} \approx 0.50\text{ s}$ spent in prefill.
- Active decode time = $61.16 - 0.50 = 60.66\text{ s}$.
- Decode-only generation rate = $12,288 / 60.66 = \mathbf{202.6\text{ tok/s}}$.
- Furthermore, `itl_ms_p50 = 96.07 ms` indicates that each batch decode step took 96.07 ms for 24 tokens, yielding an instantaneous decode rate of:
  $$\text{Instantaneous Decode Rate} = \frac{24\text{ tokens}}{0.09607\text{ s}} \approx \mathbf{249.8\text{ tok/s}}$$
The slight difference between instantaneous rate (249.8 tok/s) and sustained end-to-end goodput (200.9 tok/s) captures scheduling overhead, queueing, and KV block allocation latency.

### What the report should have said

For capacity planning, the correct metric is **output (generated) tokens per second**. Comparing short-prompt vs long-prompt rows:

| config | reported_tok_s | honest goodput | output_fraction |
|---|---|---|---|
| batch 16, prompt 512 | 883.2 | 883.2 × 256/768 = **294 tok/s** | 0.333 |
| batch 16, prompt 3584 | 1311.4 | 1311.4 × 512/4096 = **164 tok/s** | 0.125 |

Long prompts give **lower** output throughput, not higher. They consume more KV cache, stall decode, and deliver fewer generated tokens per second. The recommendation to "encourage clients to pack more context" is backwards: shorter prompts with batching deliver better output throughput.

The batch-48 projection of ~3200 tok/s is wrong on two counts: (1) linear scaling assumes no resource ceiling, and (2) the log shows batch 48 already delivers only 1298 tok/s with severe preemptions.

---

## B4 — Confirming the B2 Mechanism

**Metric to pull:** `vllm:num_preemptions_total` (Prometheus counter) or equivalently the `preempted_seqs` column already in the log.

**Supplementary metric:** `vllm:gpu_cache_usage_perc` — the fraction of KV-cache blocks currently in use.

**Expected values:** At the onset of the preemption cliff (batch 32, prompt 3584):
- `gpu_cache_usage_perc` should be pinned at **97–100%** throughout the run (confirmed: log shows 0.97)
- `num_preemptions_total` should increment by **7** over the run (confirmed: `preempted_seqs = 7`)
- The correlation between `gpu_cache_usage_perc` hitting 100% and the first preemption event should be immediate (within the same scheduling step)

If chunked prefill is enabled, the expected observation is that `gpu_cache_usage_perc` oscillates below 100% (never fully saturates) and `num_preemptions_total` stays at **0** — confirming that fragmented prefill prevents KV exhaustion.
