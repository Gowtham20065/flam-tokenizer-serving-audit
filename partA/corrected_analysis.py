#!/usr/bin/env python3
"""
corrected_analysis.py — Corrected fertility analysis for Part A3.

Runs on the FLORES-200 eval corpus (4 languages × ~1,012 sentences each).
Uses two tokenizers:
  1. GPT-2    (gpt2 via tiktoken) — English-only BPE, baseline
  2. XLM-R   (xlm-roberta-base via HuggingFace) — multilingual, Indic-aware

Uses four denominators:
  1. Whitespace words   (tok/word)
  2. Unicode codepoints (tok/char)
  3. Grapheme clusters  (tok/grapheme) — best proxy for "perceived characters"
  4. UTF-8 bytes        (tok/byte)     — best for routing/cost decisions

Usage:
    python corrected_analysis.py [--corpus-dir corpus/]

Requires: tiktoken transformers sentencepiece
"""

import argparse
import os
import sys
import unicodedata

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── Grapheme cluster counter ─────────────────────────────────────────────────

def grapheme_len(text: str) -> int:
    """
    Count Unicode grapheme clusters. Uses the 'grapheme' package if available;
    otherwise falls back to counting non-combining codepoints (slight overcount
    for Indic scripts where combining marks extend base characters).
    """
    try:
        import grapheme
        return grapheme.length(text)
    except ImportError:
        # Fallback: count only non-combining codepoints as a proxy
        count = 0
        for ch in text:
            if unicodedata.category(ch) not in ("Mn", "Mc", "Me"):
                count += 1
        return max(count, 1)


# ─── Corpus loading ───────────────────────────────────────────────────────────

def read_corpus(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = unicodedata.normalize("NFC", raw.strip())
            if line:
                lines.append(line)
    return lines


# ─── Tokenizer loaders ───────────────────────────────────────────────────────

def load_gpt2():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode, "gpt2"


def load_xlmr():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    return lambda s: tok.encode(s, add_special_tokens=False), "xlm-roberta-base"


# ─── Core analysis (micro-average, no lowercasing) ────────────────────────────

def analyze_corpus(lines: list[str], encode) -> dict:
    """
    Micro-average across the full corpus.
    Returns counts and derived ratios for all four denominators.
    """
    total_tokens = 0
    total_words = 0
    total_codepoints = 0
    total_graphemes = 0
    total_utf8_bytes = 0

    for line in lines:
        # No lowercasing — preserve original tokenization
        tokens = encode(line)
        words = line.split()                   # split() collapses all whitespace

        total_tokens     += len(tokens)
        total_words      += len(words)
        total_codepoints += len(line)
        total_graphemes  += grapheme_len(line)
        total_utf8_bytes += len(line.encode("utf-8"))

    return {
        "total_tokens":     total_tokens,
        "total_words":      total_words,
        "total_codepoints": total_codepoints,
        "total_graphemes":  total_graphemes,
        "total_utf8_bytes": total_utf8_bytes,
        "tok_per_word":     total_tokens / total_words      if total_words      else 0,
        "tok_per_char":     total_tokens / total_codepoints if total_codepoints else 0,
        "tok_per_grapheme": total_tokens / total_graphemes  if total_graphemes  else 0,
        "tok_per_byte":     total_tokens / total_utf8_bytes if total_utf8_bytes else 0,
    }


# ─── Pretty printing ──────────────────────────────────────────────────────────

def print_table(results: dict, tokenizer_name: str, base_lang: str = "eng"):
    print(f"\n{'='*72}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"{'='*72}")
    print(f"{'lang':<8} {'tok/word':>10} {'tok/char':>10} {'tok/grapheme':>14} {'tok/byte':>10} {'n_sents':>8}")
    print(f"{'-'*8} {'-'*10} {'-'*10} {'-'*14} {'-'*10} {'-'*8}")

    for lang, r in results.items():
        n = r["total_tokens"]  # just for verification
        sents = r.get("n_sentences", "?")
        print(f"{lang:<8} {r['tok_per_word']:>10.3f} {r['tok_per_char']:>10.3f} "
              f"{r['tok_per_grapheme']:>14.3f} {r['tok_per_byte']:>10.4f} {sents:>8}")

    if base_lang in results:
        print(f"\n  Ratios relative to {base_lang}:")
        base = results[base_lang]
        for lang, r in results.items():
            if lang == base_lang:
                continue
            print(f"    {lang}: "
                  f"tok/word={r['tok_per_word']/base['tok_per_word']:.2f}×  "
                  f"tok/char={r['tok_per_char']/base['tok_per_char']:.2f}×  "
                  f"tok/grapheme={r['tok_per_grapheme']/base['tok_per_grapheme']:.2f}×  "
                  f"tok/byte={r['tok_per_byte']/base['tok_per_byte']:.2f}×")


def save_csv(results: dict, tokenizer_name: str, out_path: str):
    import csv
    rows = []
    for lang, r in results.items():
        rows.append({
            "tokenizer": tokenizer_name,
            "lang": lang,
            "n_sentences": r.get("n_sentences", ""),
            "total_tokens": r["total_tokens"],
            "total_words": r["total_words"],
            "total_codepoints": r["total_codepoints"],
            "total_graphemes": r["total_graphemes"],
            "total_utf8_bytes": r["total_utf8_bytes"],
            "tok_per_word": round(r["tok_per_word"], 4),
            "tok_per_char": round(r["tok_per_char"], 4),
            "tok_per_grapheme": round(r["tok_per_grapheme"], 4),
            "tok_per_byte": round(r["tok_per_byte"], 5),
        })
    fieldnames = list(rows[0].keys())
    file_exists = os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Corrected multilingual fertility analysis")
    ap.add_argument("--corpus-dir", default=os.path.join(os.path.dirname(__file__), "corpus"))
    ap.add_argument("--output-csv", default=os.path.join(os.path.dirname(__file__), "results", "corrected_fertility.csv"))
    args = ap.parse_args()

    LANGUAGES = ["eng", "hin", "kan", "tam", "tel", "ben", "mar"]

    # Load corpora
    print("Loading corpora...")
    corpora = {}
    for lang in LANGUAGES:
        path = os.path.join(args.corpus_dir, f"{lang}.txt")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {lang}")
            continue
        lines = read_corpus(path)
        corpora[lang] = lines
        print(f"  {lang}: {len(lines)} sentences loaded from {path}")

    if not corpora:
        print("ERROR: No corpora found. Run prepare_corpus.py first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    # Clear output CSV
    if os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    # Run for each tokenizer
    tokenizer_loaders = [load_gpt2, load_xlmr]

    # Keep every tokenizer's results around so the closing analysis below can
    # reference the *actual* numbers instead of a hardcoded guess written
    # before the corpus existed. (Earlier draft of this script printed a
    # static paragraph here that silently went stale once the real 1,012-
    # sentence FLORES run produced different numbers than the toy corpus —
    # that mismatch was caught during self-review; see NOTEBOOK.md Entry 03a.)
    all_results = {}

    for loader in tokenizer_loaders:
        try:
            encode, tok_name = loader()
        except Exception as e:
            print(f"\nWARNING: Could not load tokenizer: {e}")
            continue

        results = {}
        for lang, lines in corpora.items():
            print(f"  Analyzing {lang} with {tok_name}...", end=" ", flush=True)
            r = analyze_corpus(lines, encode)
            r["n_sentences"] = len(lines)
            results[lang] = r
            print(f"done ({r['total_tokens']:,} tokens)")

        print_table(results, tok_name)
        save_csv(results, tok_name, args.output_csv)
        all_results[tok_name] = results

    print(f"\nResults saved to {args.output_csv}")

    # ─── The key question: which denominator drives routing decisions? ───────
    print("\n" + "="*72)
    print("ANALYSIS: Which denominator should drive routing and cost decisions?")
    print("="*72)
    print("""
The routing question is: "How many tokens will the model generate per unit
of user-supplied input?" — because token count directly maps to compute cost,
KV-cache memory, and latency.

  tok/word     — WRONG for cross-language comparison. A Hindi whitespace-word
                 often packs 2-3 morphemes that English separates into distinct
                 words. You are comparing unlike units.

  tok/char     — Better, but 'char' is ambiguous: a single Unicode codepoint
                 in Devanagari (Hindi/Kannada) encodes one matra/consonant
                 while a Latin char encodes one letter. Comparable only if
                 scripts have similar codepoint density per phoneme.

  tok/grapheme — Grapheme clusters are the closest to "perceived characters"
                 and normalise for combining diacritics. Good for user-facing
                 estimates (e.g., "tokens per character the user types").

  tok/byte     — UTF-8 bytes are script-neutral and directly reflect storage
                 and transfer costs. This is the right denominator for a
                 routing and cost decision: it tells you how many tokens you
                 get per byte of raw input, which maps directly to serving
                 cost per byte of traffic.

RECOMMENDATION: Use tok/UTF-8 byte as the primary routing metric.
""")

    if "gpt2" in all_results and "eng" in all_results["gpt2"]:
        gpt2 = all_results["gpt2"]
        eng_byte = gpt2["eng"]["tok_per_byte"]
        print(f"  English (GPT-2): {eng_byte:.4f} tok/byte")
        for lang in ["hin", "kan", "tam", "tel", "ben", "mar"]:
            if lang in gpt2:
                ratio = gpt2[lang]["tok_per_byte"] / eng_byte
                print(f"  {lang.capitalize():<8} (GPT-2): {gpt2[lang]['tok_per_byte']:.4f} tok/byte  "
                      f"({ratio:.2f}x English)")

    if "xlm-roberta-base" in all_results and "eng" in all_results["xlm-roberta-base"]:
        xlmr = all_results["xlm-roberta-base"]
        eng_byte = xlmr["eng"]["tok_per_byte"]
        print(f"\n  English (XLM-R): {eng_byte:.4f} tok/byte")
        for lang in ["hin", "kan", "tam", "tel", "ben", "mar"]:
            if lang in xlmr:
                ratio = xlmr[lang]["tok_per_byte"] / eng_byte
                print(f"  {lang.capitalize():<8} (XLM-R): {xlmr[lang]['tok_per_byte']:.4f} tok/byte  "
                      f"({ratio:.2f}x English)")

    if ("gpt2" in all_results and "hin" in all_results["gpt2"]
            and "xlm-roberta-base" in all_results and "hin" in all_results["xlm-roberta-base"]):
        gpt2_ratio = all_results["gpt2"]["hin"]["tok_per_byte"] / all_results["gpt2"]["eng"]["tok_per_byte"]
        xlmr_ratio = (all_results["xlm-roberta-base"]["hin"]["tok_per_byte"]
                      / all_results["xlm-roberta-base"]["eng"]["tok_per_byte"])
        print(f"\nThe 5.89x figure in REPORT_v0 (tok/word, 10-sentence toy corpus) collapses to "
              f"{gpt2_ratio:.2f}x on tok/byte with GPT-2 on the full corpus, and to {xlmr_ratio:.2f}x "
              f"with XLM-R (below 1.0x means Hindi is CHEAPER per byte than English under a "
              f"multilingual tokenizer). The 'root cause' is the tokenizer, not the script.")


if __name__ == "__main__":
    main()
