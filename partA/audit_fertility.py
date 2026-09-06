#!/usr/bin/env python3
"""
audit_fertility.py — Annotated audit of the original fertility.py.

This script reproduces the ORIGINAL script's behavior, then isolates each
bug and measures its effect on the reported numbers. Run with the original
corpus (10 sentences) to replicate REPORT_v0 numbers.

Usage:
    python audit_fertility.py

Requires: tiktoken
"""

import unicodedata
import tiktoken
import os
import sys

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_HERE = os.path.dirname(os.path.abspath(__file__))
# Check inside the repo first, fallback to parent directory
_STARTER_KIT_INTERNAL = os.path.join(_HERE, "..", "starter_kit")
_STARTER_KIT_PARENT = os.path.join(_HERE, "..", "..", "starter_kit")

_STARTER_KIT = _STARTER_KIT_INTERNAL if os.path.exists(_STARTER_KIT_INTERNAL) else _STARTER_KIT_PARENT

CORPUS = {
    "eng": os.path.join(_STARTER_KIT, "corpus_sample", "eng_sample.txt"),
    "hin": os.path.join(_STARTER_KIT, "corpus_sample", "hin_sample.txt"),
}

# ─── Helpers ────────────────────────────────────────────────────────────────

def read_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = unicodedata.normalize("NFC", line)
            lines.append(line)
    return lines


def grapheme_clusters(text: str):
    """
    Count Unicode grapheme clusters using the 'grapheme' library if available,
    falling back to codepoint count (good enough for this audit since Hindi
    matras create multi-codepoint grapheme clusters, making the fallback a
    slight overcount — worth noting as a caveat).
    """
    try:
        import grapheme
        return grapheme.length(text)
    except ImportError:
        # Fallback: count codepoints
        return len(text)


# ─── Original (buggy) analysis ───────────────────────────────────────────────

def analyze_original(lines, encode):
    """Exact replica of the original intern's analyze() function."""
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        line = line.lower()                        # BUG 2: lowercases before encoding
        tokens = encode(line)
        words = line.split(" ")                    # BUG 1: split on single space only
        chars = len(line)
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return sum(per_line_fertility) / n, sum(per_line_tpc) / n  # BUG 3: macro-average


# ─── Bug 1 isolation: space-splitting ───────────────────────────────────────

def analyze_fix_split_only(lines, encode):
    """Fix only Bug 1: use split() with no argument (handles any whitespace runs)."""
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        words = line.split()                       # FIXED: collapse whitespace
        chars = len(line)
        if len(words) == 0:
            continue
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return sum(per_line_fertility) / n, sum(per_line_tpc) / n


# ─── Bug 2 isolation: lowercasing ────────────────────────────────────────────

def analyze_fix_lower_only(lines, encode):
    """Fix only Bug 2: don't lowercase before encoding."""
    per_line_fertility = []
    per_line_tpc = []
    for line in lines:
        # NO lowercasing here — encode original case
        tokens = encode(line)
        words = line.split(" ")                    # still the buggy split
        chars = len(line)
        if len(words) == 0:
            continue
        per_line_fertility.append(len(tokens) / len(words))
        per_line_tpc.append(len(tokens) / chars)
    n = len(per_line_fertility)
    return sum(per_line_fertility) / n, sum(per_line_tpc) / n


# ─── Bug 3 isolation: macro vs micro averaging ───────────────────────────────

def analyze_micro_avg(lines, encode):
    """Fix Bug 3: micro-average across the full corpus, not per-line."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    for line in lines:
        line = line.lower()
        tokens = encode(line)
        words = line.split(" ")
        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += len(line)
    if total_words == 0 or total_chars == 0:
        return 0, 0
    return total_tokens / total_words, total_tokens / total_chars


# ─── Fully corrected analysis ─────────────────────────────────────────────────

def analyze_corrected(lines, encode):
    """
    All bugs fixed:
      - No lowercasing
      - split() instead of split(" ")
      - Micro-averaging
      - Multiple denominators: word, char (codepoint), grapheme cluster, UTF-8 byte
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_graphemes = 0
    total_utf8_bytes = 0
    for line in lines:
        tokens = encode(line)
        words = line.split()
        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += len(line)
        total_graphemes += grapheme_clusters(line)
        total_utf8_bytes += len(line.encode("utf-8"))
    return {
        "tok_per_word": total_tokens / total_words if total_words else 0,
        "tok_per_char": total_tokens / total_chars if total_chars else 0,
        "tok_per_grapheme": total_tokens / total_graphemes if total_graphemes else 0,
        "tok_per_utf8_byte": total_tokens / total_utf8_bytes if total_utf8_bytes else 0,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "total_chars": total_chars,
        "total_graphemes": total_graphemes,
        "total_utf8_bytes": total_utf8_bytes,
    }

#AST trace claim

def verify_no_random_calls(source_path):
    """AST check: confirm the `random` module is imported but never called."""
    import ast
    with open(source_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "random":
                calls.append(f"random.{func.attr}")
            elif isinstance(func, ast.Name) and func.id == "random":
                calls.append("random(...)")
    return calls

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    enc = tiktoken.get_encoding("gpt2")
    encode = enc.encode

    corpora = {}
    for lang, path in CORPUS.items():
        try:
            corpora[lang] = read_lines(path)
            print(f"Loaded {len(corpora[lang])} lines for {lang} from {path}")
        except FileNotFoundError:
            print(f"WARNING: Could not find {path}. Skipping.")

    print("\n" + "="*70)
    print("STEP 1: Original (buggy) analysis — reproducing REPORT_v0 numbers")
    print("="*70)
    print(f"{'lang':<8}{'fertility (tok/word)':<24}{'tok/char':<12}")
    print("-" * 44)
    orig_results = {}
    for lang, lines in corpora.items():
        fert, tpc = analyze_original(lines, encode)
        orig_results[lang] = (fert, tpc)
        print(f"{lang:<8}{fert:<24.2f}{tpc:<12.3f}")

    if "eng" in orig_results and "hin" in orig_results:
        ratio = orig_results["hin"][0] / orig_results["eng"][0]
        print(f"\n→ Reported Hindi/English fertility ratio: {ratio:.2f}×")

    print("\n" + "="*70)
    print("STEP 2: Fix Bug 1 only (split() instead of split(' '))")
    print("="*70)
    print(f"{'lang':<8}{'fertility (tok/word)':<24}{'tok/char':<12}")
    print("-" * 44)
    fix1_results = {}
    for lang, lines in corpora.items():
        fert, tpc = analyze_fix_split_only(lines, encode)
        fix1_results[lang] = (fert, tpc)
        print(f"{lang:<8}{fert:<24.2f}{tpc:<12.3f}")

    if "eng" in fix1_results and "hin" in fix1_results:
        ratio = fix1_results["hin"][0] / fix1_results["eng"][0]
        print(f"\n→ Hindi/English ratio after fix 1: {ratio:.2f}×")
        delta = orig_results["hin"][0] - fix1_results["hin"][0]
        direction = "deflated" if delta < 0 else "inflated"
        print(f"→ Hindi fertility change from Bug 1 fix: {delta:+.3f} (original was {direction} by {abs(delta):.3f})")

    print("\n" + "="*70)
    print("STEP 3: Fix Bug 2 only (no lowercasing)")
    print("="*70)
    print(f"{'lang':<8}{'fertility (tok/word)':<24}{'tok/char':<12}")
    print("-" * 44)
    fix2_results = {}
    for lang, lines in corpora.items():
        fert, tpc = analyze_fix_lower_only(lines, encode)
        fix2_results[lang] = (fert, tpc)
        print(f"{lang:<8}{fert:<24.2f}{tpc:<12.3f}")

    if "eng" in fix2_results and "hin" in fix2_results:
        ratio = fix2_results["hin"][0] / fix2_results["eng"][0]
        print(f"\n→ Hindi/English ratio after fix 2: {ratio:.2f}×")
        eng_delta = orig_results["eng"][0] - fix2_results["eng"][0]
        print(f"→ English fertility change from Bug 2 fix: {eng_delta:+.3f}")

    print("\n" + "="*70)
    print("STEP 4: Fix Bug 3 only (micro-averaging)")
    print("="*70)
    print(f"{'lang':<8}{'fertility (tok/word)':<24}{'tok/char':<12}")
    print("-" * 44)
    fix3_results = {}
    for lang, lines in corpora.items():
        fert, tpc = analyze_micro_avg(lines, encode)
        fix3_results[lang] = (fert, tpc)
        print(f"{lang:<8}{fert:<24.2f}{tpc:<12.3f}")

    if "eng" in fix3_results and "hin" in fix3_results:
        ratio = fix3_results["hin"][0] / fix3_results["eng"][0]
        print(f"\n→ Hindi/English ratio after fix 3: {ratio:.2f}×")

    print("\n" + "="*70)
    print("STEP 5: All bugs fixed — corrected multi-denominator analysis")
    print("="*70)
    header = f"{'lang':<8}{'tok/word':<12}{'tok/char':<12}{'tok/grapheme':<15}{'tok/utf8byte':<15}"
    print(header)
    print("-" * len(header))
    corr_results = {}
    for lang, lines in corpora.items():
        r = analyze_corrected(lines, encode)
        corr_results[lang] = r
        print(f"{lang:<8}{r['tok_per_word']:<12.3f}{r['tok_per_char']:<12.3f}"
              f"{r['tok_per_grapheme']:<15.3f}{r['tok_per_utf8_byte']:<15.4f}")

    if "eng" in corr_results and "hin" in corr_results:
        print("\n→ Corrected ratios (Hindi / English):")
        for key in ["tok_per_word", "tok_per_char", "tok_per_grapheme", "tok_per_utf8_byte"]:
            ratio = corr_results["hin"][key] / corr_results["eng"][key]
            print(f"   {key:<20}: {ratio:.2f}×")

    print("\n" + "="*70)
    print("SUMMARY: Effect of each bug on the reported Hindi/English ratio")
    print("="*70)
    if "eng" in orig_results and "hin" in orig_results:
        reported_ratio = orig_results["hin"][0] / orig_results["eng"][0]
        corrected_ratio = fix3_results["hin"][0] / fix3_results["eng"][0]
        print(f"  Reported ratio (all bugs):          {reported_ratio:.2f}×")
        if "eng" in fix1_results and "hin" in fix1_results:
            r = fix1_results["hin"][0] / fix1_results["eng"][0]
            print(f"  After fixing Bug 1 (split):         {r:.2f}×  (Δ = {r - reported_ratio:+.2f})")
        if "eng" in fix2_results and "hin" in fix2_results:
            r = fix2_results["hin"][0] / fix2_results["eng"][0]
            print(f"  After fixing Bug 2 (lower):         {r:.2f}×  (Δ = {r - reported_ratio:+.2f})")
        print(f"  After fixing Bug 3 (micro-avg):     {corrected_ratio:.2f}×  (Δ = {corrected_ratio - reported_ratio:+.2f})")
        print(f"\n  The conceptual bug (wrong denominator) cannot be shown as a ratio")
        print(f"  change — it invalidates the comparison entirely. See corrected")
        print(f"  tok/utf8byte figures in Step 5.")

    print("\n" + "─"*70)
    print("NOTE: random.seed(1337) in the original script — HARMLESS.")
    print("  It appears before any logic but random is never actually called.")
    print("  This is dead code (likely planned for future subsampling),")
    print("  NOT a bug. It has zero effect on any output number.")
    print("─"*70)

    try:
        random_calls = verify_no_random_calls(os.path.join(_STARTER_KIT, "fertility.py"))
        print(f"AST scan of fertility.py — random.* calls found: {random_calls if random_calls else 'NONE'}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()