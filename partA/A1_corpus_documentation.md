# Part A1 — Evaluation Corpus Documentation & Domain Caveats

## 1. Corpus Selection & Provenance
To replace the 10-sentence toy sample (`corpus_sample/`), we assembled a multilingual parallel evaluation corpus derived from **FLORES-200** (via the open, ungated `openlanguagedata/flores_plus` distribution, `devtest` split).

FLORES-200 was created by professional human translators translating from an English baseline to ensure parallel semantic alignment across languages, avoiding synthetic translation artifacts.

---

## 2. Language Coverage & Typology
While the prompt required at least 4 languages including English, Hindi, and two Dravidian languages, we expanded the evaluation to **7 languages** to cover all target languages relevant to Flam's conversational roadmap:

| Language Code | Language | Script | Glottocode | Language Family | Morphological Type |
|---|---|---|---|---|---|
| `eng` | English | Latin (`Latn`) | `stan1293` | Indo-European (Germanic) | Isolating / Analytic |
| `hin` | Hindi | Devanagari (`Deva`) | `hind1269` | Indo-European (Indo-Aryan) | Fusional / Inflected |
| `mar` | Marathi | Devanagari (`Deva`) | `mara1378` | Indo-European (Indo-Aryan) | Fusional / Agglutinative tendencies |
| `ben` | Bengali | Bengali (`Beng`) | `beng1280` | Indo-European (Indo-Aryan) | Fusional |
| `kan` | Kannada | Kannada (`Knda`) | `kann1255` | Dravidian (Southern) | Strongly Agglutinative |
| `tel` | Telugu | Telugu (`Telu`) | `telu1262` | Dravidian (South-Central) | Strongly Agglutinative |
| `tam` | Tamil | Tamil (`Taml`) | `tami1289` | Dravidian (Southern) | Strongly Agglutinative |

---

## 3. Corpus Size & Volume
Each language split contains exactly **1,012 parallel sentences** drawn from identical source documents:

| Language | Sentences | Whitespace Words | Unicode Codepoints | Grapheme Clusters | UTF-8 Bytes |
|---|---:|---:|---:|---:|---:|
| English (`eng`) | 1,012 | 21,901 | 131,966 | 131,966 | 132,096 |
| Hindi (`hin`) | 1,012 | 25,643 | 131,180 | 90,733 | 337,439 |
| Marathi (`mar`) | 1,012 | 19,046 | 133,051 | 89,255 | 355,511 |
| Bengali (`ben`) | 1,012 | 19,506 | 130,391 | 89,260 | 348,729 |
| Kannada (`kan`) | 1,012 | 16,100 | 138,027 | 90,371 | 375,341 |
| Telugu (`tel`) | 1,012 | 16,938 | 132,465 | 84,771 | 353,661 |
| Tamil (`tam`) | 1,012 | 16,775 | 154,131 | 99,724 | 421,635 |
| **Total** | **7,084** | **135,909** | **951,211** | **676,080** | **2,324,467** |

---

## 4. Domain Composition
The FLORES-200 sample spans multiple distinct thematic domains extracted from Wikipedia articles, Wikinews, and public web documents:
- **News & Current Affairs:** Politics, international relations, journalism.
- **Narrative & History:** Biographies, geography, historical events.
- **STEM:** Science, biology, medicine, climate, space exploration.
- **Culture & Arts:** Literature, media, sports.

---

## 5. Preprocessing Applied
To ensure a clean, reproducible, and linguistically fair evaluation:
1. **Unicode NFC Normalization:** Applied `unicodedata.normalize("NFC", line)` to resolve combining diacritics and composite characters into canonical precomposed forms.
2. **Whitespace Regularization:** Stripped leading and trailing whitespace; preserved internal single whitespace boundaries for tokenizer input fidelity.
3. **Preservation of Casing & Punctuation:** No artificial lowercasing (`.lower()`) was applied, preserving proper nouns, acronyms, and script markers without distorting BPE token boundaries.

---

## 6. What This Corpus Cannot Tell You (Domain & Sample-Size Caveats)

> [!IMPORTANT]
> A benchmark is only as valid as its alignment with production reality. The FLORES-200 eval set provides strong parallel signal for formal language, but there are specific things it **cannot tell us**:

1. **Colloquial & Mobile Conversational Register:**
   FLORES-200 consists of professionally edited, grammatically complete, formal prose. Flam builds interactive, conversational consumer products. Real user touchpoints feature short sentences, incomplete syntax, phonetic typos, colloquialisms, and slang (e.g., Hindi *"yaar"*, *"bhai"*, *"theek hai"* vs. textbook formal translations). Informal registers typically exhibit lower agglutinative density and shorter words, which may shift native fertility down by 15–25%.

2. **Code-Mixing & Script Transliteration (Hinglish, Tanglish, Kanglish):**
   A substantial fraction of Indian mobile users type Indic languages in the **Latin alphabet** rather than native Brahmic scripts (e.g., *"kya haal hai"* instead of *"क्या हाल है"*). FLORES is 100% native script. For GPT-2, Latin-transliterated Hindi bypasses byte-level fallback, dramatically improving fertility; conversely, for multilingual models trained on native scripts, transliterated text fragments into character-level chunks. FLORES cannot measure this trade-off.

3. **Domain Bias Toward Technical & Western Vocabulary:**
   Because FLORES originated from English Wikipedia and Wikinews, it contains transliterated Western terminology (e.g., "Type 1 Diabetes", "Toronto", "Space Shuttle"). In Indic scripts, borrowed technical words are represented phonetically using complex consonant clusters and viramas (halants), creating atypical subword fragmentation that would not occur in native cultural or commercial discourse.

4. **Document-Level Context & Conversational History:**
   FLORES sentences average ~20–25 words and are tested in isolation. Production generation involves multi-turn conversational dialogs where repeated entities benefit from KV-cache prefix caching. Sentence-level fertility cannot reflect how prefix caching amortizes tokenizer costs in production.
