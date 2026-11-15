"""Search-side text normalisation.

Everything here produces *search* text only. Display text is never touched
by these functions, so normalisation can be aggressive without changing what
the user sees.
"""
from __future__ import annotations

import re
import unicodedata

import snowballstemmer

_stemmer = snowballstemmer.stemmer("english")

SYMBOL_WORDS = {
    "≥": " greater or equal ",
    "≤": " less or equal ",
    "°": " degree ",
    "%": " percent ",
    "±": " plus minus ",
    "−": " minus ",
    "–": " ",
    "—": " ",
    "⋅": " times ",
    "×": " times ",
    "Ω": " ohm ",
    "µ": " micro ",
    "μ": " micro ",
    "€": " euro ",
}

# common English words that collide with 2-letter glossary keys: only expand
# these when the user typed them in capitals
AMBIGUOUS_ABBREVIATIONS = {"AS", "IN", "OC", "DC", "EI", "HY", "IA", "LV", "HV", "TS", "DQ", "DV", "DI", "CV", "EV"}

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "be", "by", "with", "as", "at", "it",
    "that", "this", "must", "may", "can", "i", "my", "we", "our", "you", "your", "do", "does", "what", "which", "how",
    "when", "where", "who", "if", "there", "any", "all", "from", "than", "not", "no", "into", "its", "their", "they",
    "them", "have", "has", "had", "will", "should", "would", "could", "about", "does", "need", "needs", "get",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def nfkc(s: str) -> str:
    # NFKC maps mathematical italic letters (𝑃, 𝑇) to plain ASCII and fixes nbsp
    return unicodedata.normalize("NFKC", s)


def normalize_for_search(s: str) -> str:
    s = nfkc(s).replace("­", "")
    for k, v in SYMBOL_WORDS.items():
        s = s.replace(k, v)
    s = s.lower()
    s = s.replace("-", " ").replace("/", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip()


def tokenize(s: str, stem: bool = True) -> list[str]:
    toks = TOKEN_RE.findall(normalize_for_search(s))
    if stem:
        toks = _stemmer.stemWords(toks)
    return toks


def raw_tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(normalize_for_search(s))


def stem(tok: str) -> str:
    return _stemmer.stemWord(tok)


def expand_glossary_in_text(text: str, glossary: dict[str, str]) -> list[str]:
    """Return glossary expansions for abbreviations that appear (case-sensitive,
    whole word) in `text`. Used when building document search text."""
    out = []
    for abbr, expansion in glossary.items():
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(abbr)}(?![A-Za-z0-9])", text):
            out.append(expansion)
    return out


def expand_glossary_in_query(query: str, glossary: dict[str, str]) -> list[tuple[str, str]]:
    """Expansions for abbreviations typed in a query. Lower-case matching is
    allowed for keys of 3+ letters; 2-letter / ambiguous keys only when typed
    in capitals exactly."""
    found = []
    words = re.findall(r"[A-Za-z0-9]+", query)
    upper_map = {k.upper(): k for k in glossary}
    for w in words:
        key = upper_map.get(w.upper())
        if not key:
            continue
        if key in AMBIGUOUS_ABBREVIATIONS or len(key) < 3:
            if w != key:
                continue
        found.append((key, glossary[key]))
    return found
