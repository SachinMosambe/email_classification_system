"""
PII masking utilities with strict position tracking (No spaCy version)
"""

import re
from typing import Tuple, List, Dict

PII_PATTERNS = {
    "phone_number": [
        # Improved international and local phone number regex
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{1,4}[-.\s]?){2,5}\d{2,4}\b"
    ],
    "full_name": [
        r"\b(?:Mr|Ms|Mrs|Dr)\.?\s+[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b",  # With titles
        r"\b[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}\b",  # Stricter name pattern
    ],
    "email": [
        r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b"
    ],
    "dob": [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b"
    ],
    "aadhar_num": [
        r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"
    ],
    "credit_debit_no": [
        r"\b(?:\d[ -]*?){16}\b",  # Generic 16-digit cards
        r"\b(?:\d[ -]*?){15}\b"   # Amex 15-digit cards
    ],
    "cvv_no": [
        r"(?<!\d)(\d{3,4})(?!\d)"
    ],
    "expiry_no": [
        r"\b(0[1-9]|1[0-2])\/(\d{2}|\d{4})\b"
    ]
}

def mask_pii(text: str) -> Tuple[str, List[Dict]]:
    """
    Mask PII with exact position tracking using only regex

    Returns:
        Tuple: (masked_text, entities)
        entities format:
        [
            {
                "position": [start, end],
                "classification": "entity_type",
                "entity": "original_text"
            }
        ]
    """
    entities = []
    matches = []

    # Collect all matches with their positions and types
    for entity_type, patterns in PII_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                # Avoid zero-length matches
                if start == end:
                    continue
                matches.append({
                    "start": start,
                    "end": end,
                    "classification": entity_type,
                    "entity": match.group()
                })

    # Sort matches by start, then by longest match (end descending)
    matches.sort(key=lambda x: (x['start'], -x['end']))

    # Remove overlapping matches, keep the longest/first
    non_overlapping = []
    last_end = -1
    for m in matches:
        if m['start'] >= last_end:
            non_overlapping.append(m)
            last_end = m['end']

    # Mask from end to start to avoid messing up indices
    masked_text = text
    offset = 0
    entities = []
    for m in reversed(non_overlapping):
        start, end = m['start'], m['end']
        replacement = f"[{m['classification']}]"
        masked_text = masked_text[:start] + replacement + masked_text[end:]
        entities.append({
            "position": [start, start + len(replacement)],
            "classification": m['classification'],
            "entity": m['entity']
        })

    # Since we processed in reverse, reverse entities to restore order
    entities.reverse()
    return masked_text, entities