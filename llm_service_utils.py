import re

_COMPLIANCE_ID_PATTERN = re.compile(
    r'\b([A-Z]{2,8}-[A-Z]{0,4}-?\d{3,})\b'
)

def _extract_compliance_ids(text: str) -> list:
    """Extract all compliance IDs from a block of text."""
    return list(dict.fromkeys(_COMPLIANCE_ID_PATTERN.findall(text)))


def _extract_id_lines(text: str, ids: list) -> dict:
    """For each ID, extract the full line from source text."""
    id_lines = {}
    for line in text.splitlines():
        for cid in ids:
            if cid in line and cid not in id_lines:
                id_lines[cid] = line.strip()
    return id_lines


def _verify_and_restore(rewritten: str, source_text: str, required_ids: list, source_lines: dict) -> str:
    """
    Post-generation completeness check.
    Any required ID missing from the rewrite output is appended verbatim from source.
    """
    missing = [cid for cid in required_ids if cid not in rewritten]
    if not missing:
        return rewritten

    restore_block = "\n\n<!-- Compliance Integrity Restore: the following entries were missing from the rewrite and have been restored verbatim -->\n"
    for cid in missing:
        restore_block += f"\n{source_lines.get(cid, cid)}"

    print(f"[ID Guard] Restored {len(missing)} missing IDs: {missing}")
    return rewritten + restore_block
