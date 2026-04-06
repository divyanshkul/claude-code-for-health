"""Parse CLI command strings into (command_name, args) tuples."""


def parse(raw: str) -> tuple[str, list[str]]:
    """
    Parse a raw command string.

    'chart.labs CBC'           -> ('chart.labs', ['CBC'])
    'ddx.confirm Dengue fever' -> ('ddx.confirm', ['Dengue fever'])
    'note.correct 5 Fixed.'    -> ('note.correct', ['5', 'Fixed.'])
    'submit 25.2'              -> ('submit', ['25.2'])
    'chart.vitals'             -> ('chart.vitals', [])
    ''                         -> ('', [])
    """
    stripped = raw.strip()
    if not stripped:
        return ("", [])

    parts = stripped.split(None, 1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    if cmd == "note.correct" and rest:
        tokens = rest.split(None, 1)
        sentence_id = tokens[0]
        correction_text = tokens[1] if len(tokens) > 1 else ""
        return (cmd, [sentence_id, correction_text])

    if rest:
        return (cmd, [rest])
    return (cmd, [])
