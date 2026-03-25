"""Parse OPR Army Forge text exports into project JSON format.

Usage:
    python import_list.py <list_file> [-o output.json]
    python import_list.py "Imported Lists/List1.txt"
    python import_list.py "Imported Lists/List2.txt" -o list2.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from models import ArmyListEntry, Weapon, resolve_entry, validate_upgrades
from templates import get_templates_dict


# ===================================================================
# ARMY FORGE DISPLAY NAME → TEMPLATE DISPLAY NAME ALIASES
# ===================================================================

_NAME_ALIASES: dict[str, str] = {
    "Anti-Gravity Tank": "AG Tank",
    "Anti-Gravity APC": "AG APC",
}


# ===================================================================
# WEAPON COMPARISON KEYS
# ===================================================================

def _wkey(w: Weapon) -> tuple:
    """Hashable key for a template Weapon (excludes melee/thrust flags)."""
    return (w.name, w.attacks, w.range_inches, w.ap, w.deadly, w.blast,
            w.crack, w.rending, w.reliable, w.bane, w.unstoppable, w.takedown)


def _wcounter(weapons: list[Weapon]) -> Counter:
    return Counter(_wkey(w) for w in weapons)


def _twkey(name: str, st: dict) -> tuple:
    """Build the same key shape from parsed text weapon stats."""
    return (name,
            st.get('attacks', 1),
            st.get('range', 0),
            st.get('ap', 0),
            st.get('deadly', 0),
            st.get('blast', 0),
            st.get('crack', False),
            st.get('rending', False),
            st.get('reliable', False),
            st.get('bane', False),
            st.get('unstoppable', False),
            st.get('takedown', False))


# ===================================================================
# TEXT PARSING HELPERS
# ===================================================================

def _split0(text: str) -> list[str]:
    """Split on commas at paren-depth 0."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for c in text:
        if c == '(':
            depth += 1
        elif c == ')':
            depth = max(0, depth - 1)
        if c == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(c)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_wstats(raw: str) -> dict:
    """Parse weapon stat tokens: '18", A1, AP(4), Deadly(3)' → dict."""
    out: dict = {}
    for p in _split0(raw):
        p = p.strip()
        if p.endswith('"'):
            try:
                out['range'] = int(p.rstrip('"'))
            except ValueError:
                pass
        elif re.fullmatch(r'A\d+', p):
            out['attacks'] = int(p[1:])
        elif m := re.fullmatch(r'AP\((\d+)\)', p):
            out['ap'] = int(m[1])
        elif m := re.fullmatch(r'Deadly\((\d+)\)', p):
            out['deadly'] = int(m[1])
        elif m := re.fullmatch(r'Blast\((\d+)\)', p):
            out['blast'] = int(m[1])
        elif p == 'Crack':
            out['crack'] = True
        elif p == 'Rending':
            out['rending'] = True
        elif p == 'Reliable':
            out['reliable'] = True
        elif p == 'Takedown':
            out['takedown'] = True
        elif p == 'Unstoppable':
            out['unstoppable'] = True
        elif p == 'Bane':
            out['bane'] = True
    return out


def _parse_weapons(line: str) -> Counter:
    """Parse a weapon line → Counter of weapon keys."""
    result: Counter = Counter()
    for part in _split0(line):
        part = part.strip()
        if not part:
            continue
        # Optional "Nx " count prefix
        m = re.match(r'^(\d+)x\s+', part)
        cnt = int(m[1]) if m else 1
        if m:
            part = part[m.end():]
        # Separate name from (stats)
        pi = part.find('(')
        if pi < 0:
            result[_twkey(part.strip(), {})] += cnt
        else:
            name = part[:pi].strip()
            ci = part.rfind(')')
            raw_stats = part[pi + 1:ci] if ci > pi else part[pi + 1:]
            result[_twkey(name, _parse_wstats(raw_stats))] += cnt
    return result


def _parse_rules(text: str) -> dict:
    """Extract simulation-relevant special rules from the rules text."""
    r: dict = dict(
        hero=False, tough=0, fearless=False, scout=False, stealth=False,
        shielded=False, flying=False, teleport=False, fast=False, fear=0,
        relentless=False, regeneration=False, furious=False, fortified=False,
        impact=0,
    )
    for raw in text.split(','):
        p = re.sub(r'^\d+x\s+', '', raw.strip())  # strip "Nx " prefix
        lo = p.lower()
        for key in ('hero', 'fearless', 'scout', 'stealth', 'shielded', 'flying',
                     'teleport', 'fast', 'relentless', 'regeneration', 'furious',
                     'fortified'):
            if lo == key:
                r[key] = True
                break
        else:
            if m2 := re.match(r'Tough\((\d+)\)', p):
                r['tough'] = int(m2[1])
            elif m2 := re.match(r'Fear\((\d+)\)', p):
                r['fear'] = int(m2[1])
            elif m2 := re.match(r'Impact\((\d+)\)', p):
                r['impact'] = int(m2[1])
    return r


# ===================================================================
# PARSED UNIT
# ===================================================================

@dataclass
class _ParsedUnit:
    name: str
    copies: int       # "3x Vanquishers" → 3
    size: int
    quality: int
    defense: int
    points: int
    rules: dict
    weapons: Counter
    joined_to: '_ParsedUnit | None' = None


# ===================================================================
# FILE PARSER
# ===================================================================

_HEADER_RE = re.compile(
    r'^(?:(\d+)x\s+)?'        # optional copies prefix
    r'(.+?)'                   # unit name
    r'\s+\[(\d+)\]'           # [size]
    r'\s+Q(\d+)\+\s+D(\d+)\+' # Q#+ D#+
)


def _is_header(line: str) -> bool:
    return bool(_HEADER_RE.match(line))


def _parse_header(line: str):
    """Parse unit header → (copies, name, size, Q, D, pts, rules)."""
    parts = line.split('|')
    m = _HEADER_RE.match(parts[0].strip())
    if not m:
        raise ValueError(f"Cannot parse unit header: {line}")
    copies = int(m[1]) if m[1] else 1
    name = m[2].strip()
    size, q, d = int(m[3]), int(m[4]), int(m[5])
    pts_m = re.search(r'(\d+)pts', parts[1]) if len(parts) > 1 else None
    pts = int(pts_m[1]) if pts_m else 0
    rules = _parse_rules(parts[2].strip()) if len(parts) > 2 else _parse_rules("")
    return copies, name, size, q, d, pts, rules


def _read_weapons_lines(lines: list[str], i: int) -> tuple[str, int]:
    """Consume consecutive weapon lines starting at index i.
    Returns (concatenated weapons text, new index)."""
    wtext = ""
    while i < len(lines):
        wl = lines[i].strip()
        if not wl or wl.startswith('++') or wl.startswith('|') or _is_header(wl):
            break
        wtext += (", " if wtext else "") + wl
        i += 1
    return wtext, i


def parse_list_file(path: str | Path) -> list[_ParsedUnit]:
    """Parse an Army Forge text export file → list of _ParsedUnit."""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    units: list[_ParsedUnit] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('++'):
            i += 1
            continue
        if not _is_header(line):
            i += 1
            continue

        copies, name, size, q, d, pts, rules = _parse_header(line)
        i += 1

        wtext, i = _read_weapons_lines(lines, i)
        weapons = _parse_weapons(wtext) if wtext else Counter()
        unit = _ParsedUnit(name, copies, size, q, d, pts, rules, weapons)

        # Hero attachment: "| Joined to:"
        if i < len(lines) and lines[i].strip() == '| Joined to:':
            i += 1
            if i < len(lines) and _is_header(lines[i].strip()):
                jc, jn, js, jq, jd, jp, jr = _parse_header(lines[i].strip())
                i += 1
                jwtext, i = _read_weapons_lines(lines, i)
                unit.joined_to = _ParsedUnit(
                    jn, jc, js, jq, jd, jp, jr,
                    _parse_weapons(jwtext) if jwtext else Counter())

        units.append(unit)
    return units


# ===================================================================
# TEMPLATE MATCHING
# ===================================================================

def _find_template_id(name: str, size: int) -> str | None:
    """Find template_id by Army Forge display name and model count."""
    lookup = _NAME_ALIASES.get(name, name)
    tpl_dict = get_templates_dict()
    # Exact name match (strip " (Combined)" from template names)
    for tid, tpl in tpl_dict.items():
        base_name = tpl.name.replace(" (Combined)", "")
        if base_name == lookup and tpl.size == size:
            return tid
    # Case-insensitive fallback
    for tid, tpl in tpl_dict.items():
        base_name = tpl.name.replace(" (Combined)", "").lower()
        if base_name == lookup.lower() and tpl.size == size:
            return tid
    return None


# ===================================================================
# UPGRADE BRUTE-FORCE SEARCH
# ===================================================================

def _rules_ok(resolved, parsed_rules: dict, tpl) -> bool:
    """Check that rules *added by upgrades* are consistent with text.

    Only rejects if the text claims a rule that the base template lacks
    and the resolved unit also lacks it (meaning no upgrade added it).
    Does NOT reject when the text lists a base rule that an upgrade removes,
    because Army Forge exports show base-profile rules regardless of upgrades.
    """
    for attr, parsed_key in [
        ('flying', 'flying'), ('teleport', 'teleport'),
        ('scout', 'scout'), ('stealth', 'stealth'),
        ('fast', 'fast'), ('relentless', 'relentless'),
        ('fearless', 'fearless'), ('regeneration', 'regeneration'),
        ('fortified', 'fortified'),
    ]:
        parsed_val = parsed_rules[parsed_key]
        base_val = getattr(tpl, attr)
        resolved_val = getattr(resolved, attr)
        # Text says unit has rule, base doesn't → upgrade must provide it
        if parsed_val and not base_val and not resolved_val:
            return False
    # Tough: if text shows a value different from base, resolved must match
    if parsed_rules['tough'] and parsed_rules['tough'] != tpl.tough:
        if resolved.tough != parsed_rules['tough']:
            return False
    # Fear: same logic
    if parsed_rules['fear'] and parsed_rules['fear'] != tpl.fear:
        if resolved.fear != parsed_rules['fear']:
            return False
    return True


def _weapon_overlap(resolved_weapons: list, target: Counter) -> float:
    """Fraction of weapons that match between resolved and target (0..1)."""
    resolved = _wcounter(resolved_weapons)
    all_keys = set(resolved) | set(target)
    if not all_keys:
        return 1.0
    matching = sum(min(resolved.get(k, 0), target.get(k, 0)) for k in all_keys)
    total = sum(max(resolved.get(k, 0), target.get(k, 0)) for k in all_keys)
    return matching / total if total else 1.0


def _find_upgrades(template_id: str, parsed: _ParsedUnit,
                   warnings: list[str]) -> dict[str, str]:
    """Try all upgrade combinations; return the one matching weapons + cost + rules.

    Pass 1: exact weapon + cost + rules match.
    Pass 2 (fallback for combined templates): cost + rules match, best weapon overlap.
    """
    tpl_dict = get_templates_dict()
    tpl = tpl_dict[template_id]

    if not tpl.upgrade_slots:
        entry = ArmyListEntry(template_id=template_id)
        resolved = resolve_entry(entry)
        if resolved.points != parsed.points:
            warnings.append(
                f"  Cost mismatch for {parsed.name}: "
                f"expected {parsed.points}, got {resolved.points}")
        if _wcounter(resolved.weapons) != parsed.weapons:
            warnings.append(f"  Weapon mismatch for {parsed.name} (no upgrade slots)")
        return {}

    # Build per-slot choice lists: [(slot_id, option_id | None), ...]
    slot_choices: list[list[tuple[str, str | None]]] = []
    for slot in tpl.upgrade_slots:
        choices: list[tuple[str, str | None]] = [(slot.id, None)]
        for opt in slot.options:
            choices.append((slot.id, opt.id))
        slot_choices.append(choices)

    target_weapons = parsed.weapons
    target_cost = parsed.points
    exact_matches: list[dict[str, str]] = []
    # For fallback: (overlap_score, upgrades, resolved_weapons)
    best_fuzzy: tuple[float, dict[str, str], list] | None = None

    for combo in product(*slot_choices):
        upgrades: dict[str, str] = {}
        for slot_id, opt_id in combo:
            if opt_id is not None:
                upgrades[slot_id] = opt_id

        entry = ArmyListEntry(template_id=template_id, chosen_upgrades=upgrades)
        if not validate_upgrades(entry):
            continue

        try:
            resolved = resolve_entry(entry)
        except Exception:
            continue

        if resolved.points != target_cost:
            continue
        if not _rules_ok(resolved, parsed.rules, tpl):
            continue

        if _wcounter(resolved.weapons) == target_weapons:
            exact_matches.append(dict(upgrades))
            if len(exact_matches) >= 2:
                break
        else:
            # Track best fuzzy match for fallback
            overlap = _weapon_overlap(resolved.weapons, target_weapons)
            if best_fuzzy is None or overlap > best_fuzzy[0]:
                best_fuzzy = (overlap, dict(upgrades), resolved.weapons)

    if exact_matches:
        if len(exact_matches) > 1:
            warnings.append(
                f"  WARNING: Multiple upgrade combinations match "
                f"{parsed.name} ({parsed.points}pts) — using first")
        return exact_matches[0]

    # Fallback: use best fuzzy match (common with combined templates)
    if best_fuzzy is not None:
        overlap_pct = best_fuzzy[0] * 100
        warnings.append(
            f"  WARNING: No exact weapon match for {parsed.name} "
            f"({parsed.points}pts) — using best-effort upgrade match "
            f"({overlap_pct:.0f}% weapon overlap)")
        return best_fuzzy[1]

    warnings.append(
        f"  WARNING: No matching upgrade combination for "
        f"{parsed.name} ({parsed.points}pts) — using empty upgrades")
    return {}


# ===================================================================
# MAIN CONVERSION
# ===================================================================

def convert_list(path: str | Path) -> tuple[dict, list[str]]:
    """Convert an Army Forge text file → JSON-ready dict + warnings."""
    parsed_units = parse_list_file(path)
    entries: list[dict] = []
    warnings: list[str] = []

    for pu in parsed_units:
        if pu.joined_to:
            # Hero (pu) attached to host (pu.joined_to)
            hero_idx = len(entries)
            host_idx = hero_idx + 1

            # --- hero ---
            tid = _find_template_id(pu.name, pu.size)
            if tid is None:
                warnings.append(
                    f"WARNING: No template for '{pu.name}' [size={pu.size}]")
                continue
            upgrades = _find_upgrades(tid, pu, warnings)
            entries.append({
                'template_id': tid,
                'upgrades': upgrades,
                'ai_role': 'killer',
                'combat_preference': 'ranged',
                'attached_to': host_idx,
                'cost': pu.points,
            })

            # --- host ---
            host = pu.joined_to
            htid = _find_template_id(host.name, host.size)
            if htid is None:
                warnings.append(
                    f"WARNING: No template for '{host.name}' [size={host.size}]")
                entries.append({
                    'template_id': '???',
                    'upgrades': {},
                    'ai_role': 'killer',
                    'combat_preference': 'ranged',
                    'attached_to': -1,
                    'cost': host.points,
                })
                continue
            h_upgrades = _find_upgrades(htid, host, warnings)
            entries.append({
                'template_id': htid,
                'upgrades': h_upgrades,
                'ai_role': 'killer',
                'combat_preference': 'ranged',
                'attached_to': -1,
                'cost': host.points,
            })

        else:
            # Regular unit — emit `copies` identical entries
            tid = _find_template_id(pu.name, pu.size)
            if tid is None:
                warnings.append(
                    f"WARNING: No template for '{pu.name}' [size={pu.size}]")
                continue
            upgrades = _find_upgrades(tid, pu, warnings)
            for _ in range(pu.copies):
                entries.append({
                    'template_id': tid,
                    'upgrades': upgrades,
                    'ai_role': 'killer',
                    'combat_preference': 'ranged',
                    'attached_to': -1,
                    'cost': pu.points,
                })

    total_cost = sum(e['cost'] for e in entries)
    return {'total_cost': total_cost, 'entries': entries}, warnings


def import_list(path: str | Path,
                output_path: str | Path | None = None) -> dict:
    """Import an Army Forge text file, print/write JSON, return result."""
    result, warnings = convert_list(path)

    for w in warnings:
        print(w, file=sys.stderr)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Written to {output_path}")
    else:
        print(json.dumps(result, indent=2))

    return result


# ===================================================================
# CLI
# ===================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <list_file> [-o output.json]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = None
    if '-o' in sys.argv:
        idx = sys.argv.index('-o')
        if idx + 1 < len(sys.argv):
            out_path = sys.argv[idx + 1]

    import_list(in_path, out_path)
