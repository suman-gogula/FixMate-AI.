"""
rules.py - Rule-based components of FixMate AI

IMPORTANT FOR VIVA:
    * The complaint CATEGORY is predicted by a Machine Learning model (model.py).
    * URGENCY, DEPARTMENT and RECOMMENDED ACTION are decided here using simple,
      transparent IF-THEN rules (keyword / pattern matching). They are NOT
      machine learning. Every decision returns the rule that caused it.
"""

import re

CATEGORIES = [
    "Plumbing",
    "Electrical",
    "Cleaning",
    "Furniture",
    "Air Conditioning",
    "Structural Issues",
]

DEPARTMENTS = {
    "Plumbing": "Plumbing Department",
    "Electrical": "Electrical Department",
    "Cleaning": "Cleaning Department",
    "Furniture": "Furniture Maintenance Department",
    "Air Conditioning": "HVAC Department",
    "Structural Issues": "Civil Maintenance Department",
}


def assign_department(category: str) -> str:
    """Category -> responsible department (simple lookup table)."""
    return DEPARTMENTS.get(category, "General Maintenance")


# ---------------------------------------------------------------------------
# Urgency rules
# ---------------------------------------------------------------------------
def _any(text, patterns):
    return any(re.search(p, text) for p in patterns)


_LEAK = r"\b(leak|leaks|leaking|leaked|leakage)\b"
_HEAVY = (r"\b(continuous|continuously|heavy|heavily|major|severe|burst|pouring|"
          r"gushing|non-stop|nonstop|ceiling|everywhere)\b")
_STRUCT_PART = r"\b(wall|walls|ceiling|pillar|beam|roof|column|slab|building)\b"
_COSMETIC = r"\b(small|minor|hairline|tiny|slight)\b"


def _fire(t):
    return _any(t, [r"\b(fire|flame|flames|smoke)\b", r"burning smell",
                    r"smell of burning", r"short ?circuit", r"\bspark(s|ing)?\b"])


def _gas(t):
    return _any(t, [r"\bgas\b.{0,25}\b(leak|leaking|smell|smells)\b",
                    r"\b(smell|smells|leak|leaking)\b.{0,15}\bgas\b",
                    r"\blpg\b"])


def _flood(t):
    return _any(t, [r"\bflood(ed|ing|s)?\b", r"water everywhere", r"submerged"])


def _major_leak(t):
    return bool(re.search(_LEAK, t)) and bool(re.search(_HEAVY, t))


def _exposed_wires(t):
    return _any(t, [
        r"\b(exposed|open|bare|live|loose|hanging)\s+(electric(al)?\s+)?wires?\b",
        r"\bwires?\s+(is\s+|are\s+)?(exposed|open|hanging|bare)\b",
        r"electric shock", r"\bshock\b"])


def _structural(t):
    if re.search(_COSMETIC, t) and re.search(r"crack", t):
        return False                              # small crack = cosmetic, not high
    return _any(t, [
        rf"{_STRUCT_PART}.{{0,30}}(collapse|collapsed|fallen|falling down|caved)",
        rf"(collapse|collapsed|caved).{{0,30}}{_STRUCT_PART}",
        rf"crack(s|ed)?\b.{{0,40}}{_STRUCT_PART}",
        rf"{_STRUCT_PART}.{{0,40}}crack(s|ed)?\b",
        r"(plaster|concrete|cement).{0,25}(falling|fall|coming off|came off)",
        rf"{_STRUCT_PART}.{{0,20}}(bulging|sinking|leaning)"])


# Order matters: first matching rule wins.
HIGH_RULES = [
    ("Fire / smoke / short circuit", _fire),
    ("Gas leakage", _gas),
    ("Flooding", _flood),
    ("Major water leakage", _major_leak),
    ("Exposed electrical wires / shock", _exposed_wires),
    ("Structural damage", _structural),
]

MEDIUM_RULES = [
    ("AC / air-conditioning not working", lambda t: _any(t, [r"\b(ac|a/c|hvac)\b", r"air[- ]?condition\w*"])),
    ("Water tap malfunction", lambda t: _any(t, [r"\b(tap|taps|faucet)\b"])),
    ("Drainage problem", lambda t: _any(t, [r"\b(drain|drains|drainage|clog|clogged|blocked|choked|sewage|flush)\b"])),
    ("Broken furniture", lambda t: _any(t, [
        r"\b(chair|table|desk|bed|cot|cupboard|wardrobe|almirah|drawer|bench|sofa|shelf|"
        r"bookshelf|cabinet|furniture|mattress|locker)\b"]) and _any(t, [
        r"broken|damage|torn|loose|stuck|jammed|not closing"])),
]

LOW_RULES = [
    ("Light bulb / tube-light replacement", lambda t: _any(t, [r"\bbulbs?\b", r"tube ?lights?"])),
    ("Light not working (minor electrical)", lambda t: _any(t, [
        r"\blights?\b.{0,30}(not working|not glowing|fused|dim|flicker\w*)"])),
    ("Minor cleaning", lambda t: _any(t, [r"\b(dust|dusty|cobwebs?|stains?|stained|mop|mopping|sweep|sweeping)\b"])),
    ("Small cosmetic repair", lambda t: _any(t, [
        r"\b(paint|peeling|scratch|scratched|cosmetic|minor)\b",
        r"(small|hairline|tiny|slight) crack"])),
]

_DEFAULT_URGENCY = {"Cleaning": "Low"}      # every other category defaults to Medium


def detect_urgency(text: str, category: str = "") -> dict:
    """
    Decide urgency with transparent rules.
    Order: HIGH rules -> specific MEDIUM rules -> LOW rules -> category default.
    Returns {"level": ..., "reason": ...}
    """
    t = text.lower()
    for label, rule in HIGH_RULES:
        if rule(t):
            return {"level": "High", "reason": label}
    for label, rule in MEDIUM_RULES:
        if rule(t):
            return {"level": "Medium", "reason": label}
    for label, rule in LOW_RULES:
        if rule(t):
            return {"level": "Low", "reason": label}
    level = _DEFAULT_URGENCY.get(category, "Medium")
    return {"level": level, "reason": f"No special keyword found - default for {category or 'unknown'}"}


# ---------------------------------------------------------------------------
# Recommended action
# ---------------------------------------------------------------------------
_AREA = {
    "Plumbing": "plumbing",
    "Electrical": "electrical",
    "Cleaning": "cleaning",
    "Furniture": "furniture",
    "Air Conditioning": "air-conditioning",
    "Structural Issues": "civil",
}

_SAFETY = {
    "Plumbing": "Close the nearest water valve if you can do so safely.",
    "Electrical": "Switch off the power supply and keep away from the affected area.",
    "Cleaning": "Keep the area clear until it is cleaned.",
    "Furniture": "Stop using the damaged item until it is repaired.",
    "Air Conditioning": "Switch off the AC and keep the room ventilated.",
    "Structural Issues": "Evacuate and avoid the damaged area until engineers inspect it.",
}


def recommend_action(category: str, urgency: str, text: str) -> str:
    t = text.lower()
    if urgency == "High":
        return "Immediate inspection and repair."
    if category == "Electrical" and urgency == "Low" and re.search(r"\bbulbs?\b|tube ?light", t):
        return "Schedule a light bulb replacement."
    if urgency == "Medium":
        return "Schedule inspection and repair within 24 to 48 hours."
    return f"Schedule routine {_AREA.get(category, 'maintenance')} maintenance."


def safety_note(category: str, urgency: str) -> str:
    """Short safety tip, shown only for High urgency complaints."""
    return _SAFETY.get(category, "") if urgency == "High" else ""
