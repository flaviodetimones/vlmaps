"""
room_priors.py
==============
Phase C — Object → room prior computation.

Computes a plausibility score in [0, 1] for each known room given an object query.
Three sources are fused:

  prior_final(r) = a * prior_llm(r) + b * prior_manual(r) + c * evidence(r)

  a=0.50  LLM soft signal  (GPT-4o-mini, cheap single call)
  b=0.30  Manual table     (stable, deterministic, fast)
  c=0.20  Scene evidence   (objects already seen per room)

The result is normalised so the scores sum to 1 across known rooms.
"""

from __future__ import annotations

import os
import re as _re
from typing import Dict, List, Optional

# ── Fusion weights (used for indirect queries without heatmap evidence) ──────
_W_LLM      = 0.45
_W_MANUAL   = 0.30
_W_EVIDENCE = 0.25

# ── Dynamic weights for direct queries (heatmap evidence available) ──────────
# direct:   heatmap dominates → 0.65 heatmap + 0.20 manual + 0.15 llm
# indirect: language priors → 0.45 llm + 0.30 manual + 0.25 scene-evidence
_W_DIRECT_HEATMAP = 0.65
_W_DIRECT_MANUAL  = 0.20
_W_DIRECT_LLM     = 0.15

# ── Manual object→room table ────────────────────────────────────────────────
# Format: object_keyword → {room_fragment: score}
# Room fragments are matched as substrings of the known room name (lower-cased).
# Score 1.0 = very likely, 0.5 = plausible, 0.2 = uncommon but possible.

_MANUAL_TABLE: Dict[str, Dict[str, float]] = {
    # bedroom objects
    "bed":          {"bedroom": 1.0},
    "pillow":       {"bedroom": 1.0},
    "nightstand":   {"bedroom": 1.0},
    "wardrobe":     {"bedroom": 0.9},
    "dresser":      {"bedroom": 0.9},
    # bathroom objects
    "toilet":       {"bathroom": 1.0},
    "bathtub":      {"bathroom": 1.0},
    "shower":       {"bathroom": 1.0},
    "sink":         {"bathroom": 0.9, "kitchen": 0.7, "laundry": 0.4},
    "mirror":       {"bathroom": 0.8, "bedroom": 0.4, "hallway": 0.2},
    # kitchen objects
    "refrigerator": {"kitchen": 1.0},
    "fridge":       {"kitchen": 1.0},
    "oven":         {"kitchen": 1.0},
    "microwave":    {"kitchen": 1.0},
    "counter":      {"kitchen": 0.9},
    "stove":        {"kitchen": 1.0},
    "dishwasher":   {"kitchen": 1.0},
    "bottle":       {"kitchen": 0.8, "dining": 0.6, "living": 0.3},
    # living room objects
    "sofa":         {"living": 1.0, "lounge": 1.0},
    "couch":        {"living": 1.0, "lounge": 1.0},
    "tv":           {"living": 0.9, "bedroom": 0.5},
    "television":   {"living": 0.9, "bedroom": 0.5},
    "monitor":      {"office": 1.0, "bedroom": 0.5, "living": 0.3},
    "plant":        {"living": 0.6, "hallway": 0.4, "balcony": 0.5},
    "fireplace":    {"living": 1.0},
    # office / desk
    "desk":         {"office": 1.0, "bedroom": 0.6},
    "laptop":       {"office": 0.9, "bedroom": 0.6, "living": 0.3},
    "computer":     {"office": 1.0, "bedroom": 0.4},
    "chair":        {"office": 0.5, "dining": 0.6, "living": 0.5, "kitchen": 0.4},
    # laundry
    "washer":       {"laundry": 1.0, "bathroom": 0.3},
    "dryer":        {"laundry": 1.0, "bathroom": 0.3},
    # hallway / entry
    "door":         {"hallway": 0.5, "entry": 0.5},
    "shelf":        {"living": 0.4, "office": 0.4, "bedroom": 0.3, "kitchen": 0.3},
    "shelving":     {"living": 0.4, "office": 0.4, "bedroom": 0.3},
    "cabinet":      {"kitchen": 0.6, "bathroom": 0.4, "office": 0.3},
    # balcony / outdoor
    "balcony":      {"balcony": 1.0},
    "table":        {"dining": 0.7, "kitchen": 0.5, "living": 0.4, "office": 0.4},
    "stool":        {"kitchen": 0.5, "bathroom": 0.4, "bar": 0.5},
    "lamp":         {"living": 0.5, "bedroom": 0.5, "office": 0.4},
    "picture":      {"living": 0.5, "hallway": 0.4, "bedroom": 0.4},
    "cushion":      {"living": 0.8, "bedroom": 0.4},
    "clothes":      {"bedroom": 0.7, "laundry": 0.8, "closet": 0.9},
    "appliances":   {"kitchen": 0.7, "laundry": 0.5},
    "window":       {"living": 0.4, "bedroom": 0.4, "kitchen": 0.3},
    "furniture":    {"living": 0.4, "bedroom": 0.4, "office": 0.3},
}

# ── Semantic room alias table ────────────────────────────────────────────────
# Maps scene-specific or atypical room names to the semantic families used in
# the manual prior table.  Applied only for semantic reasoning — room instance
# identity (e.g. bathroom.001) is never collapsed by this mapping.
_SEMANTIC_ALIASES: Dict[str, str] = {
    "tv":               "living",
    "tv room":          "living",
    "media":            "living",
    "media room":       "living",
    "lounge":           "living",
    "den":              "living",
    "family room":      "living",
    "sitting room":     "living",
    "laundryroom":      "laundry",
    "laundry room":     "laundry",
    "utility":          "laundry",
    "utility room":     "laundry",
    "washing":          "laundry",
    "study":            "office",
    "home office":      "office",
    "nursery":          "bedroom",
    "guest room":       "bedroom",
    "master bedroom":   "bedroom",
    "dining":           "dining",
    "dining room":      "dining",
    "eating area":      "dining",
    "breakfast":        "dining",
    "entrance":         "hallway",
    "entry":            "hallway",
    "foyer":            "hallway",
    "corridor":         "hallway",
    "passageway":       "hallway",
}

# ── Evidence co-occurrence table ────────────────────────────────────────────
# Objects that, when seen in a room, reinforce particular room types.
# Format: seen_object_keyword → room_fragment → reinforcement_score
_CO_OCCURRENCE: Dict[str, Dict[str, float]] = {
    "toilet":       {"bathroom": 0.9},
    "bathtub":      {"bathroom": 0.8},
    "shower":       {"bathroom": 0.8},
    "sink":         {"bathroom": 0.5, "kitchen": 0.5},
    "counter":      {"kitchen": 0.7},
    "oven":         {"kitchen": 0.8},
    "microwave":    {"kitchen": 0.7},
    "refrigerator": {"kitchen": 0.9},
    "bed":          {"bedroom": 0.9},
    "wardrobe":     {"bedroom": 0.7},
    "sofa":         {"living": 0.8},
    "tv":           {"living": 0.7},
    "desk":         {"office": 0.7},
    "laptop":       {"office": 0.6, "bedroom": 0.3},
    "washer":       {"laundry": 0.8},
    "dryer":        {"laundry": 0.8},
}


# ── Room name normalisation ──────────────────────────────────────────────────

def _normalize_room_for_priors(room_instance: str) -> str:
    """Return the semantic family name for a room instance.

    Strips numeric suffix (e.g. .001), then applies the semantic alias table.
    Used ONLY for matching against the manual prior / evidence tables.
    Never destroys room_instance identity — the original name is always kept
    for state tracking.

    Examples
    --------
    "tv"         → "living"
    "tv.001"     → "living"
    "bathroom"   → "bathroom"   (no alias → unchanged)
    "dining room"→ "dining"
    """
    r = _re.sub(r'\.\d+$', '', room_instance.lower().strip())
    return _SEMANTIC_ALIASES.get(r, r)


def compute_heatmap_room_evidence(kept_components: list) -> Dict[str, float]:
    """Aggregate heatmap component quality by room instance.

    Sums component quality scores (from the 'quality' field set by
    postprocess_heatmap) per room instance (from the 'room' field annotated
    by the nav system).  Normalises to [0, 1] so the strongest room scores 1.

    Returns an empty dict if no components have room annotations.
    """
    raw: Dict[str, float] = {}
    for comp in kept_components:
        room = comp.get("room")
        if room is None:
            continue
        raw[room] = raw.get(room, 0.0) + float(comp.get("quality", 0.0))
    if not raw:
        return {}
    max_q = max(raw.values())
    if max_q < 1e-6:
        return {}
    return {r: min(1.0, v / max_q) for r, v in raw.items()}


# ── LLM signal ──────────────────────────────────────────────────────────────

def _query_llm_room_prior(
    query: str, known_rooms: List[str]
) -> Dict[str, float]:
    """
    Ask GPT-4o-mini which rooms are most likely to contain *query*.
    Returns a dict {room_name: score} with values in [0, 1].
    Rooms not mentioned get 0.
    """
    try:
        import json
        import openai

        key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return {}
        client = openai.OpenAI(api_key=key)

        rooms_str = ", ".join(known_rooms)
        system = (
            "You are a spatial reasoning assistant for a home robot. "
            "Given an object name and a list of rooms, rank the rooms by how likely "
            "they are to contain the object. Reply ONLY with a JSON object mapping "
            "room names (exactly as given) to scores from 0.0 to 1.0. "
            "Include only rooms with score > 0. Example:\n"
            '{"kitchen": 0.9, "bathroom": 0.6}'
        )
        user_msg = f"Object: {query}\nRooms: {rooms_str}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        # strip markdown fences if present
        text = text.strip("`").replace("```json", "").replace("```", "").strip()
        raw = json.loads(text)
        # Validate: keep only known rooms, clamp to [0,1]
        result = {}
        for room in known_rooms:
            if room in raw:
                result[room] = float(min(1.0, max(0.0, raw[room])))
        return result
    except Exception as e:
        print(f"  [priors] LLM call failed: {e}")
        return {}


# ── Manual table lookup ──────────────────────────────────────────────────────

def _manual_prior(query: str, known_rooms: List[str]) -> Dict[str, float]:
    """
    Look up the manual table for *query* and map scores to *known_rooms*.

    Matching uses both the raw room name and its normalized semantic family
    (via _normalize_room_for_priors), so scene-specific names like 'tv' are
    correctly matched against fragments like 'living'.
    """
    q = query.lower().strip()
    # Find best matching key in table (exact first, then substring)
    table_entry: Optional[Dict[str, float]] = _MANUAL_TABLE.get(q)
    if table_entry is None:
        for key, entry in _MANUAL_TABLE.items():
            if key in q or q in key:
                table_entry = entry
                break
    if table_entry is None:
        return {}

    result: Dict[str, float] = {}
    for room in known_rooms:
        room_lower = room.lower()
        room_normalized = _normalize_room_for_priors(room_lower)
        best = 0.0
        for fragment, score in table_entry.items():
            if fragment in room_lower or fragment in room_normalized:
                best = max(best, score)
        if best > 0:
            result[room] = best
    return result


# ── Evidence from scene ──────────────────────────────────────────────────────

def _evidence_prior(
    objects_seen_by_room: Dict[str, List[str]],
    known_rooms: List[str],
) -> Dict[str, float]:
    """
    For each room, accumulate co-occurrence scores for objects already seen there.

    Matching uses both the raw room name and its normalized semantic family
    so scene-specific names like 'tv' benefit from co-occurrence signals.
    Returns scores in [0, 1].
    """
    raw: Dict[str, float] = {r: 0.0 for r in known_rooms}

    for room, seen_objs in objects_seen_by_room.items():
        if room not in raw:
            continue
        room_lower = room.lower()
        room_normalized = _normalize_room_for_priors(room_lower)
        acc = 0.0
        for obj in seen_objs:
            obj_lower = obj.lower()
            for key, fragments in _CO_OCCURRENCE.items():
                if key in obj_lower or obj_lower in key:
                    for frag, score in fragments.items():
                        if frag in room_lower or frag in room_normalized:
                            acc += score
        # Clamp per room to [0, 1]
        raw[room] = min(1.0, acc)

    return raw


# ── Main entry point ─────────────────────────────────────────────────────────


def canonical_room_type(room_instance: str) -> str:
    """Strip trailing numeric suffix to get the base room type.

    Examples:
        canonical_room_type("bathroom")      -> "bathroom"
        canonical_room_type("bathroom.001")  -> "bathroom"
        canonical_room_type("bedroom.002")   -> "bedroom"
        canonical_room_type("closet.1")      -> "closet"
    """
    return _re.sub(r'\.\d+$', '', room_instance.lower().strip())


def compatible_room_types(room_priors: Dict[str, float], threshold: float = 0.05) -> set:
    """Return the set of canonical room types with prior above threshold."""
    return {
        canonical_room_type(r)
        for r, s in room_priors.items()
        if s > threshold
    }


def compute_room_priors(
    query: str,
    known_rooms: List[str],
    objects_seen_by_room: Optional[Dict[str, List[str]]] = None,
    llm_output: Optional[Dict[str, float]] = None,
    heatmap_evidence: Optional[Dict[str, float]] = None,
    query_type: str = "indirect",
) -> Dict[str, float]:
    """
    Compute fused room priors for *query* over *known_rooms*.

    Parameters
    ----------
    query               : object being searched (e.g. "sink")
    known_rooms         : list of room names in the scene
    objects_seen_by_room: {room_name: [objects confirmed in that room]}
    llm_output          : pre-computed LLM scores {room: score} — if None,
                          will call LLM automatically
    heatmap_evidence    : {room_instance: normalised_quality} from heatmap
                          components; only used when query_type == "direct"
    query_type          : "direct"   — heatmap has signal, spatial evidence
                                       dominates (0.65 heatmap + 0.20 manual
                                       + 0.15 llm)
                          "indirect" — no heatmap signal, language priors
                                       dominate (0.45 llm + 0.30 manual
                                       + 0.25 scene-evidence)

    Returns
    -------
    {room_name: normalised_prior}  — scores sum to 1 (or all 0 if no signal)
    """
    if not known_rooms:
        return {}

    # 1. LLM signal
    if llm_output is None:
        llm_scores = _query_llm_room_prior(query, known_rooms)
    else:
        llm_scores = llm_output

    # 2. Manual table (uses semantic aliases for scene-specific room names)
    manual_scores = _manual_prior(query, known_rooms)

    # 3. Scene evidence (co-occurrence from observed objects)
    if objects_seen_by_room is None:
        objects_seen_by_room = {}
    evidence_scores = _evidence_prior(objects_seen_by_room, known_rooms)

    if heatmap_evidence is None:
        heatmap_evidence = {}

    # 4. Dynamic fusion: direct queries let heatmap evidence dominate
    if query_type == "direct" and heatmap_evidence:
        w_llm      = _W_DIRECT_LLM
        w_manual   = _W_DIRECT_MANUAL
        w_evidence = 0.0     # scene co-occurrence superseded by heatmap
        w_heatmap  = _W_DIRECT_HEATMAP
    else:
        w_llm      = _W_LLM
        w_manual   = _W_MANUAL
        w_evidence = _W_EVIDENCE
        w_heatmap  = 0.0

    # 5. Fuse
    fused: Dict[str, float] = {}
    for room in known_rooms:
        s = (
            w_llm      * llm_scores.get(room, 0.0)
            + w_manual   * manual_scores.get(room, 0.0)
            + w_evidence * evidence_scores.get(room, 0.0)
            + w_heatmap  * heatmap_evidence.get(room, 0.0)
        )
        fused[room] = min(1.0, max(0.0, s))

    # 6. Normalise
    total = sum(fused.values())
    if total > 1e-6:
        fused = {r: v / total for r, v in fused.items()}

    return fused
