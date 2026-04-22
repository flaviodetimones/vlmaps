"""
object_priors.py
================
Phase G — level-3 object support: object → surrogate furniture mapping.

The VLMap is built over a small furniture vocabulary (~36 HSSD categories).
When the user asks for a specific small object such as "bottle", "cup" or
"laptop", the heatmap for that exact label is empty: the object is not part
of the map's category set. The room-aware policy still has a path to find
it though: navigate to the *kind of furniture where that object usually
lives* (a "surrogate" furniture category that IS in the VLMap), arrive
close to it, and then verify the actual object visually with YOLOE.

This module exposes a deterministic, hand-curated mapping from common
small objects to a ranked list of surrogate furniture categories. It is
intentionally tiny and easy to audit. It is consulted by the strategic
policy (Phase G) when the requested target is not available in the
robot's VLMap categories.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


# Hand-curated, ordered lists of surrogate furniture for common small objects.
# Each entry says: "if the user asks for this object, search on top of these
# furniture types (in order of preference)". Categories used here MUST match
# the HSSD furniture vocabulary in `matterport3d_categories.hssd_categories`.
_OBJECT_TO_SURROGATES: Dict[str, List[str]] = {
    "bottle":     ["counter", "table", "refrigerator", "shelf", "cabinet"],
    "cup":        ["table", "counter", "shelf", "cabinet"],
    "mug":        ["table", "counter", "shelf", "cabinet"],
    "glass":      ["table", "counter", "shelf"],
    "plate":      ["table", "counter", "shelf", "cabinet"],
    "bowl":       ["table", "counter", "shelf", "cabinet"],
    "fork":       ["table", "counter"],
    "knife":      ["counter", "table"],
    "spoon":      ["table", "counter"],
    "pan":        ["counter", "oven", "shelf"],
    "pot":        ["counter", "oven", "shelf"],
    "kettle":     ["counter", "table"],
    "laptop":     ["desk", "table", "sofa", "bed"],
    "book":       ["shelf", "shelving", "table", "desk"],
    "remote":     ["sofa", "table", "tv"],
    "phone":      ["table", "desk", "sofa", "bed"],
    "keys":       ["table", "counter", "cabinet"],
    "wallet":     ["table", "counter", "cabinet"],
    "backpack":   ["sofa", "chair", "bed", "table"],
    "bag":        ["sofa", "chair", "bed", "table"],
    "ball":       ["sofa", "floor", "chair"],
    "box":        ["table", "counter", "shelf", "floor"],
    "vase":       ["table", "shelf", "counter"],
    "lamp":       ["table", "desk", "shelf"],
    "clock":      ["wall", "shelf", "table"],
    "toothbrush": ["sink", "counter"],
    "soap":       ["sink", "shower", "bathtub"],
    "towel":      ["bathtub", "shower", "sink"],
    "candle":     ["table", "shelf", "counter"],
    "plant":      ["table", "shelf", "floor"],
}


def has_surrogates(target: str) -> bool:
    """Return True iff the target has a hand-curated surrogate list."""
    return target.strip().lower() in _OBJECT_TO_SURROGATES


def get_surrogates(target: str) -> List[str]:
    """Return the ordered list of surrogate furniture for ``target`` (possibly empty)."""
    return list(_OBJECT_TO_SURROGATES.get(target.strip().lower(), []))


def pick_surrogate(
    target: str,
    available_categories: Sequence[str],
) -> Tuple[str, List[str]]:
    """Pick the first surrogate that is actually available in the VLMap vocabulary.

    Returns ``(chosen, considered)``:
      - ``chosen`` is the first surrogate from the curated list that also appears
        in ``available_categories`` (case-insensitive). Empty string if none match.
      - ``considered`` is the full ranked list looked up for ``target``, useful
        for telemetry / logging.
    """
    considered = get_surrogates(target)
    if not considered:
        return "", []
    available_lower = {c.strip().lower() for c in available_categories if c}
    for cand in considered:
        if cand in available_lower:
            return cand, considered
    return "", considered
