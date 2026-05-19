"""
room_map_utils.py
=================
Semantic room pre-labelling from VLMap heatmaps (Option A)
with optional GPT-4o Vision refinement (Option B).

Pipeline
--------
1. For each room category, query VLMap → binary 2D mask
2. Per-cell argmax across categories → initial room segmentation
3. Connected-component analysis → structured regions with quality scores
4. (Optional) Send rendered map to GPT-4o Vision → label corrections
5. Save room_map.npy + regions.json + room_map_viz.png to scene dir
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── Default room vocabulary ───────────────────────────────────────────────────

ROOM_CATEGORIES = [
    "living room",
    "bedroom",
    "kitchen",
    "bathroom",
    "office",
    "hallway",
    "dining room",
    "laundry room",
    "closet",
    "garage",
]

# BGR colors for visualization
ROOM_COLORS_BGR = [
    (60,  100, 255),   # living room  — coral-red
    (255, 100,  60),   # bedroom      — blue
    (60,  200,  60),   # kitchen      — green
    (60,  230, 230),   # bathroom     — yellow
    (50,  140, 255),   # office       — orange
    (180, 180, 180),   # hallway      — gray
    (200,  80, 180),   # dining room  — purple
    (220, 200,  80),   # laundry      — cyan
    (100, 150, 200),   # closet       — tan
    (80,  220, 160),   # garage       — mint
]


# ── Step A: VLMap-based segmentation ─────────────────────────────────────────

def compute_room_masks(robot, categories: List[str]) -> Dict[str, np.ndarray]:
    """
    Query VLMap for each room category.
    Returns a dict {category: 2D bool mask}.
    """
    from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

    masks = {}
    for cat in categories:
        try:
            mask_3d = robot.map.index_map(cat, with_init_cat=True)
            mask_2d = pool_3d_label_to_2d(
                mask_3d, robot.map.grid_pos, robot.map.gs
            ).astype(bool)
            masks[cat] = mask_2d
            n = int(mask_2d.sum())
            print(f"    {cat:20s}  {n:5d} cells")
        except Exception as e:
            print(f"    [warn] '{cat}': {e}")
            gs = robot.map.gs
            masks[cat] = np.zeros((gs, gs), dtype=bool)
    return masks


def build_room_segmentation(
    masks: Dict[str, np.ndarray],
    min_area: int = 30,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Build room segmentation from per-category binary masks.

    Returns
    -------
    room_map : 2D int8 array  (cell value = category index, -1 = unknown)
    categories : list of category names (same order as indices in room_map)
    regions : dict {category: [{'centroid':(r,c), 'area':int, 'quality':float}]}
    """
    categories = list(masks.keys())
    gs = next(iter(masks.values())).shape[0]

    # Stack: (n_cats, gs, gs) — count how many categories claim each cell
    stack = np.stack([masks[c].astype(np.float32) for c in categories], axis=0)
    total = stack.sum(axis=0)

    # Argmax: which category has highest claim per cell
    argmax_map = stack.argmax(axis=0).astype(np.int8)

    # Cells where no category claims them → -1
    room_map = np.where(total > 0, argmax_map, np.int8(-1))

    # ── Extract connected regions per category ────────────────────────────
    regions = {}
    for i, cat in enumerate(categories):
        binary = (room_map == i).astype(np.uint8)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        cat_regions = []
        for j in range(1, n_labels):
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            # centroid as (row, col)
            cy = int(round(centroids[j][1]))
            cx = int(round(centroids[j][0]))
            # quality ∝ area (normalized) — can extend with score later
            quality = area / float(gs * gs)
            cat_regions.append({
                "centroid": [cy, cx],
                "area": area,
                "quality": quality,
            })
        cat_regions.sort(key=lambda r: r["quality"], reverse=True)
        if cat_regions:
            regions[cat] = cat_regions

    return room_map, categories, regions


# ── Step B: GPT-4o Vision refinement ─────────────────────────────────────────

def render_room_map(
    room_map: np.ndarray,
    categories: List[str],
    regions: Dict,
    rgb_bg: Optional[np.ndarray] = None,
    scale: int = 1,
) -> np.ndarray:
    """Render the room segmentation as a labeled BGR image."""
    h, w = room_map.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if rgb_bg is not None:
        canvas = (rgb_bg[:, :, ::-1] * 0.35).astype(np.uint8)  # dim RGB bg

    for i, cat in enumerate(categories):
        color = ROOM_COLORS_BGR[i % len(ROOM_COLORS_BGR)]
        canvas[room_map == i] = color

    # Draw centroid dots and labels
    for cat, regs in regions.items():
        i = categories.index(cat)
        color = ROOM_COLORS_BGR[i % len(ROOM_COLORS_BGR)]
        for reg in regs[:1]:   # label only the best region
            r, c = reg["centroid"]
            cv2.circle(canvas, (c, r), 5, (255, 255, 255), -1)
            cv2.putText(canvas, cat, (c + 7, r + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
                        cv2.LINE_AA)

    if scale > 1:
        canvas = cv2.resize(canvas, (w * scale, h * scale),
                            interpolation=cv2.INTER_NEAREST)
    return canvas


def refine_with_gpt4v(
    render_img: np.ndarray,
    categories: List[str],
    regions: Dict,
    openai_key: str,
) -> Dict:
    """
    Send the rendered room map to GPT-4o Vision for label validation.

    Returns a dict:
      {
        "corrections": [{"from": "old", "to": "new"}, ...],
        "confidence": "high|medium|low",
        "notes": "..."
      }
    """
    import base64
    import openai

    client = openai.OpenAI(api_key=openai_key)

    # Encode image
    _, buf = cv2.imencode(".png", render_img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Build region summary for the prompt
    lines = []
    for cat, regs in regions.items():
        if regs:
            r = regs[0]
            lines.append(
                f"  - {cat}: centroid=({r['centroid'][0]},{r['centroid'][1]})"
                f", area={r['area']} cells"
            )
    region_text = "\n".join(lines) if lines else "  (no regions detected)"

    prompt = (
        "This is a top-down 2D semantic map of an indoor residential environment. "
        "The colored regions were automatically labelled with the following room types:\n"
        f"{region_text}\n\n"
        "Based on the spatial layout and shape of the regions, do the labels look "
        "correct for a typical home? "
        "Reply ONLY with a JSON object (no markdown) in this exact format:\n"
        '{"corrections": [{"from": "mislabelled_room", "to": "correct_room"}], '
        '"confidence": "high", "notes": "brief comment"}\n'
        "If everything looks correct, return an empty corrections list. "
        f"Valid room names: {categories}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=400,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        text = text.strip("`").replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [warn] GPT-4o refinement failed: {e}")
        return {"corrections": [], "confidence": "low", "notes": str(e)}


def apply_corrections(
    room_map: np.ndarray,
    regions: Dict,
    categories: List[str],
    corrections: List[Dict],
) -> Tuple[np.ndarray, Dict]:
    """Apply GPT-4o label corrections to room_map and regions."""
    corrected_map = room_map.copy()
    corrected_regions = {k: list(v) for k, v in regions.items()}

    for corr in corrections:
        old = corr.get("from", "").strip()
        new = corr.get("to", "").strip()
        if old not in categories or new not in categories:
            print(f"  [skip] Unknown label in correction: '{old}' → '{new}'")
            continue
        old_idx = categories.index(old)
        new_idx = categories.index(new)
        corrected_map[corrected_map == old_idx] = new_idx
        if old in corrected_regions:
            corrected_regions.setdefault(new, []).extend(
                corrected_regions.pop(old)
            )
        print(f"  Corrected: '{old}' → '{new}'")

    return corrected_map, corrected_regions


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_room_map(
    save_dir: Path,
    room_map: np.ndarray,
    categories: List[str],
    regions: Dict,
    viz_img: Optional[np.ndarray] = None,
) -> None:
    save_dir = Path(save_dir) / "room_map"
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "room_map.npy", room_map)
    with open(save_dir / "regions.json", "w") as f:
        json.dump({"categories": categories, "regions": regions}, f, indent=2)
    if viz_img is not None:
        cv2.imwrite(str(save_dir / "room_map_viz.png"), viz_img)
    print(f"  Saved room map to {save_dir}")


def load_room_map(scene_dir: Path) -> Optional[Tuple[np.ndarray, List[str], Dict]]:
    """Load a previously built room map. Returns None if not found."""
    room_dir = Path(scene_dir) / "room_map"
    map_file = room_dir / "room_map.npy"
    reg_file = room_dir / "regions.json"

    if not map_file.exists() or not reg_file.exists():
        return None

    room_map = np.load(map_file)
    with open(reg_file) as f:
        data = json.load(f)
    return room_map, data["categories"], data["regions"]


# ── Navigation helper ─────────────────────────────────────────────────────────

def find_room_goal(
    query: str,
    regions: Dict,
) -> Optional[Tuple[int, int]]:
    """
    Given a natural-language query, return the best (row, col) navigation goal
    from the pre-labelled room regions.

    Tries exact match first, then substring match.
    Returns None if no region matches.
    """
    q = query.lower().strip()

    # Exact match
    if q in regions and regions[q]:
        centroid = regions[q][0]["centroid"]
        return (centroid[0], centroid[1])

    # Substring match (e.g. "office area" → "office")
    for room_name, regs in regions.items():
        if q in room_name or room_name in q:
            if regs:
                centroid = regs[0]["centroid"]
                return (centroid[0], centroid[1])

    return None
