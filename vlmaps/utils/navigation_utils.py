import heapq

import numpy as np
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
import pyvisgraph as vg
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict


def _snap_to_nearest_free(point, obstacles: np.ndarray, min_clearance: float = 2.0):
    """Return the nearest free cell to *point* that also has >= min_clearance.

    Uses the distance transform so the result is guaranteed navigable in the
    same map the visgraph was built from.  Falls back to nearest free cell
    (clearance=0 ok) if no cell with the requested clearance exists.

    Parameters
    ----------
    point : (row, col) — may be inside an obstacle.
    obstacles : 2-D uint8 array, 1=free 0=obstacle (cropped map space).
    min_clearance : desired minimum clearance in cells.

    Returns
    -------
    (row, col) tuple guaranteed to be inside the map bounds and free.
    """
    free_mask = (obstacles == 1)
    if free_mask.sum() == 0:
        return (int(point[0]), int(point[1]))  # nothing we can do

    # Distance of every free cell from the nearest obstacle
    dist_map = distance_transform_edt(free_mask)

    rows, cols = np.where(free_mask)
    dist_sq = (rows - point[0]) ** 2 + (cols - point[1]) ** 2

    # Try to find the nearest free cell that also has enough clearance
    cl = dist_map[rows, cols]
    good_mask = cl >= min_clearance
    if good_mask.any():
        idx = np.argmin(np.where(good_mask, dist_sq, np.inf))
    else:
        # No cell with that clearance — just take the nearest free cell
        idx = np.argmin(dist_sq)

    return (int(rows[idx]), int(cols[idx]))


def _clip_cell(point, shape: Tuple[int, int]) -> Tuple[int, int]:
    """Clamp a floating-point cell position to valid integer grid bounds."""
    h, w = shape
    row = int(np.clip(np.round(point[0]), 0, h - 1))
    col = int(np.clip(np.round(point[1]), 0, w - 1))
    return (row, col)


def _disk_cells(center: Tuple[int, int], radius: int, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Enumerate integer cells inside a disk centered at *center*."""
    row, col = center
    rmin = max(0, row - radius)
    rmax = min(shape[0] - 1, row + radius)
    cmin = max(0, col - radius)
    cmax = min(shape[1] - 1, col + radius)
    cells = []
    rad_sq = radius * radius
    for rr in range(rmin, rmax + 1):
        dr_sq = (rr - row) * (rr - row)
        for cc in range(cmin, cmax + 1):
            if dr_sq + (cc - col) * (cc - col) <= rad_sq:
                cells.append((rr, cc))
    return cells


def _segment_is_free(
    start: Tuple[int, int],
    end: Tuple[int, int],
    free_map: np.ndarray,
) -> bool:
    """Return True if the rasterized line segment lies entirely in free space."""
    start = _clip_cell(start, free_map.shape)
    end = _clip_cell(end, free_map.shape)

    rmin = min(start[0], end[0])
    rmax = max(start[0], end[0])
    cmin = min(start[1], end[1])
    cmax = max(start[1], end[1])

    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.uint8)
    cv2.line(
        mask,
        (start[1] - cmin, start[0] - rmin),
        (end[1] - cmin, end[0] - rmin),
        1,
        1,
    )
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return bool(free_map[start[0], start[1]]) and bool(free_map[end[0], end[1]])
    return bool(np.all(free_map[rmin + rows, cmin + cols] == 1))


def _segment_cells(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Rasterize a grid segment into integer cells, including endpoints."""
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))

    rmin = min(start[0], end[0])
    rmax = max(start[0], end[0])
    cmin = min(start[1], end[1])
    cmax = max(start[1], end[1])

    mask = np.zeros((rmax - rmin + 1, cmax - cmin + 1), dtype=np.uint8)
    cv2.line(
        mask,
        (start[1] - cmin, start[0] - rmin),
        (end[1] - cmin, end[0] - rmin),
        1,
        1,
    )
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return [start, end] if start != end else [start]
    pts = [(int(rmin + rr), int(cmin + cc)) for rr, cc in zip(rows, cols)]
    pts.sort(key=lambda cell: (cell[0] - start[0]) ** 2 + (cell[1] - start[1]) ** 2)
    return pts


def _find_best_centered_cell(
    waypoint: Tuple[int, int],
    prev_waypoint: Tuple[int, int],
    next_waypoint: Tuple[int, int],
    dist_map: np.ndarray,
    support_map: np.ndarray,
    search_radius: int,
) -> Tuple[int, int]:
    """Find the highest-clearance nearby cell that preserves local connectivity."""
    waypoint = _clip_cell(waypoint, dist_map.shape)
    prev_waypoint = _clip_cell(prev_waypoint, dist_map.shape)
    next_waypoint = _clip_cell(next_waypoint, dist_map.shape)

    base_len = float(np.linalg.norm(np.subtract(prev_waypoint, waypoint))) + float(
        np.linalg.norm(np.subtract(next_waypoint, waypoint))
    )
    best_cell = waypoint
    best_key = (
        float(dist_map[waypoint[0], waypoint[1]]),
        -0.0,
        -0.0,
    )

    for cand in _disk_cells(waypoint, search_radius, dist_map.shape):
        if not support_map[cand[0], cand[1]]:
            continue
        if not _segment_is_free(prev_waypoint, cand, support_map):
            continue
        if not _segment_is_free(cand, next_waypoint, support_map):
            continue

        detour = (
            float(np.linalg.norm(np.subtract(prev_waypoint, cand)))
            + float(np.linalg.norm(np.subtract(next_waypoint, cand)))
            - base_len
        )
        disp = float(np.linalg.norm(np.subtract(cand, waypoint)))
        cand_key = (
            float(dist_map[cand[0], cand[1]]),
            -detour,
            -disp,
        )
        if cand_key > best_key:
            best_cell = cand
            best_key = cand_key

    return best_cell


def _smooth_centered_path(
    path: List[List[float]],
    shifted_flags: List[bool],
    dist_map: np.ndarray,
    support_map: np.ndarray,
    snap_radius: int = 2,
) -> List[List[float]]:
    """Apply a conservative 3-point smoothing pass on already-centered waypoints."""
    if len(path) < 3:
        return path

    smoothed = [list(point) for point in path]
    for i in range(1, len(path) - 1):
        if not shifted_flags[i]:
            continue
        prev_cell = _clip_cell(smoothed[i - 1], dist_map.shape)
        curr_cell = _clip_cell(smoothed[i], dist_map.shape)
        next_cell = _clip_cell(smoothed[i + 1], dist_map.shape)

        weighted = (
            0.25 * np.asarray(prev_cell, dtype=np.float32)
            + 0.5 * np.asarray(curr_cell, dtype=np.float32)
            + 0.25 * np.asarray(next_cell, dtype=np.float32)
        )
        target = _clip_cell(weighted, dist_map.shape)
        best_cell = curr_cell
        best_key = (
            -float(np.linalg.norm(np.subtract(curr_cell, target))),
            float(dist_map[curr_cell[0], curr_cell[1]]),
        )

        for cand in _disk_cells(target, snap_radius, dist_map.shape):
            if not support_map[cand[0], cand[1]]:
                continue
            if not _segment_is_free(prev_cell, cand, support_map):
                continue
            if not _segment_is_free(cand, next_cell, support_map):
                continue
            # Smoothing should not undo the gained clearance in tight spaces.
            if dist_map[cand[0], cand[1]] + 0.25 < dist_map[curr_cell[0], curr_cell[1]]:
                continue

            cand_key = (
                -float(np.linalg.norm(np.subtract(cand, target))),
                float(dist_map[cand[0], cand[1]]),
            )
            if cand_key > best_key:
                best_cell = cand
                best_key = cand_key

        smoothed[i] = [int(best_cell[0]), int(best_cell[1])]

    return smoothed


def _insert_narrow_passage_waypoints(
    path: List[List[float]],
    dist_map: np.ndarray,
    support_map: np.ndarray,
    narrow_th: float,
    search_radius: int,
) -> Tuple[List[List[float]], int, List[bool]]:
    """Insert helper waypoints where a long segment crosses a narrow passage."""
    if len(path) < 2:
        return [list(point) for point in path], 0, [False] * len(path)

    augmented = [[int(path[0][0]), int(path[0][1])]]
    inserted_flags = [False]
    inserted_count = 0

    # Minimum segment length (cells) to insert a centering waypoint.
    # Short segments (< MIN_SEG) already have their endpoints close to the bottleneck;
    # adding an intermediate waypoint just creates more heading corrections.
    _MIN_SEG_FOR_INSERTION = 10

    for i in range(len(path) - 1):
        seg_start = _clip_cell(path[i], dist_map.shape)
        seg_end = _clip_cell(path[i + 1], dist_map.shape)
        cells = _segment_cells(seg_start, seg_end)

        # Skip insertion for short segments — the visgraph vertex is already close
        # to the bottleneck, so a mid-segment waypoint would only add corrections.
        if len(cells) < _MIN_SEG_FOR_INSERTION:
            end_cell = [int(seg_end[0]), int(seg_end[1])]
            if end_cell != augmented[-1]:
                augmented.append(end_cell)
                inserted_flags.append(False)
            continue

        run_start = None
        for idx, cell in enumerate(cells):
            is_narrow = float(dist_map[cell[0], cell[1]]) < narrow_th
            if is_narrow and run_start is None:
                run_start = idx
            is_run_end = run_start is not None and (not is_narrow or idx == len(cells) - 1)
            if not is_run_end:
                continue

            run_end = idx if is_narrow and idx == len(cells) - 1 else idx - 1
            run_cells = cells[run_start:run_end + 1]
            narrow_cell = min(run_cells, key=lambda cell_: float(dist_map[cell_[0], cell_[1]]))
            centered_cell = _find_best_centered_cell(
                narrow_cell,
                seg_start,
                seg_end,
                dist_map,
                support_map,
                search_radius,
            )
            if centered_cell != augmented[-1] and centered_cell != seg_end:
                augmented.append([int(centered_cell[0]), int(centered_cell[1])])
                inserted_flags.append(True)
                inserted_count += 1
            run_start = None

        end_cell = [int(seg_end[0]), int(seg_end[1])]
        if end_cell != augmented[-1]:
            augmented.append(end_cell)
            inserted_flags.append(False)

    return augmented, inserted_count, inserted_flags


def center_path_on_medial_axis(
    path: List[List[float]],
    dist_map: np.ndarray,
    support_map: np.ndarray = None,
    narrow_th: float = 4.0,
    search_radius: int = 6,
    smooth: bool = True,
    tight_clearance_th: float = 1.5,
) -> List[List[float]]:
    """Shift narrow-passage waypoints toward higher-clearance cells.

    The path is assumed to be valid in *support_map* (typically the planner's
    dilated free-space map). Clearance is measured in *dist_map* computed from
    the raw obstacle map so doorway centering reflects physical wall distance.
    """
    if len(path) < 3:
        return [list(point) for point in path]

    if support_map is None:
        support_map = (dist_map > 0).astype(np.uint8)

    centered, inserted_count, inserted_flags = _insert_narrow_passage_waypoints(
        path,
        dist_map,
        support_map,
        narrow_th,
        search_radius,
    )
    shifted_flags = list(inserted_flags)
    shifted_count = 0
    tight_count = 0

    for i in range(1, len(centered) - 1):
        waypoint = _clip_cell(centered[i], dist_map.shape)
        clearance = float(dist_map[waypoint[0], waypoint[1]])
        if clearance >= narrow_th:
            centered[i] = [int(waypoint[0]), int(waypoint[1])]
            continue

        prev_waypoint = _clip_cell(centered[i - 1], dist_map.shape)
        next_waypoint = _clip_cell(centered[i + 1], dist_map.shape)
        best_cell = _find_best_centered_cell(
            waypoint,
            prev_waypoint,
            next_waypoint,
            dist_map,
            support_map,
            search_radius,
        )
        best_clearance = float(dist_map[best_cell[0], best_cell[1]])
        centered[i] = [int(best_cell[0]), int(best_cell[1])]

        if best_cell != waypoint:
            shifted_flags[i] = True
            shifted_count += 1
        if best_clearance < tight_clearance_th:
            tight_count += 1

    if smooth:
        centered = _smooth_centered_path(centered, shifted_flags, dist_map, support_map)

    if inserted_count > 0 or shifted_count > 0:
        print(
            f"[planner] Medial-axis centered {shifted_count} waypoint(s), "
            f"inserted {inserted_count} narrow-passage waypoint(s) "
            f"(narrow_th={narrow_th:.1f}, radius={search_radius})"
        )
    if tight_count > 0:
        print(
            f"[planner] WARNING: {tight_count} centered waypoint(s) remain below "
            f"{tight_clearance_th:.1f} cells of raw clearance"
        )

    return centered


def _segment_min_clearance(
    start: Tuple[int, int],
    end: Tuple[int, int],
    dist_map: np.ndarray,
) -> float:
    """Return the minimum clearance value along the rasterized segment."""
    cells = _segment_cells(start, end)
    if not cells:
        return float("inf")
    h, w = dist_map.shape
    min_cl = float("inf")
    for r, c in cells:
        cl = float(dist_map[int(np.clip(r, 0, h - 1)), int(np.clip(c, 0, w - 1))])
        if cl < min_cl:
            min_cl = cl
    return min_cl


def _shortcut_path_clearance(
    path: List[List[float]],
    free_map: np.ndarray,
    dist_map: np.ndarray,
    min_shortcut_clearance: float = 1.5,
) -> List[List[float]]:
    """Reduce A* path by skipping intermediate waypoints when the direct segment
    is free and maintains at least *min_shortcut_clearance* raw clearance.

    This converts the dense grid path into a compact polyline while preserving
    the clearance property of the A* solution.
    """
    if len(path) < 3:
        return [list(p) for p in path]

    result = [list(path[0])]
    i = 0
    while i < len(path) - 1:
        # Scan from the current waypoint toward the end; take the farthest
        # reachable waypoint with both free line-of-sight and adequate clearance.
        j = len(path) - 1
        while j > i + 1:
            a = _clip_cell(result[-1], dist_map.shape)
            b = _clip_cell(path[j], dist_map.shape)
            if (
                _segment_is_free(a, b, free_map)
                and _segment_min_clearance(a, b, dist_map) >= min_shortcut_clearance
            ):
                break
            j -= 1
        result.append(list(path[j]))
        i = j

    return result


def plan_clearance_aware_astar(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    free_map: np.ndarray,
    dist_map: np.ndarray,
    clearance_weight: float = 2.0,
    min_shortcut_clearance: float = 1.5,
) -> List[List[float]]:
    """Plan a path from *start* to *goal* that explicitly trades path length
    for clearance from obstacles.

    Parameters
    ----------
    start, goal : (row, col) in cropped-map coordinates.
    free_map    : 2-D uint8 array — 1=free, 0=obstacle (dilated planning map).
    dist_map    : 2-D float array — EDT of the RAW obstacle map; gives physical
                  clearance in cells. High values = far from walls.
    clearance_weight : λ in the per-cell cost  ``1 + λ/(cl + 0.5)``.
                  Higher → stronger preference for high-clearance cells.
    min_shortcut_clearance : minimum clearance preserved during post-A* shortcutting.

    Returns
    -------
    Compact list of [row, col] waypoints in cropped-map coordinates.
    Falls back to [start, goal] if no path is found.
    """
    h, w = free_map.shape
    sr = int(np.clip(round(start[0]), 0, h - 1))
    sc = int(np.clip(round(start[1]), 0, w - 1))
    gr = int(np.clip(round(goal[0]),  0, h - 1))
    gc = int(np.clip(round(goal[1]),  0, w - 1))

    # Snap start / goal to nearest free cell if they land in an obstacle.
    if not free_map[sr, sc]:
        snap = _snap_to_nearest_free((sr, sc), free_map, min_clearance=1.0)
        sr, sc = snap
    if not free_map[gr, gc]:
        snap = _snap_to_nearest_free((gr, gc), free_map, min_clearance=2.0)
        gr, gc = snap

    if (sr, sc) == (gr, gc):
        return [[sr, sc]]

    # Pre-compute cost map: traversing a cell costs (1 + λ/(cl+0.5)).
    # Cells with higher clearance are cheaper → A* naturally prefers them.
    cl_map = dist_map.astype(np.float32)
    cost_map = (1.0 + clearance_weight / (cl_map + 0.5)).astype(np.float32)

    # 8-connected A* with diagonal step-cost √2.
    _DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    _DIAG = {(-1,-1),(-1,1),(1,-1),(1,1)}

    def heuristic(r: int, c: int) -> float:
        return float(np.hypot(r - gr, c - gc))

    g_score: dict = {(sr, sc): 0.0}
    came_from: dict = {}
    # heap entries: (f, g, row, col)
    heap = [(heuristic(sr, sc), 0.0, sr, sc)]

    while heap:
        f, g, r, c = heapq.heappop(heap)
        if (r, c) == (gr, gc):
            break
        if g > g_score.get((r, c), float("inf")) + 1e-9:
            continue
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not free_map[nr, nc]:
                continue
            step = (1.41421356 if (dr, dc) in _DIAG else 1.0)
            new_g = g + step * float(cost_map[nr, nc])
            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

    # Reconstruct dense path
    if (gr, gc) not in came_from and (gr, gc) != (sr, sc):
        print(f"[astar] WARNING: no path found from {(sr,sc)} to {(gr,gc)} — falling back")
        return [[sr, sc], [gr, gc]]

    dense: list = []
    cur = (gr, gc)
    while cur in came_from:
        dense.append(list(cur))
        cur = came_from[cur]
    dense.append([sr, sc])
    dense.reverse()

    # Reduce dense path to compact waypoints while preserving clearance.
    compact = _shortcut_path_clearance(dense, free_map, dist_map, min_shortcut_clearance)

    print(
        f"[astar] Path: {len(dense)} dense cells → {len(compact)} waypoints "
        f"(clearance_weight={clearance_weight:.1f})"
    )
    return compact


def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy


def find_closest_points_between_two_contours(obs_map, contour_a, contour_b):
    a = np.zeros_like(obs_map, dtype=np.uint8)
    b = np.zeros_like(obs_map, dtype=np.uint8)
    cv2.drawContours(a, [contour_a[:, [1, 0]]], 0, 255, 1)
    cv2.drawContours(b, [contour_b[:, [1, 0]]], 0, 255, 1)
    rows_a, cols_a = np.where(a == 255)
    rows_b, cols_b = np.where(b == 255)
    pts_a = np.concatenate([rows_a.reshape((-1, 1)), cols_a.reshape((-1, 1))], axis=1)
    pts_b = np.concatenate([rows_b.reshape((-1, 1)), cols_b.reshape((-1, 1))], axis=1)
    dists = cdist(pts_a, pts_b)
    id = np.argmin(dists)
    ida, idb = np.unravel_index(id, dists.shape)
    return [rows_a[ida], cols_a[ida]], [rows_b[idb], cols_b[idb]]


def point_in_contours(obs_map, contours_list, point):
    """
    obs_map: np.ndarray, 1 free, 0 occupied
    contours_list: a list of cv2 contours [[(col1, row1), (col2, row2), ...], ...]
    point: (row, col)
    """
    row, col = int(point[0]), int(point[1])
    ids = []
    print("contours num: ", len(contours_list))
    for con_i, contour in enumerate(contours_list):
        contour_cv2 = contour[:, [1, 0]]
        con_mask = np.zeros_like(obs_map, dtype=np.uint8)
        cv2.drawContours(con_mask, [contour_cv2], 0, 255, -1)
        # con_mask_copy = con_mask.copy()
        # cv2.circle(con_mask_copy, (col, row), 10, 0, 3)
        # cv2.imshow("contour_mask", con_mask_copy)
        # cv2.waitKey()
        if con_mask[row, col] == 255:
            ids.append(con_i)

    return ids


def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
    if vis:
        cv2.imshow("obs", obs_map_vis)
        cv2.waitKey()

    contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    if use_internal_contour:
        ids = point_in_contours(obs_map, contours_list, internal_point)
        assert len(ids) == 2, f"The internal point is not in 2 contours, but {len(ids)}"
        point_a, point_b = find_closest_points_between_two_contours(
            obs_map, contours_list[ids[0]], contours_list[ids[1]]
        )
        obs_map = cv2.line((obs_map * 255).astype(np.uint8), (point_a[1], point_a[0]), (point_b[1], point_b[0]), 255, 5)
        obs_map = obs_map == 255
        contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
            obs_map, 0, detect_internal_contours=False
        )

    poly_list = []

    for contour in contours_list:
        if vis:
            contour_cv2 = contour[:, [1, 0]]
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
            cv2.imshow("obs", obs_map_vis)
        contour_pos = []
        for [row, col] in contour:
            contour_pos.append(vg.Point(row, col))
        poly_list.append(contour_pos)
        xlist = [x.x for x in contour_pos]
        zlist = [x.y for x in contour_pos]
        if vis:
            # plt.plot(xlist, zlist)

            cv2.waitKey()
    g = vg.VisGraph()
    g.build(poly_list, workers=4)
    return g


def get_nearby_position(goal: Tuple[float, float], G: vg.VisGraph) -> Tuple[float, float]:
    for dr, dc in zip([-1, 1, -1, 1], [-1, -1, 1, 1]):
        goalvg_new = vg.Point(goal[0] + dr, goal[1] + dc)
        poly_id_new = G.point_in_polygon(goalvg_new)
        if poly_id_new == -1:
            return (goal[0] + dr, goal[1] + dc)


def plan_to_pos_v2(start, goal, obstacles, G: vg.VisGraph = None, vis=False):
    """
    plan a path on a cropped obstacles map represented by a graph.
    Start and goal are tuples of (row, col) in the map.
    """

    print("start: ", start)
    print("goal: ", goal)

    # Clamp indices to valid map bounds before any indexing
    h, w = obstacles.shape
    start_r = int(np.clip(start[0], 0, h - 1))
    start_c = int(np.clip(start[1], 0, w - 1))
    goal_r  = int(np.clip(goal[0],  0, h - 1))
    goal_c  = int(np.clip(goal[1],  0, w - 1))

    start_nav = bool(obstacles[start_r, start_c])
    goal_nav  = bool(obstacles[goal_r,  goal_c])
    print(f"[planner] Start navigable: {start_nav}  Goal navigable: {goal_nav}")

    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
        obs_map_vis = cv2.circle(obs_map_vis, (start_c, start_r), 3, (255, 0, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (goal_c,  goal_r),  3, (0, 0, 255), -1)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    path = []

    # ── Start snapping ───────────────────────────────────────────────────
    if not start_nav:
        print("[planner] Start in obstacle — snapping to nearest free cell")
        new_start = _snap_to_nearest_free((start_r, start_c), obstacles, min_clearance=2.0)
        print(f"[planner] Snapped start: {new_start}")
        path.append(list(new_start))
        startvg = vg.Point(new_start[0], new_start[1])
    else:
        startvg = vg.Point(start_r, start_c)

    # ── Goal snapping ────────────────────────────────────────────────────
    if not goal_nav:
        print("[planner] Goal in obstacle — snapping to nearest free cell with clearance")
        new_goal = _snap_to_nearest_free((goal_r, goal_c), obstacles, min_clearance=3.0)
        print(f"[planner] Snapped goal: {new_goal}")
        goalvg = vg.Point(new_goal[0], new_goal[1])
    else:
        goalvg = vg.Point(goal_r, goal_c)

    path_vg = G.shortest_path(startvg, goalvg)

    # Validate: shortest_path should return >= 2 waypoints for a real path.
    # If start==goal the list may have 1 entry, which is fine.
    if not path_vg:
        print("[planner] WARNING: shortest_path returned empty path — check map connectivity")
        return path

    for point in path_vg:
        subgoal = [point.x, point.y]
        path.append(subgoal)
    print(path)

    # check the final goal is not in obstacles
    # if obstacles[int(goal[0]), int(goal[1])] == 0:
    #     path = path[:-1]

    if vis:
        obs_map_vis = (obstacles[:, :, None] * 255).astype(np.uint8)
        obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])

        for i, point in enumerate(path):
            subgoal = (int(point[1]), int(point[0]))
            print(i, subgoal)
            obs_map_vis = cv2.circle(obs_map_vis, subgoal, 5, (255, 0, 0), -1)
            if i > 0:
                cv2.line(obs_map_vis, last_subgoal, subgoal, (255, 0, 0), 2)
            last_subgoal = subgoal
        obs_map_vis = cv2.circle(obs_map_vis, (int(start[1]), int(start[0])), 5, (0, 255, 0), -1)
        obs_map_vis = cv2.circle(obs_map_vis, (int(goal[1]), int(goal[0])), 5, (0, 0, 255), -1)

        seg = Image.fromarray(obs_map_vis)
        cv2.imshow("planned path", obs_map_vis)
        cv2.waitKey()

    return path


def get_bbox(center, size):
    """
    Return min corner and max corner coordinate
    """
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner


def get_dist_to_bbox_2d(center, size, pos):
    min_corner_2d, max_corner_2d = get_bbox(center, size)

    dx = pos[0] - center[0]
    dy = pos[1] - center[1]

    if pos[0] < min_corner_2d[0] or pos[0] > max_corner_2d[0]:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
            *  |  |  *
            ___|__|___
               |  |
            ___|__|___
               |  |
            *  |  |  *
            """

            dx_c = np.abs(dx) - size[0] / 2
            dy_c = np.abs(dy) - size[1] / 2
            dist = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            return dist
        else:
            """
            star region
               |  |
            ___|__|___
            *  |  |  *
            ___|__|___
               |  |
               |  |
            """
            dx_b = np.abs(dx) - size[0] / 2
            return dx_b
    else:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
               |* |
            ___|__|___
               |  |
            ___|__|___
               |* |
               |  |
            """
            dy_b = np.abs(dy) - size[1] / 2
            return dy_b

        """
        star region
           |  |  
        ___|__|___
           |* |   
        ___|__|___
           |  |   
           |  |  
        """
        return 0
