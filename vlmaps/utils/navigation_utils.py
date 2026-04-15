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
