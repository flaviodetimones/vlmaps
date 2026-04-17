import numpy as np
from scipy.ndimage import distance_transform_edt
from vlmaps.utils.navigation_utils import plan_clearance_aware_astar
from typing import Tuple, List


class Navigator:
    def __init__(self):
        pass

    def build_visgraph(
        self,
        obstacle_map: np.ndarray,
        rowmin: float,
        colmin: float,
        vis: bool = False,
        raw_obstacle_map: np.ndarray = None,
    ):
        """Store the planning maps and precompute the raw distance transform.

        The visgraph is no longer built here — the clearance-aware A* planner
        operates directly on the grid maps, so pyvisgraph is not needed.
        The method name is kept for interface compatibility.
        """
        self.obs_map = obstacle_map
        self.raw_obs_map = raw_obstacle_map if raw_obstacle_map is not None else obstacle_map
        self.raw_dist_map = distance_transform_edt(self.raw_obs_map)
        self.rowmin = rowmin
        self.colmin = colmin
        print(
            f"[navigator] Clearance-aware A* ready — "
            f"map {obstacle_map.shape}, "
            f"free cells: {int(obstacle_map.sum())}"
        )

    def plan_to(
        self, start_full_map: Tuple[float, float], goal_full_map: Tuple[float, float], vis: bool = False
    ) -> List[List[float]]:
        """Plan from start to goal using clearance-aware A*.

        The path naturally routes through high-clearance areas (doorway centers,
        corridor midlines) without any post-processing centering step.

        Parameters
        ----------
        start_full_map, goal_full_map : (row, col) in full map coordinates.

        Returns
        -------
        List of [row, col] waypoints in full map coordinates.
        """
        start = self._convert_full_map_pos_to_cropped_map_pos(start_full_map)
        goal = self._convert_full_map_pos_to_cropped_map_pos(goal_full_map)
        h, w = self.obs_map.shape[:2]
        goal = [
            max(0, min(goal[0], h - 1)),
            max(0, min(goal[1], w - 1)),
        ]
        paths = plan_clearance_aware_astar(
            start,
            goal,
            free_map=self.obs_map,
            dist_map=self.raw_dist_map,
            clearance_weight=2.0,
            min_shortcut_clearance=3.0,
        )
        if paths:
            path_cls = [
                float(
                    self.raw_dist_map[
                        int(np.clip(p[0], 0, self.raw_dist_map.shape[0] - 1)),
                        int(np.clip(p[1], 0, self.raw_dist_map.shape[1] - 1)),
                    ]
                )
                for p in paths
            ]
            print(
                f"[navigator] Path clearances: min={min(path_cls):.1f} "
                f"mean={float(np.mean(path_cls)):.1f} waypoints={len(paths)}"
            )
        paths = self.shift_path(paths, self.rowmin, self.colmin)
        return paths

    def shift_path(self, paths: List[List[float]], row_shift: int, col_shift: int) -> List[List[float]]:
        shifted_paths = []
        for point in paths:
            shifted_paths.append([point[0] + row_shift, point[1] + col_shift])
        return shifted_paths

    def _convert_full_map_pos_to_cropped_map_pos(self, full_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """full_map_pos: (row, col) in full map → (row, col) in cropped map."""
        print("full_map_pos: ", full_map_pos)
        print("self.rowmin: ", self.rowmin)
        print("self.colmin: ", self.colmin)
        return [full_map_pos[0] - self.rowmin, full_map_pos[1] - self.colmin]

    def _convert_cropped_map_pos_to_full_map_pos(self, cropped_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """cropped_map_pos: (row, col) in cropped map → (row, col) in full map."""
        return [cropped_map_pos[0] + self.rowmin, cropped_map_pos[1] + self.colmin]
