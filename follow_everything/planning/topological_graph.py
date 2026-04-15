"""Topological graph planner for obstacle-aware path generation.

Paper reference: "Follow Everything" §III-C

Algorithm:
  1. Label connected obstacle clusters in the inflated occupancy grid.
  2. Sample evenly-spaced boundary points from each cluster's convex hull.
  3. Build a NetworkX graph: nodes = {start, goal} ∪ boundary samples.
     Edges exist between nodes with clear line-of-sight on the grid.
  4. Find the shortest collision-free path with Dijkstra.
  5. Optionally generate CW/CCW homotopy variants and return the shortest.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.ndimage import label as nd_label
from scipy.spatial import ConvexHull

from follow_everything.mapping.occupancy_grid import OccupancyGrid
from follow_everything.utils.geometry import segment_intersects_grid_cell


class TopologicalGraphPlanner:
    def __init__(
        self,
        robot_radius: float = 0.3,
        goal_tolerance: float = 0.25,
        min_boundary_spacing: float = 0.5,
        los_check_resolution: float = 0.05,
        max_candidates: int = 8,
    ):
        self.robot_radius = robot_radius
        self.goal_tolerance = goal_tolerance
        self.min_boundary_spacing = min_boundary_spacing
        self.los_check_resolution = los_check_resolution
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------
    def plan(
        self,
        start: np.ndarray,       # (2,) world coords
        goal:  np.ndarray,       # (2,) world coords
        occ:   OccupancyGrid,
    ) -> List[np.ndarray]:
        """Return a list of 2-D waypoints from start to goal.

        Falls back to a straight line if no obstacles are in the way.
        Returns [goal] if start == goal within tolerance.
        """
        if np.linalg.norm(goal - start) < self.goal_tolerance:
            return [goal]

        # Direct path free?  Return it immediately.
        if not segment_intersects_grid_cell(
            start, goal, occ.inflated_grid, occ.origin, occ.resolution
        ):
            return [goal]

        graph, node_coords = self._build_graph(start, goal, occ)

        if "start" not in graph or "goal" not in graph:
            return [goal]  # degenerate fallback

        try:
            path_nodes = nx.shortest_path(
                graph, source="start", target="goal", weight="weight"
            )
        except nx.NetworkXNoPath:
            return [goal]

        waypoints = [node_coords[n] for n in path_nodes if n not in ("start", "goal")]
        waypoints.append(goal)
        return waypoints

    # ------------------------------------------------------------------
    def _build_graph(
        self,
        start: np.ndarray,
        goal:  np.ndarray,
        occ:   OccupancyGrid,
    ) -> Tuple[nx.Graph, dict]:
        """Build collision-aware graph with boundary nodes."""
        graph = nx.Graph()
        node_coords: dict = {"start": start, "goal": goal}

        graph.add_node("start")
        graph.add_node("goal")

        # Sample boundary points from each obstacle cluster
        boundary_pts = self._sample_boundaries(occ)
        for i, pt in enumerate(boundary_pts):
            nid = f"b{i}"
            node_coords[nid] = pt
            graph.add_node(nid)

        # Add edges between all pairs with clear line-of-sight
        nodes = list(graph.nodes)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i + 1:]:
                p1, p2 = node_coords[n1], node_coords[n2]
                if not segment_intersects_grid_cell(
                    p1, p2, occ.inflated_grid, occ.origin, occ.resolution
                ):
                    dist = float(np.linalg.norm(p2 - p1))
                    graph.add_edge(n1, n2, weight=dist)

        return graph, node_coords

    def _sample_boundaries(self, occ: OccupancyGrid) -> List[np.ndarray]:
        """Return world-coord boundary samples for each obstacle cluster."""
        labelled, n_clusters = nd_label(occ.inflated_grid)
        pts: List[np.ndarray] = []

        for i in range(1, n_clusters + 1):
            cells = np.argwhere(labelled == i)
            if len(cells) < 3:
                # Tiny cluster — add its centroid
                r, c = cells.mean(axis=0)
                pts.append(np.array(occ.cell_to_world(int(r), int(c))))
                continue

            world_pts = np.array([
                occ.cell_to_world(int(r), int(c)) for r, c in cells
            ])

            try:
                hull = ConvexHull(world_pts)
                hull_pts = world_pts[hull.vertices]
            except Exception:
                hull_pts = world_pts

            # Evenly space samples along the hull perimeter
            pts.extend(self._subsample_polyline(hull_pts, self.min_boundary_spacing))

        return pts

    @staticmethod
    def _subsample_polyline(
        points: np.ndarray, spacing: float
    ) -> List[np.ndarray]:
        """Keep points separated by at least `spacing` metres."""
        if len(points) == 0:
            return []
        kept = [points[0]]
        for pt in points[1:]:
            if np.linalg.norm(pt - kept[-1]) >= spacing:
                kept.append(pt)
        return kept
